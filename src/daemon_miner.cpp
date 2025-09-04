#include "../include/daemon_miner.h"
#include "../include/mining_types.h"
#include "../include/utils/logger.h"
#include "../include/utils/crypto_utils.h"
#include "../include/velora/velora_miner.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <chrono>
#include <thread>
#include <curl/curl.h>

namespace pastella {

DaemonMiner::DaemonMiner() : isConnected(false), miningActive(false) {}

bool DaemonMiner::initialize(const std::string& daemonUrl, const std::string& apiKey) {
    this->daemonUrl = daemonUrl;
    this->apiKey = apiKey;

    LOG_INFO_CAT("Initializing daemon miner", "DAEMON");
    std::string apiKeyMsg = "API Key provided: ";
    apiKeyMsg += (apiKey.empty() ? "NO" : "YES");
    LOG_INFO_CAT(apiKeyMsg, "DAEMON");

    // Check connectivity
    LOG_INFO_CAT("Checking daemon connectivity...", "DAEMON");
    if (!checkConnectivity()) {
        LOG_ERROR_CAT("Failed to connect to daemon during initialization", "DAEMON");
        return false;
    }

    isConnected = true;
    LOG_INFO_CAT("Daemon miner initialized successfully", "DAEMON");
    return true;
}

bool DaemonMiner::checkConnectivity() {
    CURL* curl = curl_easy_init();
    if (!curl) {
        LOG_ERROR_CAT("Failed to initialize CURL", "DAEMON");
        return false;
    }

    std::string url = daemonUrl + "/api/blockchain/status";

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &DaemonMiner::writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 3L);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 3L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "pastella-miner/1.0.0");
    curl_easy_setopt(curl, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_1_1);
    curl_easy_setopt(curl, CURLOPT_FORBID_REUSE, 1L);
    curl_easy_setopt(curl, CURLOPT_FRESH_CONNECT, 1L);

    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_HTTP_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::string errorMsg = "Failed to connect to daemon: ";
        errorMsg += curl_easy_strerror(res);
        LOG_ERROR_CAT(errorMsg, "DAEMON");
        return false;
    }

    if (http_code != 200) {
        LOG_ERROR_CAT("Daemon returned HTTP code: " + std::to_string(http_code), "DAEMON");
        return false;
    }

    return true;
}

DaemonBlock DaemonMiner::getLatestBlock() {
    DaemonBlock block;

    std::string response;
    if (!performHttpRequest("/api/blockchain/latest", response)) {
        LOG_ERROR_CAT("Failed to get latest block from daemon", "DAEMON");
        return block;
    }

    // Parse JSON response
    rapidjson::Document doc;
    if (doc.Parse(response.c_str()).HasParseError()) {
        LOG_ERROR_CAT("Failed to parse daemon response", "DAEMON");
        return block;
    }

    if (doc.HasMember("index")) block.index = doc["index"].GetInt();
    if (doc.HasMember("timestamp")) block.timestamp = doc["timestamp"].GetUint64();
    if (doc.HasMember("previousHash")) block.previousHash = doc["previousHash"].GetString();
    if (doc.HasMember("merkleRoot")) block.merkleRoot = doc["merkleRoot"].GetString();
    if (doc.HasMember("difficulty")) block.difficulty = doc["difficulty"].GetUint();
    if (doc.HasMember("hash")) block.hash = doc["hash"].GetString();

    return block;
}

bool DaemonMiner::submitBlock(const MiningResult& result) {
    if (!isConnected) {
        LOG_ERROR_CAT("Cannot submit block - daemon not connected", "DAEMON");
        return false;
    }

    // Create JSON payload for block submission
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();

    doc.AddMember("blockIndex", currentBlockIndex_ + 1, allocator);
    doc.AddMember("nonce", result.nonce, allocator);
    doc.AddMember("hash", rapidjson::Value(utils::CryptoUtils::hashToHex(result.hash).c_str(), allocator), allocator);
    doc.AddMember("previousHash", rapidjson::Value(utils::CryptoUtils::hashToHex(result.hash).c_str(), allocator), allocator);
    doc.AddMember("merkleRoot", rapidjson::Value(utils::CryptoUtils::hashToHex(result.hash).c_str(), allocator), allocator);
    doc.AddMember("difficulty", result.difficulty, allocator);
    doc.AddMember("timestamp", result.timestamp, allocator);

    // Convert to JSON string
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    std::string jsonPayload = buffer.GetString();

    // Submit to daemon
    std::string response;
    bool accepted = performHttpRequest("/api/blocks/submit", response, jsonPayload, "POST");

    if (accepted) {
        LOG_INFO_CAT("Block hash: " + utils::CryptoUtils::hashToHex(result.hash), "DAEMON");
    } else {
        // Parse error response if available
        rapidjson::Document responseDoc;
        if (!responseDoc.Parse(response.c_str()).HasParseError() && responseDoc.HasMember("error")) {
            LOG_ERROR_CAT("Error: " + std::string(responseDoc["error"].GetString()), "DAEMON");
        }
    }

    return accepted;
}

bool DaemonMiner::stopMining() {
    if (miningActive) {
        LOG_INFO_CAT("Stopping daemon mining", "DAEMON");
        miningActive = false;
    }
    return true;
}

bool DaemonMiner::isMining() const {
    return miningActive;
}

size_t DaemonMiner::writeCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;

    if (totalSize > 0 && totalSize < 1000000) { // Sanity check: less than 1MB
        std::string* response = static_cast<std::string*>(userp);
        response->append(static_cast<char*>(contents), totalSize);
    } else {
        LOG_ERROR_CAT("WriteCallback received invalid size: " + std::to_string(totalSize), "DAEMON");
        return 0; // Signal error
    }

    return totalSize;
}

bool DaemonMiner::performHttpRequest(const std::string& endpoint, std::string& response, const std::string& postData, const std::string& method) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        return false;
    }

    std::string url = daemonUrl + endpoint;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, &DaemonMiner::writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

    // Set headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (!apiKey.empty()) {
        std::string authHeader = "Authorization: Bearer " + apiKey;
        headers = curl_slist_append(headers, authHeader.c_str());
    }

    if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        if (!postData.empty()) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, postData.c_str());
        }
    }

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_HTTP_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        LOG_ERROR_CAT("HTTP request failed: " + std::string(curl_easy_strerror(res)), "DAEMON");
        return false;
    }

    if (http_code != 200) {
        LOG_ERROR_CAT("HTTP request returned code: " + std::to_string(http_code), "DAEMON");
        return false;
    }

    return true;
}

} // namespace pastella
