#include "../../include/daemon/daemon_client.h"
#include "../../include/utils/logger.h"
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

namespace pastella {
namespace daemon {

// Static member initialization
bool DaemonClient::curlInitialized = false;

DaemonClient::DaemonClient() {
    if (!curlInitialized) {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        curlInitialized = true;
    }
}

DaemonClient::~DaemonClient() {
    // Note: We don't call curl_global_cleanup() here as other instances might still be using CURL
}

bool DaemonClient::checkConnectivity(const std::string& daemonUrl) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        LOG_ERROR_CAT("Failed to initialize CURL", "DAEMON");
        return false;
    }

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, (daemonUrl + "/api/blockchain/status").c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, CURL_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, CURL_CONNECT_TIMEOUT);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_HTTP_CODE, &http_code);
    curl_easy_cleanup(curl);

    if (!handleCurlError(res, "connectivity check")) {
        return false;
    }

    if (!handleHttpError(http_code, "connectivity check")) {
        return false;
    }

    LOG_INFO_CAT("Daemon connectivity check passed", "DAEMON");
    return true;
}

DaemonBlock DaemonClient::getLatestBlock(const std::string& daemonUrl, const std::string& apiKey) {
    DaemonBlock block;
    CURL* curl = curl_easy_init();
    if (!curl) {
        LOG_ERROR_CAT("Failed to initialize CURL for getting latest block", "DAEMON");
        return block;
    }

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, (daemonUrl + "/api/blockchain/latest").c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, CURL_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, CURL_CONNECT_TIMEOUT);

    // Add additional timeout settings to prevent hanging
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);  // 1 byte per second minimum
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 10L);  // for 10 seconds

    // Add API key header if provided
    struct curl_slist* headers = nullptr;
    if (!apiKey.empty()) {
        headers = curl_slist_append(headers, ("X-API-Key: " + apiKey).c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }

    LOG_DEBUG("Requesting latest block from: " + daemonUrl + "/api/blockchain/latest", "DAEMON");

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_HTTP_CODE, &http_code);

    if (headers) {
        curl_slist_free_all(headers);
    }
    curl_easy_cleanup(curl);

    if (!handleCurlError(res, "get latest block")) {
        LOG_ERROR_CAT("CURL error getting latest block: " + std::string(curl_easy_strerror(res)), "DAEMON");
        return block;
    }

    if (!handleHttpError(http_code, "get latest block")) {
        LOG_ERROR_CAT("HTTP error getting latest block: " + std::to_string(http_code), "DAEMON");
        return block;
    }

    LOG_DEBUG("Received response from daemon: " + response.substr(0, 100) + "...", "DAEMON");

    // Parse JSON response
    rapidjson::Document doc;
    if (doc.Parse(response.c_str()).HasParseError()) {
        LOG_ERROR_CAT("Failed to parse daemon response: " + response, "DAEMON");
        return block;
    }

    if (doc.HasMember("index")) block.index = doc["index"].GetInt();
    if (doc.HasMember("timestamp")) block.timestamp = doc["timestamp"].GetUint64();
    if (doc.HasMember("previousHash")) block.previousHash = doc["previousHash"].GetString();
    if (doc.HasMember("merkleRoot")) block.merkleRoot = doc["merkleRoot"].GetString();
    if (doc.HasMember("difficulty")) block.difficulty = doc["difficulty"].GetUint();
    if (doc.HasMember("hash")) block.hash = doc["hash"].GetString();

    LOG_DEBUG("Parsed block - index: " + std::to_string(block.index) +
              ", hash: " + block.hash.substr(0, 16) + "...", "DAEMON");

    return block;
}

bool DaemonClient::submitBlock(const std::string& jsonPayload, const std::string& daemonUrl, const std::string& apiKey) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        LOG_ERROR_CAT("Failed to initialize CURL for block submission", "DAEMON");
        return false;
    }

    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, (daemonUrl + "/api/blocks/submit").c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonPayload.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, SUBMISSION_TIMEOUT);

    // Add headers
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    if (!apiKey.empty()) {
        headers = curl_slist_append(headers, ("X-API-Key: " + apiKey).c_str());
    }
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_HTTP_CODE, &http_code);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (!handleCurlError(res, "block submission")) {
        return false;
    }

    if (!handleHttpError(http_code, "block submission")) {
        return false;
    }

    return true;
}

bool DaemonClient::getMiningTemplate(const std::string& daemonUrl,
                                     const std::string& apiKey,
                                     const std::string& address,
                                     DaemonBlock& outBlock) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        LOG_ERROR_CAT("Failed to initialize CURL for mining template", "DAEMON");
        return false;
    }

    std::string url = daemonUrl + "/api/mining/template?address=" + address;
    std::string response;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, CURL_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, CURL_CONNECT_TIMEOUT);

    struct curl_slist* headers = nullptr;
    if (!apiKey.empty()) {
        headers = curl_slist_append(headers, ("X-API-Key: " + apiKey).c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }

    LOG_DEBUG("Requesting mining template from: " + url, "DAEMON");
    CURLcode res = curl_easy_perform(curl);
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_HTTP_CODE, &http_code);
    if (headers) curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (!handleCurlError(res, "get mining template")) {
        return false;
    }
    if (!handleHttpError(http_code, "get mining template")) {
        return false;
    }

    rapidjson::Document doc;
    if (doc.Parse(response.c_str()).HasParseError() || !doc.IsObject()) {
        LOG_ERROR_CAT("Failed to parse mining template response", "DAEMON");
        return false;
    }

    // Fill outBlock
    outBlock.index = doc.HasMember("index") ? doc["index"].GetInt() : 0;
    outBlock.timestamp = doc.HasMember("timestamp") ? doc["timestamp"].GetUint64() : 0;
    outBlock.previousHash = doc.HasMember("previousHash") && doc["previousHash"].IsString() ? doc["previousHash"].GetString() : "";
    outBlock.merkleRoot = doc.HasMember("merkleRoot") && doc["merkleRoot"].IsString() ? doc["merkleRoot"].GetString() : "";
    outBlock.nonce = 0;
    outBlock.difficulty = doc.HasMember("difficulty") ? doc["difficulty"].GetUint() : 0;
    outBlock.hash.clear();
    outBlock.algorithm = "velora";
    outBlock.transactions.clear();
    if (doc.HasMember("transactions") && doc["transactions"].IsArray()) {
        for (auto& tx : doc["transactions"].GetArray()) {
            if (tx.HasMember("id") && tx["id"].IsString()) {
                outBlock.transactions.push_back(tx["id"].GetString());
            }
        }
    }
    return true;
}

void DaemonClient::setDaemonUrl(const std::string& url) {
    daemonUrl_ = url;
}

void DaemonClient::setApiKey(const std::string& apiKey) {
    apiKey_ = apiKey;
}

void DaemonClient::setCurlOptions(CURL* curl) {
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, CURL_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, CURL_CONNECT_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
}

void DaemonClient::setCurlHeaders(CURL* curl, bool isPost) {
    struct curl_slist* headers = nullptr;

    if (isPost) {
        headers = curl_slist_append(headers, "Content-Type: application/json");
    }

    if (!apiKey_.empty()) {
        headers = curl_slist_append(headers, ("X-API-Key: " + apiKey_).c_str());
    }

    if (headers) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }
}

bool DaemonClient::handleCurlError(CURLcode res, const std::string& operation) {
    if (res != CURLE_OK) {
        LOG_ERROR_CAT("Failed " + operation + ": " + std::string(curl_easy_strerror(res)), "DAEMON");
        return false;
    }
    return true;
}

bool DaemonClient::handleHttpError(long httpCode, const std::string& operation) {
    if (httpCode != 200) {
        LOG_ERROR_CAT("Daemon " + operation + " failed with HTTP code: " + std::to_string(httpCode), "DAEMON");
        return false;
    }
    return true;
}

// Static callback function
size_t DaemonClient::WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    userp->append((char*)contents, size * nmemb);
    return size * nmemb;
}

} // namespace daemon
} // namespace pastella
