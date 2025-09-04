#include "../include/pool_miner.h"
#include "../include/utils/logger.h"
#include "../include/velora/velora_miner.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#endif

#include <thread>
#include <chrono>
#include <sstream>
#include <regex>

using namespace pastella;
using namespace pastella::utils;

// ANSI Color codes
const std::string COLOR_GREEN = "\033[32m";
const std::string COLOR_RED = "\033[31m";
const std::string COLOR_WHITE_BOLD = "\033[1;37m";
const std::string COLOR_WHITE = "\033[37m";
const std::string COLOR_PURPLE = "\033[35m";
const std::string COLOR_DARK_GRAY = "\033[90m";
const std::string COLOR_RESET = "\033[0m";

PoolMiner::PoolMiner()
    : connected(false)
    , miningActive(false)
    , waitingForBlock(false)
    , currentDifficulty(1)
    , acceptedShares(0)
    , rejectedShares(0)
    , socket_(-1)
    , messageId_(0)
    , miner_(nullptr)
{
    // 🎯 CRITICAL FIX: Initialize currentWork with default values to prevent garbage data
    currentWork.jobId = "";
    currentWork.height = 0;
    currentWork.timestamp = 0; // Initialize to 0 to detect uninitialized usage
    currentWork.previousHash = "";
    currentWork.merkleRoot = "";
    currentWork.difficulty = 1;
    currentWork.poolDifficulty = 1;
    currentWork.transactionCount = 1;
    

#ifdef _WIN32
    WSADATA wsaData;
    WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
}

PoolMiner::~PoolMiner() {
    disconnect();
#ifdef _WIN32
    WSACleanup();
#endif
}

bool PoolMiner::initialize(const std::string& poolUrl, int port,
                          const std::string& wallet, const std::string& workerName) {
    this->poolUrl = poolUrl;
    this->port = port;
    this->wallet = wallet;
    this->workerName = workerName;


    return true;
}

bool PoolMiner::connect() {
    // Parse URL to extract host
    std::string host = poolUrl;
    if (host.find("stratum+tcp://") == 0) {
        host = host.substr(14); // Remove "stratum+tcp://" prefix
    }

    // Create socket
#ifdef _WIN32
    socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket_ == INVALID_SOCKET) {
        LOG_ERROR_CAT("Failed to create socket: " + std::to_string(WSAGetLastError()), "POOL");
        return false;
    }
#else
    socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_ < 0) {
        LOG_ERROR_CAT("Failed to create socket", "POOL");
        return false;
    }
#endif

    // Resolve hostname
    struct hostent* server = gethostbyname(host.c_str());
    if (server == nullptr) {
        LOG_ERROR_CAT("Failed to resolve hostname: " + host, "POOL");
        closeSocket();
        return false;
    }

    // Set up server address
    struct sockaddr_in serverAddr;
    memset(&serverAddr, 0, sizeof(serverAddr));
    serverAddr.sin_family = AF_INET;
    memcpy(&serverAddr.sin_addr.s_addr, server->h_addr, server->h_length);
    serverAddr.sin_port = htons(port);

    // Connect to server
    if (::connect(socket_, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
        LOG_ERROR_CAT("Failed to connect to pool server", "POOL");
        closeSocket();
        return false;
    }

    connected = true;
    LOG_INFO_CAT("Connected to pool: " + poolUrl + ":" + std::to_string(port), "POOL");

    // Start message handling thread
    messageThread_ = std::thread(&PoolMiner::messageHandler, this);

    return true;
}

void PoolMiner::disconnect() {
    if (connected) {
        connected = false;
        miningActive = false;

        if (messageThread_.joinable()) {
            messageThread_.join();
        }

        closeSocket();
    }
}

void PoolMiner::closeSocket() {
    if (socket_ != -1) {
#ifdef _WIN32
        closesocket(socket_);
#else
        close(socket_);
#endif
        socket_ = -1;
    }
}

bool PoolMiner::isConnected() const {
    return connected;
}

bool PoolMiner::sendMessage(const rapidjson::Document& message) {
    if (!connected) {
        return false;
    }

    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    message.Accept(writer);

    std::string jsonStr = buffer.GetString();
    jsonStr += "\n"; // Stratum protocol requires newline termination


#ifdef _WIN32
    int result = send(socket_, jsonStr.c_str(), jsonStr.length(), 0);
    if (result == SOCKET_ERROR) {
        LOG_ERROR_CAT("Failed to send message to pool: " + std::to_string(WSAGetLastError()), "POOL");
        return false;
    }
#else
    ssize_t result = send(socket_, jsonStr.c_str(), jsonStr.length(), 0);
    if (result < 0) {
        LOG_ERROR_CAT("Failed to send message to pool", "POOL");
        return false;
    }
#endif

    return true;
}

std::string PoolMiner::receiveMessage() {
    if (!connected) {
        return "";
    }

    char buffer[4096];

#ifdef _WIN32
    int bytesReceived = recv(socket_, buffer, sizeof(buffer) - 1, 0);
    if (bytesReceived == SOCKET_ERROR) {
        if (WSAGetLastError() != WSAEWOULDBLOCK) {
            LOG_ERROR_CAT("Failed to receive message from pool: " + std::to_string(WSAGetLastError()), "POOL");
            connected = false;
        }
        return "";
    }
#else
    ssize_t bytesReceived = recv(socket_, buffer, sizeof(buffer) - 1, 0);
    if (bytesReceived < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            LOG_ERROR_CAT("Failed to receive message from pool", "POOL");
            connected = false;
        }
        return "";
    }
#endif

    if (bytesReceived == 0) {
        LOG_INFO_CAT("Pool server closed connection", "POOL");
        connected = false;
        return "";
    }

    buffer[bytesReceived] = '\0';
    return std::string(buffer);
}

void PoolMiner::messageHandler() {
    std::string messageBuffer;

    while (connected) {
        std::string data = receiveMessage();
        if (data.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        messageBuffer += data;

        // Process complete messages (newline-separated)
        size_t pos = 0;
        while ((pos = messageBuffer.find('\n')) != std::string::npos) {
            std::string message = messageBuffer.substr(0, pos);
            messageBuffer.erase(0, pos + 1);

            if (!message.empty()) {
                processMessage(message);
            }
        }
    }
}

void PoolMiner::processMessage(const std::string& message) {

    rapidjson::Document doc;
    doc.Parse(message.c_str());

    if (doc.HasParseError()) {
        LOG_ERROR_CAT("Failed to parse JSON message from pool", "POOL");
        return;
    }

    // Handle different message types
    if (doc.HasMember("method")) {
        // Method call from server (like job notification)
        std::string method = doc["method"].GetString();

        if (method == "job") {
            handleJobNotification(doc);
        } else if (method == "mining.notify") {
            handleMiningNotify(doc);
        } else if (method == "mining.set_difficulty") {
            handleSetDifficulty(doc);
        }
    } else if (doc.HasMember("result")) {
        // Response to our request
        handleResponse(doc);
    } else if (doc.HasMember("error")) {
        // Error response
        handleError(doc);
    }
}

bool PoolMiner::subscribe() {
    rapidjson::Document request;
    request.SetObject();
    rapidjson::Document::AllocatorType& allocator = request.GetAllocator();

    int subId = ++messageId_;
    request.AddMember("id", subId, allocator);
    request.AddMember("method", "mining.subscribe", allocator);

    rapidjson::Value params(rapidjson::kArrayType);
    params.PushBack(rapidjson::Value("pastella-miner/1.0", allocator), allocator);
    request.AddMember("params", params, allocator);

    return sendMessage(request);
}

bool PoolMiner::authorize() {
    rapidjson::Document request;
    request.SetObject();
    rapidjson::Document::AllocatorType& allocator = request.GetAllocator();

    int authId = ++messageId_;
    request.AddMember("id", authId, allocator);
    request.AddMember("method", "login", allocator);

    rapidjson::Value params(rapidjson::kArrayType);
    params.PushBack(rapidjson::Value(wallet.c_str(), allocator), allocator);
    params.PushBack(rapidjson::Value("x", allocator), allocator); // Password (not used)
    request.AddMember("params", params, allocator);

    return sendMessage(request);
}

bool PoolMiner::submitSolution(u32 nonce, const std::string& hash) {

    rapidjson::Document request;
    request.SetObject();
    rapidjson::Document::AllocatorType& allocator = request.GetAllocator();

    request.AddMember("id", ++messageId_, allocator);
    request.AddMember("method", "submit", allocator);

    // Format nonce as 8-character hex string (pool requirement)
    char nonceHex[9];
    sprintf(nonceHex, "%08x", nonce);

    rapidjson::Value params(rapidjson::kObjectType);
    params.AddMember("id", rapidjson::Value(workerName.c_str(), allocator), allocator);
    params.AddMember("job_id", rapidjson::Value(currentWork.jobId.c_str(), allocator), allocator);
    params.AddMember("nonce", rapidjson::Value(nonceHex, allocator), allocator);
    params.AddMember("result", rapidjson::Value(hash.c_str(), allocator), allocator);

    request.AddMember("params", params, allocator);

    return sendMessage(request);
}

void PoolMiner::handleJobNotification(const rapidjson::Document& doc) {
    if (!doc.HasMember("params") || !doc["params"].IsObject()) {
        LOG_ERROR_CAT("Invalid job notification format", "POOL");
        return;
    }

    const auto& params = doc["params"];

    if (!params.HasMember("job_id") || !params.HasMember("height")) {
        LOG_ERROR_CAT("Missing required job parameters", "POOL");
        return;
    }

    // Update current work with structured data
    currentWork.jobId = params["job_id"].GetString();
    currentWork.height = params["height"].GetUint();

    // Extract structured job data with detailed logging
    if (params.HasMember("timestamp")) {
        u64 oldTimestamp = currentWork.timestamp;
        currentWork.timestamp = params["timestamp"].GetUint64();
    }
    if (params.HasMember("previous_hash")) currentWork.previousHash = params["previous_hash"].GetString();
    if (params.HasMember("merkle_root")) currentWork.merkleRoot = params["merkle_root"].GetString();
    if (params.HasMember("difficulty")) currentWork.difficulty = params["difficulty"].GetUint();
    if (params.HasMember("pool_difficulty")) currentWork.poolDifficulty = params["pool_difficulty"].GetUint();
    if (params.HasMember("transaction_count")) {
        currentWork.transactionCount = params["transaction_count"].GetUint();
    } else {
        currentWork.transactionCount = 1; // Default to 1 (coinbase transaction)
    }

    // Reset waiting flag for new job
    waitingForBlock = false;

    // Create formatted log line with colors:
    // "New job" in purple, "from" in dark gray, pool:port in white, "difficulty" and "height" in dark gray
    std::string poolInfo = poolUrl + ":" + std::to_string(port);
    std::string logLine = COLOR_PURPLE + "New job " + COLOR_RESET + 
                         COLOR_DARK_GRAY + "from " + COLOR_RESET +
                         COLOR_WHITE + poolInfo + COLOR_RESET + " " +
                         COLOR_DARK_GRAY + "difficulty " + COLOR_RESET + std::to_string(currentWork.poolDifficulty) + " " +
                         COLOR_DARK_GRAY + "height: " + COLOR_RESET + std::to_string(currentWork.height) + " " +
                         COLOR_DARK_GRAY + "(" + std::to_string(currentWork.transactionCount) + " tx)" + COLOR_RESET;
    
    LOG_INFO_CAT(logLine, "POOL");

    // Start mining with the structured data
    if (miner_ && miningActive) {
        startMiningJob();
    }
}



void PoolMiner::startMiningJob() {
    if (!miner_) {
        LOG_ERROR_CAT("No miner instance available", "POOL");
        return;
    }

    // 🎯 CRITICAL FIX: Prevent mining with uninitialized timestamp
    if (currentWork.timestamp == 0) {
        LOG_ERROR_CAT("Invalid job timestamp - mining stopped", "POOL");
        return;
    }

    // Create block header from current work
    BlockHeader header;
    header.index = currentWork.height;
    header.timestamp = currentWork.timestamp; // Already in milliseconds from pool
    header.previousHash = currentWork.previousHash;
    header.merkleRoot = currentWork.merkleRoot;
    header.difficulty = currentWork.difficulty; // Use block difficulty for Velora hash calculation
    header.nonce = 0;
    header.algorithm = "velora";

    // Stop current mining and start new job
    miner_->stopMining();
    miner_->updateBlockTemplate(header);
    miner_->setDifficulty(currentWork.poolDifficulty);
    if (!miner_->startMining()) {
        LOG_ERROR_CAT("Failed to start mining for job " + currentWork.jobId, "POOL");
    }
}

void PoolMiner::handleMiningNotify(const rapidjson::Document& doc) {
    handleJobNotification(doc);
}

void PoolMiner::handleSetDifficulty(const rapidjson::Document& doc) {
    if (!doc.HasMember("params") || !doc["params"].IsArray()) {
        LOG_ERROR_CAT("Invalid mining.set_difficulty format - missing params array", "POOL");
        return;
    }

    const auto& params = doc["params"];
    if (params.Size() < 1 || !params[0].IsUint()) {
        LOG_ERROR_CAT("Invalid mining.set_difficulty format - invalid difficulty value", "POOL");
        return;
    }

    u32 newDifficulty = params[0].GetUint();
    u32 oldDifficulty = currentDifficulty;
    
    // Update current difficulty
    currentDifficulty = newDifficulty;
    
    // Update pool difficulty in current work
    currentWork.poolDifficulty = newDifficulty;
    
    // Apply new difficulty to miner if active
    if (miner_ && miningActive) {
        miner_->setDifficulty(newDifficulty);
        LOG_INFO_CAT("Pool difficulty adjusted: " + std::to_string(oldDifficulty) + " -> " + 
                 std::to_string(newDifficulty), "POOL");
    } else {
        LOG_INFO_CAT("Pool difficulty updated: " + std::to_string(oldDifficulty) + " -> " + 
                 std::to_string(newDifficulty) + " (will apply when mining starts)", "POOL");
    }
}

void PoolMiner::handleResponse(const rapidjson::Document& doc) {

    if (doc.HasMember("id") && doc.HasMember("result")) {
        int id = doc["id"].GetInt();
        bool result = doc["result"].IsBool() ? doc["result"].GetBool() : false;


        // Handle specific responses based on message ID or content
        if (doc["result"].IsObject()) {
            const auto& resultObj = doc["result"];

            // Check for job first (login response)
            if (resultObj.HasMember("job")) {
                // Login response with job - extract job data directly
                const auto& job = resultObj["job"];

                if (job.HasMember("job_id") && job.HasMember("height")) {
                    // Update current work directly from login response
                    currentWork.jobId = job["job_id"].GetString();
                    currentWork.height = job["height"].GetUint();

                    // Extract structured job data with detailed logging
                    if (job.HasMember("timestamp")) {
                        u64 oldTimestamp = currentWork.timestamp;
                        currentWork.timestamp = job["timestamp"].GetUint64();
                    }
                    if (job.HasMember("previous_hash")) currentWork.previousHash = job["previous_hash"].GetString();
                    if (job.HasMember("merkle_root")) currentWork.merkleRoot = job["merkle_root"].GetString();
                    if (job.HasMember("difficulty")) currentWork.difficulty = job["difficulty"].GetUint();
                    if (job.HasMember("pool_difficulty")) currentWork.poolDifficulty = job["pool_difficulty"].GetUint();

                    // Reset waiting flag for new job
                    waitingForBlock = false;

                    // Start mining with the structured data
                    if (miner_ && miningActive) {
                        startMiningJob();
                    }
                    return; // Exit after processing job
                }
            }

            // Handle submit response with status (only if no job was processed)
            if (resultObj.HasMember("status")) {
                std::string status = resultObj["status"].GetString();

                if (status == "WAIT") {
                    // Block solution submitted log removed for cleaner output
                    waitingForBlock = true;
                    if (miner_) {
                        miner_->stopMining();
                    }
                } else if (status == "OK") {
                    acceptedShares++;
                    if (miner_) {
                        miner_->incrementAcceptedShares(0); // Update VeloraMiner share count (assume GPU 0)
                    }
                    auto now = std::chrono::steady_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSubmitTime).count();
                    
                    LOG_INFO_CAT(COLOR_GREEN + "Share accepted " + COLOR_WHITE_BOLD + "(" + 
                            std::to_string(acceptedShares.load()) + " / " + std::to_string(rejectedShares.load()) + ") " +
                            COLOR_RESET + "difficulty: " + std::to_string(currentWork.poolDifficulty) + " " +
                            COLOR_DARK_GRAY + "(" + std::to_string(duration) + "ms)" + COLOR_RESET, "POOL");
                }
                return;
            }
        }
    }
}

void PoolMiner::handleError(const rapidjson::Document& doc) {
    if (doc.HasMember("error") && doc["error"].IsArray()) {
        const auto& error = doc["error"];
        if (error.Size() >= 2) {
            int code = error[0].GetInt();
            std::string message = error[1].GetString();
            
            // Check if this is a share rejection
            if (message.find("share") != std::string::npos || message.find("Invalid") != std::string::npos) {
                rejectedShares++;
                if (miner_) {
                    miner_->incrementRejectedShares(0); // Update VeloraMiner share count (assume GPU 0)
                }
                auto now = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastSubmitTime).count();
                
                LOG_ERROR_CAT(COLOR_RED + "Share rejected " + COLOR_WHITE_BOLD + "(" + 
                         std::to_string(acceptedShares.load()) + " / " + std::to_string(rejectedShares.load()) + ") " +
                         COLOR_RESET + "difficulty: " + std::to_string(currentWork.poolDifficulty) + " " +
                         COLOR_DARK_GRAY + "(" + std::to_string(duration) + "ms)" + COLOR_RESET, "POOL");
            } else {
                LOG_ERROR_CAT("Pool error " + std::to_string(code) + ": " + message, "POOL");
            }
        }
    }
}

bool PoolMiner::startMining(velora::VeloraMiner& miner, const MinerConfig& config) {
    miner_ = &miner;

    // Set up callbacks
    miner.setHashFoundCallback([this](const MiningResult& result) {
        this->onHashFound(result);
    });

    miner.setErrorCallback([this](const ErrorInfo& error) {
        LOG_ERROR_CAT("Mining error: " + error.message, "POOL");
    });

    // Connect to pool
    if (!connect()) {
        return false;
    }

    // Subscribe and authorize
    if (!subscribe()) {
        LOG_ERROR_CAT("Failed to subscribe to pool", "POOL");
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    miningActive = true; // Set this before authorize so we can start mining on login response job

    if (!authorize()) {
        LOG_ERROR_CAT("Failed to authorize with pool", "POOL");
        return false;
    }

    LOG_INFO_CAT("Pool mining started", "POOL");

    return true;
}

void PoolMiner::stopMining() {
    miningActive = false;
    if (miner_) {
        miner_->stopMining();
    }
    disconnect();
}

void PoolMiner::onHashFound(const MiningResult& result) {
    // Convert hash vector to hex string
    std::string hashHex;
    for (const auto& byte : result.hash) {
        char hex[3];
        sprintf(hex, "%02x", byte);
        hashHex += hex;
    }
    

    // Calculate what difficulty this hash actually meets
    u32 actualDifficulty = calculateActualDifficulty(hashHex);
    bool meetsBlockDifficulty = actualDifficulty >= currentWork.difficulty;

    // Validate timestamp matches current job before submission
    if (result.timestamp != currentWork.timestamp) {
        rejectedShares++;
        if (miner_) {
            miner_->incrementRejectedShares(); // Update VeloraMiner share count
        }
        LOG_ERROR_CAT(COLOR_RED + "Share rejected " + COLOR_WHITE_BOLD + "(" + 
                 std::to_string(acceptedShares.load()) + " / " + std::to_string(rejectedShares.load()) + ") " +
                 COLOR_RESET + "difficulty: " + std::to_string(currentWork.poolDifficulty) + " " +
                 COLOR_DARK_GRAY + "(0ms)" + COLOR_RESET, "POOL");
        return;
    }

    if (meetsBlockDifficulty) {
        LOG_INFO_CAT("BLOCK SOLUTION! Nonce: " + std::to_string(result.nonce), "POOL");
    }

    // Check if we're waiting for block processing
    if (waitingForBlock) {
        return;
    }

    // Submit solution to pool
    lastSubmitTime = std::chrono::steady_clock::now();
    submitSolution(result.nonce, hashHex);
}

std::string PoolMiner::getStatus() const {
    if (connected && miningActive) {
        return "Mining on job " + currentWork.jobId;
    } else if (connected) {
        return "Connected, waiting for work";
    } else {
        return "Disconnected";
    }
}

u32 PoolMiner::getCurrentDifficulty() const {
    return currentDifficulty;
}



u32 PoolMiner::calculateActualDifficulty(const std::string& hashHex) {
    if (hashHex.empty()) {
        return 0;
    }

    try {
        // Count leading zeros in the hash
        int leadingZeros = 0;
        for (char c : hashHex) {
            if (c == '0') {
                leadingZeros++;
            } else {
                break;
            }
        }

        // Simple difficulty estimation based on leading zeros
        // Each leading zero roughly doubles the difficulty
        u32 estimatedDifficulty = 1 << (leadingZeros / 2);

        // For more precise calculation, convert hash to big integer and compare
        // This is a simplified version - the pool will do the exact calculation
        return estimatedDifficulty;
    } catch (const std::exception& e) {
        LOG_ERROR_CAT("Error calculating actual difficulty from hash " + hashHex + ": " + e.what(), "POOL");
        return 0;
    }
}
