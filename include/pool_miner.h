#pragma once

#include "mining_types.h"
#include <string>
#include <functional>
#include <thread>
#include <atomic>
#include <chrono>

#include "rapidjson/document.h"

namespace pastella {
namespace velora {
    class VeloraMiner;
}

/**
 * Pool Miner - Handles mining to a mining pool
 * Implements Stratum protocol for pool communication
 */
class PoolMiner {
public:
    PoolMiner();
    ~PoolMiner();

    // Initialize pool connection
    bool initialize(const std::string& poolUrl, int port,
                   const std::string& wallet, const std::string& workerName);

    // Connect to pool
    bool connect();

    // Disconnect from pool
    void disconnect();

    // Check pool connectivity
    bool isConnected() const;

    // Submit solution to pool
    bool submitSolution(u32 nonce, const std::string& hash);

    // Start pool mining
    bool startMining(velora::VeloraMiner& miner, const MinerConfig& config);

    // Stop pool mining
    void stopMining();

    // Get pool status
    std::string getStatus() const;

    // Get current difficulty from pool
    u32 getCurrentDifficulty() const;

private:
    std::string poolUrl;
    int port;
    std::string wallet;
    std::string workerName;
    bool connected;
    bool miningActive;
    bool waitingForBlock; // Flag to prevent further submissions during block processing
    u32 currentDifficulty;
    
    // Share counters
    std::atomic<int> acceptedShares;
    std::atomic<int> rejectedShares;
    
    // Request timing
    std::chrono::steady_clock::time_point lastSubmitTime;

    // Socket and networking
    int socket_;
    std::thread messageThread_;
    std::atomic<int> messageId_;

    // Miner reference
    velora::VeloraMiner* miner_;

    // Work management
    struct WorkData {
        std::string jobId;
        u32 height;
        std::string previousHash;
        std::string merkleRoot;
        u64 timestamp;
        u32 difficulty;        // Block difficulty from job
        u32 poolDifficulty;    // Pool difficulty for share finding
        u32 transactionCount;  // Number of transactions in block
    };

    WorkData currentWork;

    // Socket helpers
    void closeSocket();

    // Message handling
    bool sendMessage(const rapidjson::Document& message);
    std::string receiveMessage();
    void messageHandler();
    void processMessage(const std::string& message);

    // Stratum protocol
    bool subscribe();
    bool authorize();

    // Job handling
    void handleJobNotification(const rapidjson::Document& doc);
    void handleMiningNotify(const rapidjson::Document& doc);
    void handleSetDifficulty(const rapidjson::Document& doc);
    void handleResponse(const rapidjson::Document& doc);
    void handleError(const rapidjson::Document& doc);

    void startMiningJob();
    void onHashFound(const MiningResult& result);

    // Utility functions
    u32 calculateActualDifficulty(const std::string& hashHex);
};

} // namespace pastella
