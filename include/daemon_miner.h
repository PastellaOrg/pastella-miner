#pragma once

#include "mining_types.h"
#include "types.h"
#include <string>
#include <functional>

namespace pastella {
namespace velora {
    class VeloraMiner; // Forward declaration
}

/**
 * Daemon Miner - Handles mining to a local daemon
 * Communicates with the Pastella daemon via HTTP API
 */
class DaemonMiner {
public:
    DaemonMiner();
    ~DaemonMiner() = default;

    // Initialize daemon connection
    bool initialize(const std::string& daemonUrl, const std::string& apiKey);

    // Check daemon connectivity
    bool checkConnectivity();

    // Get latest block from daemon
    DaemonBlock getLatestBlock();

    // Submit mined block to daemon
    bool submitBlock(const MiningResult& result);

    // Stop daemon mining
    bool stopMining();

    // Check if daemon mining is active
    bool isMining() const;

private:
    std::string daemonUrl;
    std::string apiKey;
    bool isConnected;
    bool miningActive;

    // HTTP communication helpers
    static size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp);
    bool performHttpRequest(const std::string& endpoint, std::string& response,
                           const std::string& postData = "", const std::string& method = "GET");

    // Current block state
    u64 currentBlockIndex_;
    Hash256 previousHash_;
    Hash256 merkleRoot_;
};

} // namespace pastella
