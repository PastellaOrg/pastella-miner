#pragma once

#include "mining_types.h"
#include "daemon_miner.h"
#include "pool_miner.h"
#include <memory>
#include <functional>

namespace pastella {

// Forward declarations
class VeloraMiner;

/**
 * Mining Coordinator - Manages mining operations
 * Coordinates between pool mining, daemon mining, and the Velora miner
 */
class MiningCoordinator {
public:
    MiningCoordinator();
    ~MiningCoordinator() = default;

    // Initialize mining coordinator
    bool initialize(const MinerConfig& config);

    // Start mining based on configuration
    bool startMining(const MinerConfig& config);

    // Stop all mining operations
    void stopMining();

    // Check if mining is active
    bool isMining() const;

    // Get mining statistics
    struct MiningStats {
        u64 hashesProcessed;
        u64 hashesPerSecond;
        u32 currentDifficulty;
        std::string currentMode; // "pool" or "daemon"
        std::chrono::steady_clock::time_point startTime;
        std::chrono::steady_clock::time_point lastUpdate;
    };

    MiningStats getStats() const;

    // Set callbacks
    void setProgressCallback(std::function<void(u64, u64)> callback);
    void setHashFoundCallback(std::function<void(const MiningResult&)> callback);
    void setErrorCallback(std::function<void(const ErrorInfo&)> callback);

    // Handle hash found
    void onHashFound(const MiningResult& result);
    // Handle errors
    void onError(const ErrorInfo& error);

private:
    std::unique_ptr<DaemonMiner> daemonMiner;
    std::unique_ptr<PoolMiner> poolMiner;
    VeloraMiner* veloraMiner;

    MinerConfig currentConfig;
    bool miningActive;
    MiningStats stats;

    // Callbacks
    std::function<void(u64, u64)> progressCallback;
    std::function<void(const MiningResult&)> hashFoundCallback;
    std::function<void(const ErrorInfo&)> errorCallback;

    // Mining mode management
    bool startPoolMining();
    bool startDaemonMining();

    // Statistics management
    void updateStats();
    void resetStats();

    // Block header management
    struct BlockHeader;
    BlockHeader createBlockHeader(const DaemonBlock& latestBlock = DaemonBlock{});
};

} // namespace pastella
