#pragma once

#include "../mining_types.h"
#include "cpu_miner.h"
#include "gpu_miner.h"
#include <memory>
#include <vector>

namespace pastella {
namespace mining {

/**
 * Mining Manager - Coordinates between CPU and GPU miners
 * Manages mining modes and coordinates mining operations
 */
class MiningManager {
public:
    MiningManager();
    ~MiningManager();

    // Configuration
    void setBlockHeader(const BlockHeader& header);
    void setDifficulty(u32 difficulty);
    void setMaxNonces(u64 maxNonces);
    void setCPUThreads(u32 numThreads);
    void setGPUConfig(const GPUConfig& config);

    // Mining control
    bool startMining();
    void stopMining();
    bool isMining() const;

    // Set callbacks
    void setHashFoundCallback(CPUMiner::HashFoundCallback callback);
    void setProgressCallback(CPUMiner::ProgressCallback callback);
    void setErrorCallback(CPUMiner::ErrorCallback callback);

    // Status and statistics
    PerformanceMetrics getPerformanceMetrics() const;
    u64 getCurrentNonce() const;
    u64 getHashesProcessed() const;
    u64 getHashesPerSecond() const;

    // Block header access
    BlockHeader getCurrentBlockHeader() const;

    // Error handling
    ErrorInfo getLastError() const;
    void clearError();

    // GPU management
    bool initializeGPU();
    void cleanupGPU();
    bool isGPUAvailable() const;

    // Utility methods
    void reset();
    void updateBlockHeader(const BlockHeader& header);

private:
    // Mining instances
    std::unique_ptr<CPUMiner> cpuMiner;
    std::unique_ptr<GPUMiner> gpuMiner;

    // Configuration
    BlockHeader blockHeader_;
    u32 difficulty_;
    u64 maxNonces_;
    u32 numCPUThreads_;
    GPUConfig gpuConfig_;

    // Internal methods
    void setError(ErrorCode code, const std::string& message, const std::string& details = "");

    // Constants
    static constexpr u32 DEFAULT_DIFFICULTY = 1000;
    static constexpr u64 DEFAULT_MAX_NONCES = 1000000;
    static constexpr u32 DEFAULT_CPU_THREADS = 4;
};

} // namespace mining
} // namespace pastella
