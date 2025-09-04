#pragma once

#include "../mining_types.h"
#include "../types.h"
#include <memory>
#include <thread>
#include <atomic>
#include <functional>

namespace pastella {
namespace mining {

/**
 * GPU Miner - Handles GPU-based mining operations
 * Manages CUDA kernels and GPU memory
 */
class GPUMiner {
public:
    // Callback types
    using HashFoundCallback = std::function<void(const MiningResult&)>;
    using ProgressCallback = std::function<void(u64 hashesProcessed, u64 hashesPerSecond)>;
    using ErrorCallback = std::function<void(const ErrorInfo&)>;

    GPUMiner();
    ~GPUMiner();

    // Disable copy constructor and assignment
    GPUMiner(const GPUMiner&) = delete;
    GPUMiner& operator=(const GPUMiner&) = delete;

    // Configuration
    void setBlockHeader(const BlockHeader& header);
    void setDifficulty(u32 difficulty);
    void setMaxNonces(u64 maxNonces);
    void setGPUConfig(const GPUConfig& config);

    // Callbacks
    void setHashFoundCallback(HashFoundCallback callback);
    void setProgressCallback(ProgressCallback callback);
    void setErrorCallback(ErrorCallback callback);

    // Mining control
    bool startMining();
    void stopMining();
    bool isMining() const;

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
    // Mining state
    std::atomic<bool> mining_;
    std::atomic<bool> shouldStop_;
    std::atomic<u64> currentNonce_;
    std::atomic<u64> hashesProcessed_;

    // Configuration
    BlockHeader blockHeader_;
    u32 difficulty_;
    u64 startNonce_;
    u64 maxNonces_;
    GPUConfig gpuConfig_;

    // Algorithm instance
    std::unique_ptr<class VeloraAlgorithm> algorithm_;

    // Mining threads
    std::thread gpuMiningThread_;
    std::thread progressThread_;

    // Callbacks
    HashFoundCallback hashFoundCallback_;
    ProgressCallback progressCallback_;
    ErrorCallback errorCallback_;

    // Performance tracking
    mutable PerformanceMetrics metrics_;
    std::chrono::steady_clock::time_point startTime_;

    // Error tracking
    mutable ErrorInfo lastError_;

    // Internal methods
    void gpuMiningWorker();
    void progressWorker();
    void updateProgress();
    bool processGPUResults();
    bool detectCUDA();
    void setError(ErrorCode code, const std::string& message, const std::string& details = "");
    void updatePerformanceMetrics(u64 hashesProcessed, u64 timeMs);
    bool meetsDifficulty(const Hash256& hash) const;

    // Constants
    static constexpr u32 DEFAULT_DIFFICULTY = 1000;
    static constexpr u64 DEFAULT_MAX_NONCES = 1000000;
    static constexpr u32 PROGRESS_UPDATE_INTERVAL_MS = 1000;
};

} // namespace mining
} // namespace pastella
