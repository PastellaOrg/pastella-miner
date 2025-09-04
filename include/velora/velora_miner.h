#pragma once

#include "../types.h"
#include "../mining_types.h"
#include "velora_algorithm.h"
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <mutex>


namespace pastella {
namespace velora {

/**
 * Velora Miner - Manages the mining process
 * Supports both CPU and GPU mining with automatic fallback
 */
class VeloraMiner {
public:
    // Callback types
    using HashFoundCallback = std::function<void(const MiningResult&)>;
    using ProgressCallback = std::function<void(u64 hashesProcessed, u64 hashesPerSecond)>;
    using ErrorCallback = std::function<void(const ErrorInfo&)>;

    VeloraMiner();
    ~VeloraMiner();

    // Disable copy constructor and assignment
    VeloraMiner(const VeloraMiner&) = delete;
    VeloraMiner& operator=(const VeloraMiner&) = delete;

    // Configuration
    void setBlockHeader(const BlockHeader& header);
    void setDifficulty(u32 difficulty);
    void setGPUConfig(const GPUConfig& config);
    void setUseGPU(bool useGPU);

    // Callbacks
    void setHashFoundCallback(HashFoundCallback callback);
    void setProgressCallback(ProgressCallback callback);
    void setErrorCallback(ErrorCallback callback);

    // Mining control
    bool startMining();
    void stopMining();
    bool isMining() const;

    // Mining parameters
    void setStartNonce(u64 startNonce);
    void setMaxNonces(u64 maxNonces);
    void setTargetHash(const Hash256& targetHash);
    void setNumThreads(u32 numThreads);
    void setCPUEnabled(bool enabled);

    // Status and statistics
    PerformanceMetrics getPerformanceMetrics() const;
    u64 getCurrentNonce() const;
    u64 getHashesProcessed() const;
    u64 getHashesPerSecond() const;
    std::vector<u64> getPerThreadHashesProcessed() const;

    // 🎯 ROLLING WINDOW HASH RATE: Get current smooth hashrate
    double getCurrentHashrate() const;
    
    // Record batch completion for accurate hashrate calculation
    void recordBatchCompletion(u64 batchSize) const;
    
    // 🎯 ACCURATE BATCH TIMING: Track GPU batch start/end for precise hashrate
    void startBatchTiming(size_t gpuIndex, u64 noncesPerBatch) const;
    void endBatchTiming(size_t gpuIndex) const;

    // 🎯 MULTI-GPU SUPPORT: Get individual GPU hashrates
    std::vector<double> getIndividualGPUHashrates() const;
    
    // Share tracking
    void incrementAcceptedShares(int gpuId = -1);
    void incrementRejectedShares(int gpuId = -1);
    u64 getAcceptedShares() const;
    u64 getRejectedShares() const;
    std::vector<u64> getPerGPUAcceptedShares() const;
    std::vector<u64> getPerGPURejectedShares() const;
    
    // Hashrate display
    void displayHashrate() const;
    void startPeriodicHashrateDisplay();
    void stopPeriodicHashrateDisplay();

    // 🎯 PER-GPU HASH COUNTING: Update hashes processed by specific GPU
    void updateGPUHashCount(size_t gpuIndex, u64 hashesProcessed);

    // Daemon mining support
    bool isDaemonMining() const;
    void setDaemonMining(bool enabled);
    u64 getCurrentBlockIndex() const;
    Hash256 getPreviousHash() const;
    Hash256 getMerkleRoot() const;

    // Block header access
    BlockHeader getCurrentBlockHeader() const;

    // Error handling
    ErrorInfo getLastError() const;
    void clearError();

    // 🎯 MULTI-GPU SUPPORT: GPU configuration and management
    bool initializeGPU();
    void cleanupGPU();
    bool isGPUAvailable() const;

    // Multi-GPU support
    void setMultiGPUConfig(const std::vector<GPUConfig>& configs);
    std::vector<GPUConfig> getMultiGPUConfig() const;
    int getActiveGPUCount() const;
    u64 getTotalBatchSize() const; // Combined batch size across all GPUs

    // Utility methods
    void reset();
    void updateBlockHeader(const BlockHeader& header);

    // 🔒 COORDINATED MINING: Update block template and reset mining state
    void updateBlockTemplate(const BlockHeader& newHeader);

    // Expose resolved GPU config (after auto-tune)
    GPUConfig getGPUConfig() const { return gpuConfig_; }

private:
    // Mining state
    std::atomic<bool> mining_;
    std::atomic<bool> shouldStop_;
    std::atomic<u64> currentNonce_;
    std::atomic<u64> hashesProcessed_;
    
    // Share tracking
    std::atomic<u64> acceptedShares_;
    std::atomic<u64> rejectedShares_;
    std::vector<std::unique_ptr<std::atomic<u64>>> perGPUAcceptedShares_;
    std::vector<std::unique_ptr<std::atomic<u64>>> perGPURejectedShares_;

    // Configuration
    BlockHeader blockHeader_;
    u32 difficulty_;
    u64 startNonce_;
    u64 maxNonces_;
    Hash256 targetHash_;
    GPUConfig gpuConfig_;
    bool useGPU_;
    bool cpuEnabled_;
    u32 numThreads_;

    // 🎯 MULTI-GPU SUPPORT: Multiple GPU configurations and state
    std::vector<GPUConfig> multiGPUConfigs_;
    std::vector<std::thread> gpuMiningThreads_; // One thread per GPU
    std::vector<u64> gpuCurrentNonces_; // Current nonce for each GPU (non-atomic for now)
    std::atomic<u64> totalHashesProcessed_; // Combined hashes from all GPUs

    // 🎯 PER-GPU HASH COUNTING: Track hashes processed by each GPU
    std::vector<std::unique_ptr<std::atomic<u64>>> perGPUHashesProcessed_; // Per-GPU hash counters

    // Daemon mining state
    bool daemonMining_;
    u64 currentBlockIndex_;
    Hash256 previousHash_;
    Hash256 merkleRoot_;

    // Algorithm instance
    std::unique_ptr<VeloraAlgorithm> algorithm_;
    // 🎯 MULTI-GPU: Separate algorithm instance for each GPU to avoid context sharing
    std::vector<std::unique_ptr<VeloraAlgorithm>> gpuAlgorithms_;
    // Per-thread counters for hashrate per core stats
    std::vector<u64> perThreadHashes_;

    // Mining threads
    std::vector<std::thread> miningThreads_;
    std::thread progressThread_;
    
    // Hashrate display thread
    std::thread hashrateDisplayThread_;
    std::atomic<bool> displayHashrateActive_;

    // Callbacks
    HashFoundCallback hashFoundCallback_;
    ProgressCallback progressCallback_;
    ErrorCallback errorCallback_;

    // Performance tracking
    mutable PerformanceMetrics metrics_;
    std::chrono::steady_clock::time_point startTime_;



    // 🔒 COORDINATED MINING SYSTEM: Prevent duplicate block submissions
    std::atomic<u64> currentBlockTemplateVersion_;
    std::atomic<bool> blockFoundByAnySystem_;

    // 🎯 ROLLING WINDOW HASH RATE SYSTEM: Smooth out GPU batch processing spikes
    mutable std::vector<std::pair<u64, u64>> hashrateSamples_; // (timestamp, hashes) pairs
    mutable std::mutex hashrateMutex_;
    static constexpr size_t MAX_SAMPLES = 30; // 30 seconds of data (30 samples × 1 second)
    static constexpr u64 SAMPLE_INTERVAL_MS = 1000; // 1 second per sample for better responsiveness

    // 🎯 PER-GPU HASH RATE TRACKING: Individual hashrate tracking per GPU
    mutable std::vector<std::vector<std::pair<u64, u64>>> perGPUHashrateSamples_; // Per-GPU (timestamp, hashes) pairs
    mutable std::vector<std::unique_ptr<std::mutex>> perGPUHashrateMutexes_; // Per-GPU mutexes for thread safety

    // 🎯 ACCURATE BATCH TIMING: Track precise GPU batch performance
    mutable std::vector<u64> perGPUBatchStartTimes_;    // When current batch started (ms)
    mutable std::vector<u64> perGPUTotalBatches_;       // Total completed batches per GPU
    mutable std::vector<u64> perGPUTotalTimeMs_;        // Total time spent mining (ms) per GPU
    mutable std::vector<u64> perGPUNoncesPerBatch_;     // Nonces per batch for each GPU
    mutable std::mutex batchTimingMutex_;



    // Error tracking
    mutable ErrorInfo lastError_;

    // Internal methods
    void miningWorker(u32 threadIndex);
    void progressWorker();
    void updateProgress();

    // GPU mining methods
    void gpuMiningWorker();
    void gpuMiningWorker(size_t gpuIndex); // 🎯 MULTI-GPU: Worker for specific GPU
    bool processGPUResults();

    // CPU mining methods
    void cpuMiningWorker();
    bool processCPUResults();

    // GPU detection methods
    bool detectCUDA();
    bool detectOpenCL();
    bool detectDirectX();

    // Utility methods
    void setError(ErrorCode code, const std::string& message, const std::string& details = "");
    void updatePerformanceMetrics(u64 hashesProcessed, u64 timeMs);
    bool meetsDifficulty(const Hash256& hash) const;

    // Constants
    static constexpr u32 DEFAULT_DIFFICULTY = 1000;
    static constexpr u64 DEFAULT_MAX_NONCES = 1000000;
    static constexpr u32 PROGRESS_UPDATE_INTERVAL_MS = 1000;
    static constexpr u32 DEFAULT_NUM_THREADS = 4;
};

} // namespace velora
} // namespace pastella
