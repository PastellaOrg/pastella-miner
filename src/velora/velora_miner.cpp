#include "../../include/velora/velora_miner.h"
#include "../../include/utils/crypto_utils.h"
#include "../../include/utils/logger.h"
#include "../../include/utils/mining_utils.h"
#include <chrono>
#include <iostream>
#include <atomic>
#include <thread>
#include <sstream>
#include <iomanip>

namespace pastella {
namespace velora {

VeloraMiner::VeloraMiner()
    : difficulty_(DEFAULT_DIFFICULTY), startNonce_(0), maxNonces_(DEFAULT_MAX_NONCES),
      useGPU_(false), cpuEnabled_(true), mining_(false), shouldStop_(false), numThreads_(DEFAULT_NUM_THREADS),
      acceptedShares_(0), rejectedShares_(0), displayHashrateActive_(false),
      daemonMining_(false), currentBlockIndex_(0) {

    LOG_DEBUG("VeloraMiner constructor called - difficulty: " + std::to_string(difficulty_) +
             ", maxNonces: " + std::to_string(maxNonces_) +
             ", numThreads: " + std::to_string(numThreads_), "VELORA");

    // Initialize atomic variables properly
    currentNonce_.store(0);
    hashesProcessed_.store(0);



    // 🔒 Initialize coordinated mining system
    currentBlockTemplateVersion_.store(0);
    blockFoundByAnySystem_.store(false);

    // 🎯 Initialize rolling window hashrate system
    hashrateSamples_.clear();



    LOG_DEBUG("Atomic variables initialized - currentNonce: 0, hashesProcessed: 0", "VELORA");

    algorithm_ = std::make_unique<VeloraAlgorithm>();
    LOG_DEBUG("VeloraAlgorithm instance created", "VELORA");
    startTime_ = std::chrono::steady_clock::now();
    LOG_DEBUG("Start time initialized", "VELORA");
    perThreadHashes_.clear();
}

VeloraMiner::~VeloraMiner() {
    LOG_DEBUG("VeloraMiner destructor called", "VELORA");
    stopMining();
    cleanupGPU();
}

void VeloraMiner::setBlockHeader(const BlockHeader& header) {
    LOG_DEBUG("Setting block header - index: " + std::to_string(header.index) +
             ", difficulty: " + std::to_string(header.difficulty) +
             ", previousHash: " + header.previousHash.substr(0, 16) + "...", "VELORA");
    blockHeader_ = header;
}

void VeloraMiner::setDifficulty(u32 difficulty) {
    LOG_DEBUG("Setting difficulty: " + std::to_string(difficulty), "VELORA");
    difficulty_ = difficulty;
}

void VeloraMiner::setGPUConfig(const GPUConfig& config) {
        LOG_DEBUG("Setting GPU config - deviceId: " + std::to_string(config.deviceId) +
             ", threadsPerBlock: " + std::to_string(config.threadsPerBlock) +
             ", blocksPerGrid: " + std::to_string(config.blocksPerGrid), "VELORA");
    gpuConfig_ = config;
    if (algorithm_) {
        algorithm_->setGPUConfig(config);
    }
}

void VeloraMiner::setUseGPU(bool useGPU) {
    LOG_DEBUG("Setting GPU usage: " + std::to_string(useGPU), "VELORA");
    useGPU_ = useGPU;
    if (algorithm_) {
        // Ensure GPU config is set in algorithm before enabling GPU
        algorithm_->setGPUConfig(gpuConfig_);
        algorithm_->setUseGPU(useGPU);
    }
}

void VeloraMiner::setHashFoundCallback(HashFoundCallback callback) {
    LOG_DEBUG("Setting hash found callback", "VELORA");
    hashFoundCallback_ = callback;
}

void VeloraMiner::setProgressCallback(ProgressCallback callback) {
    LOG_DEBUG("Setting progress callback", "VELORA");
    progressCallback_ = callback;
}

void VeloraMiner::setErrorCallback(ErrorCallback callback) {
    LOG_DEBUG("Setting error callback", "VELORA");
    errorCallback_ = callback;
}

bool VeloraMiner::startMining() {
    if (mining_) {
        LOG_DEBUG("Cannot start mining - already mining", "VELORA");
        return false;
    }

    try {
        LOG_DEBUG("Starting mining - resetting state", "VELORA");
        
        // 🎯 CRITICAL FIX: Ensure template data is fully synchronized before starting mining
        std::atomic_thread_fence(std::memory_order_seq_cst);
        
        
        mining_ = true;
        shouldStop_ = false;

        // 🔒 COORDINATED MINING: Reset mining state for new block
        blockFoundByAnySystem_.store(false);

        LOG_DEBUG("Mining state set - mining_: " + std::to_string(mining_.load()) +
                 ", shouldStop_: " + std::to_string(shouldStop_.load()), "VELORA");
        currentNonce_.store(startNonce_);
        hashesProcessed_.store(0);
        LOG_DEBUG("Mining state initialized - startNonce: " + std::to_string(startNonce_) +
                 ", currentNonce: " + std::to_string(currentNonce_.load()) +
                 ", hashesProcessed: " + std::to_string(hashesProcessed_.load()), "VELORA");

        startTime_ = std::chrono::steady_clock::now();
        LOG_DEBUG("Start time updated for new mining session", "VELORA");

        // Check if GPU mode is requested and available
        if (useGPU_) {
            LOG_DEBUG("GPU mode requested", "VELORA");
            // Ensure GPU config is passed to algorithm before initialization
            if (algorithm_) {
                algorithm_->setGPUConfig(gpuConfig_);
            }

            // 🎯 GPU REINITIALIZATION: Ensure clean CUDA state for new mining session
            for (size_t i = 0; i < gpuAlgorithms_.size(); i++) {
                if (gpuAlgorithms_[i]) {
                    LOG_DEBUG("Reinitializing GPU " + std::to_string(i) + " for new mining session", "VELORA");
                    // Clean up any existing GPU state
                    gpuAlgorithms_[i]->cleanupGPU();
                    // Reinitialize with fresh CUDA context
                    if (!gpuAlgorithms_[i]->initializeGPU()) {
                        LOG_ERROR_CAT("Failed to reinitialize GPU " + std::to_string(i), "VELORA");
                        return false;
                    }
                }
            }

            // Don't initialize GPU here - let the algorithm handle it when needed
        } else {
            LOG_DEBUG("CPU mode requested", "VELORA");
        }

        // 🚀 SIMULTANEOUS GPU + CPU MINING: Start both for maximum performance
        if (useGPU_) {
            // 🎯 MULTI-GPU: Launch one mining thread per GPU
            LOG_DEBUG("Starting " + std::to_string(multiGPUConfigs_.size()) + " GPU mining workers", "VELORA");
            for (size_t i = 0; i < multiGPUConfigs_.size(); i++) {
                miningThreads_.emplace_back([this, i]() { this->gpuMiningWorker(i); });
                LOG_DEBUG("Started GPU mining worker " + std::to_string(i) + " for Device " +
                         std::to_string(multiGPUConfigs_[i].deviceId), "VELORA");
            }
        }

        // Start CPU mining threads only if CPU mining is enabled
        if (cpuEnabled_) {
            LOG_DEBUG("Starting " + std::to_string(numThreads_) + " CPU mining threads", "VELORA");
            miningThreads_.reserve(numThreads_);
            perThreadHashes_.assign(numThreads_, 0);
            for (u32 i = 0; i < numThreads_; i++) {
                try {
                    miningThreads_.emplace_back(&VeloraMiner::miningWorker, this, i);
                } catch (const std::exception& e) {
                    setError(ErrorCode::MINING_FAILED, "Failed to start CPU mining thread", e.what());
                }
            }
        } else {
            LOG_DEBUG("CPU mining disabled - not starting CPU threads", "VELORA");
        }

        // Start progress thread
        LOG_DEBUG("Starting progress thread", "VELORA");
        try {
            progressThread_ = std::thread(&VeloraMiner::progressWorker, this);
        } catch (const std::exception& e) {
            setError(ErrorCode::MINING_FAILED, "Failed to start progress thread", e.what());
            mining_ = false;
            shouldStop_ = true;
            return false;
        }

        // Start periodic hashrate display
        startPeriodicHashrateDisplay();

        return true;

    } catch (const std::exception& e) {
        setError(ErrorCode::MINING_FAILED, "Failed to start mining", e.what());
        LOG_DEBUG("Error in startMining - clearing mining state", "VELORA");
        mining_ = false;
        return false;
    }
}

void VeloraMiner::stopMining() {
    // Even if not actively mining, ensure all threads are joined safely
    if (!mining_) {
        LOG_DEBUG("Stop requested while not mining - ensuring threads are joined", "VELORA");
    } else {
        LOG_DEBUG("🚨 STOPPING MINING - setting shouldStop flag (called from external code)", "VELORA");
        // Add stack trace or caller identification for debugging
        LOG_DEBUG("🔍 stopMining() called - this indicates pool job change or external stop", "VELORA");
    }
    shouldStop_ = true;
    mining_ = false;
    LOG_DEBUG("Mining state cleared - mining_: " + std::to_string(mining_.load()) +
             ", shouldStop_: " + std::to_string(shouldStop_.load()), "VELORA");

    // Stop periodic hashrate display
    stopPeriodicHashrateDisplay();

    // Wait for mining threads to finish
    LOG_DEBUG("Waiting for " + std::to_string(miningThreads_.size()) + " mining threads to finish", "VELORA");
    for (auto& thread : miningThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    miningThreads_.clear();
    LOG_DEBUG("All mining threads finished and cleared", "VELORA");

    // Wait for progress thread to finish
    LOG_DEBUG("Waiting for progress thread to finish", "VELORA");
    if (progressThread_.joinable()) {
        progressThread_.join();
        LOG_DEBUG("Progress thread joined", "VELORA");
    } else {
        LOG_DEBUG("No joinable progress thread", "VELORA");
    }
}

bool VeloraMiner::isMining() const {
    bool result = mining_;
    return result;
}

void VeloraMiner::setStartNonce(u64 startNonce) {
    LOG_DEBUG("Setting start nonce: " + std::to_string(startNonce), "VELORA");
    startNonce_ = startNonce;
}

void VeloraMiner::setMaxNonces(u64 maxNonces) {
    LOG_DEBUG("Setting max nonces: " + std::to_string(maxNonces), "VELORA");
    maxNonces_ = maxNonces;
}

void VeloraMiner::setNumThreads(u32 numThreads) {
    LOG_DEBUG("Setting number of threads: " + std::to_string(numThreads), "VELORA");
    numThreads_ = numThreads;
}

void VeloraMiner::setCPUEnabled(bool enabled) {
    LOG_DEBUG("Setting CPU enabled: " + std::to_string(enabled), "VELORA");
    cpuEnabled_ = enabled;
}

void VeloraMiner::setTargetHash(const Hash256& targetHash) {
    LOG_DEBUG("Setting target hash", "VELORA");
    targetHash_ = targetHash;
}

PerformanceMetrics VeloraMiner::getPerformanceMetrics() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime_);

    metrics_.totalTimeMs = duration.count();
    metrics_.hashesProcessed = hashesProcessed_.load();

    if (duration.count() > 0) {
        metrics_.hashesPerSecond = (hashesProcessed_.load() * 1000) / duration.count();
        metrics_.averageTimePerHashMs = duration.count() / (hashesProcessed_.load() ? hashesProcessed_.load() : 1);
    }

    LOG_DEBUG("getPerformanceMetrics() called - hashesProcessed: " + std::to_string(metrics_.hashesProcessed) +
             ", totalTimeMs: " + std::to_string(metrics_.totalTimeMs) +
             ", hashesPerSecond: " + std::to_string(metrics_.hashesPerSecond), "VELORA");

    return metrics_;
}

u64 VeloraMiner::getCurrentNonce() const {
    u64 result = currentNonce_.load();
    return result;
}

u64 VeloraMiner::getHashesProcessed() const {
    // Simple atomic read - no mutex to avoid deadlock
    u64 currentHashes = hashesProcessed_.load();
    LOG_DEBUG("getHashesProcessed() called - returning: " + std::to_string(currentHashes), "VELORA");
    return currentHashes;
}

u64 VeloraMiner::getHashesPerSecond() const {
    // 🎯 OPTIMIZED HASH RATE CALCULATION: Use rolling window for smooth hashrate
    double currentHashrate = getCurrentHashrate();
    u64 result = static_cast<u64>(currentHashrate);

    // Fallback to performance metrics if rolling window not ready
    if (result == 0) {
        auto metrics = getPerformanceMetrics();
        result = metrics.hashesPerSecond;
    }

    LOG_DEBUG("getHashesPerSecond() called - returning: " + std::to_string(result), "VELORA");
    return result;
}

double VeloraMiner::getCurrentHashrate() const {
    try {
        if (!mining_.load()) {
            return 0.0; // Not mining
        }
        
        // 🎯 PROPER BATCH-BASED CALCULATION: (completedBatches × noncesPerBatch) / totalRunningTime
        double totalHashrate = 0.0;
        
        for (size_t gpuIndex = 0; gpuIndex < perGPUTotalBatches_.size(); gpuIndex++) {
            u64 completedBatches = perGPUTotalBatches_[gpuIndex];
            u64 totalRunningTimeMs = perGPUTotalTimeMs_[gpuIndex];
            u64 noncesPerBatch = perGPUNoncesPerBatch_[gpuIndex];
            
            if (completedBatches > 0 && totalRunningTimeMs > 0 && noncesPerBatch > 0) {
                // Calculate: (batches × nonces per batch) / running time in seconds
                u64 totalNonces = completedBatches * noncesPerBatch;
                double gpuHashrate = static_cast<double>(totalNonces) * 1000.0 / static_cast<double>(totalRunningTimeMs);
                totalHashrate += gpuHashrate;
                
            }
        }
        
        return totalHashrate;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception in getCurrentHashrate(): " + std::string(e.what()), "VELORA");
        return 0.0;
    } catch (...) {
        LOG_ERROR("Unknown exception in getCurrentHashrate()", "VELORA");
        return 0.0;
    }
}

// Record batch completion for hashrate calculation
void VeloraMiner::recordBatchCompletion(u64 batchSize) const {
    // This method is kept for compatibility but new timing system is more accurate
    LOG_DEBUG("Legacy batch completion record: " + std::to_string(batchSize) + " hashes", "VELORA");
}

// 🎯 ACCURATE BATCH TIMING: Start timing a GPU batch
void VeloraMiner::startBatchTiming(size_t gpuIndex, u64 noncesPerBatch) const {
    try {
        std::lock_guard<std::mutex> lock(batchTimingMutex_);
        
        if (gpuIndex < perGPUBatchStartTimes_.size()) {
            auto now = std::chrono::steady_clock::now();
            u64 currentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
            
            perGPUBatchStartTimes_[gpuIndex] = currentTimeMs;
            perGPUNoncesPerBatch_[gpuIndex] = noncesPerBatch;
            
            LOG_DEBUG("GPU" + std::to_string(gpuIndex) + " batch started: " + std::to_string(noncesPerBatch) + " nonces at " + std::to_string(currentTimeMs), "VELORA");
        }
    } catch (...) {
        // Silent fail to avoid disrupting mining
    }
}

// 🎯 ACCURATE BATCH TIMING: End timing a GPU batch
void VeloraMiner::endBatchTiming(size_t gpuIndex) const {
    try {
        std::lock_guard<std::mutex> lock(batchTimingMutex_);
        
        if (gpuIndex < perGPUBatchStartTimes_.size() && perGPUBatchStartTimes_[gpuIndex] > 0) {
            auto now = std::chrono::steady_clock::now();
            u64 currentTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
            
            u64 batchDuration = currentTimeMs - perGPUBatchStartTimes_[gpuIndex];
            
            // Update totals
            perGPUTotalBatches_[gpuIndex]++;
            perGPUTotalTimeMs_[gpuIndex] += batchDuration;
            
            
            // Reset start time
            perGPUBatchStartTimes_[gpuIndex] = 0;
        }
    } catch (...) {
        // Silent fail to avoid disrupting mining
    }
}

std::vector<u64> VeloraMiner::getPerThreadHashesProcessed() const {
    // Snapshot per-thread counters
    return perThreadHashes_;
}

ErrorInfo VeloraMiner::getLastError() const {
    LOG_DEBUG("getLastError() called", "VELORA");
    return lastError_;
}

void VeloraMiner::clearError() {
    LOG_DEBUG("Clearing last error", "VELORA");
    lastError_ = ErrorInfo{};
}

bool VeloraMiner::initializeGPU() {
    LOG_DEBUG("Initializing GPU", "VELORA");
    try {
        // Basic GPU detection for Windows
        #ifdef _WIN32
        // Try to detect CUDA first
        if (detectCUDA()) {
            LOG_DEBUG("CUDA detection successful", "VELORA");
            return true;
        }

        // Try to detect OpenCL
        if (detectOpenCL()) {
            LOG_DEBUG("OpenCL detection successful", "VELORA");
            return true;
        }

        // Try to detect DirectX compute
        if (detectDirectX()) {
            return true;
        }
        #endif

        return false;

    } catch (const std::exception&) {
        return false;
    }
}

bool VeloraMiner::detectCUDA() {
    LOG_DEBUG("Detecting CUDA", "VELORA");
    // TODO: Implement CUDA detection
    // For now, just check if CUDA libraries are available
    LOG_DEBUG("CUDA detection not implemented, returning false", "VELORA");
    return false;
}

bool VeloraMiner::detectOpenCL() {
    LOG_DEBUG("Detecting OpenCL", "VELORA");
    // TODO: Implement OpenCL detection
    // For now, just check if OpenCL libraries are available
    LOG_DEBUG("OpenCL detection not implemented, returning false", "VELORA");
    return false;
}

bool VeloraMiner::detectDirectX() {
    LOG_DEBUG("Detecting DirectX", "VELORA");
    try {
        // Basic DirectX detection using Windows APIs
        #ifdef _WIN32
        // For now, simulate DirectX detection
        // In a real implementation, we would:
        // 1. Load d3d11.dll
        // 2. Check for compute shader support
        // 3. Enumerate adapters

        // 🎯 RESPECT CONFIG: Don't override deviceId from config - let setMultiGPUConfig handle it
        // Only set defaults for threads/blocks if not already configured
        if (gpuConfig_.threadsPerBlock == 0) gpuConfig_.threadsPerBlock = 256;
        if (gpuConfig_.blocksPerGrid == 0) gpuConfig_.blocksPerGrid = 1024;
        // gpuConfig_.maxNonces is already set by user, don't override it
        gpuConfig_.useDoublePrecision = false;

                LOG_DEBUG("DirectX detection successful - deviceId: " + std::to_string(gpuConfig_.deviceId) +
                 ", threadsPerBlock: " + std::to_string(gpuConfig_.threadsPerBlock) +
                 ", blocksPerGrid: " + std::to_string(gpuConfig_.blocksPerGrid), "VELORA");
        return true;
        #else
        LOG_DEBUG("DirectX detection not available on this platform", "VELORA");
        return false;
        #endif

    } catch (const std::exception&) {
        return false;
    }
}

void VeloraMiner::cleanupGPU() {
    LOG_DEBUG("Cleaning up GPU", "VELORA");
    // TODO: Implement GPU cleanup
}

bool VeloraMiner::isGPUAvailable() const {
    // Check if GPU was successfully initialized
    bool result = useGPU_ && gpuConfig_.deviceId >= 0;
    LOG_DEBUG("isGPUAvailable() called - useGPU_: " + std::to_string(useGPU_) +
             ", gpuConfig_.deviceId: " + std::to_string(gpuConfig_.deviceId) +
             ", returning: " + std::to_string(result), "VELORA");
    return result;
}

void VeloraMiner::reset() {
    LOG_DEBUG("Resetting miner state", "VELORA");
    stopMining();
    currentNonce_.store(0);
    hashesProcessed_.store(0);



    // 🔒 Reset coordinated mining system
    blockFoundByAnySystem_.store(false);

    perThreadHashes_.clear();
    clearError();
    if (algorithm_) {
        algorithm_->resetPerformanceMetrics();
    }
    LOG_DEBUG("Miner state reset complete", "VELORA");
}

void VeloraMiner::updateBlockHeader(const BlockHeader& header) {
    LOG_DEBUG("Updating block header - index: " + std::to_string(header.index) +
             ", difficulty: " + std::to_string(header.difficulty) +
             ", previousHash: " + header.previousHash.substr(0, 16) + "...", "VELORA");
    blockHeader_ = header;
}

void VeloraMiner::updateBlockTemplate(const BlockHeader& newHeader) {
    LOG_DEBUG("🔒 COORDINATED MINING: Updating block template - index: " + std::to_string(newHeader.index) +
             ", previous: " + std::to_string(blockHeader_.index), "VELORA");

    // 🔒 IMMEDIATELY STOP ALL MINING to prevent duplicate submissions
    shouldStop_ = true;
    blockFoundByAnySystem_.store(true);
    
    // 🎯 CRITICAL FIX: Wait longer for all mining threads to fully stop before updating template
    // This prevents race conditions where old mining threads use new template data
    std::this_thread::sleep_for(std::chrono::milliseconds(200)); // Longer wait to ensure GPU batches complete
    
    // 🎯 FORCE STOP: Additional safety check to ensure mining has completely stopped
    int waitCount = 0;
    while (mining_.load() && waitCount < 50) { // Wait up to 5 seconds
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        waitCount++;
    }
    
    if (mining_.load()) {
        LOG_ERROR_CAT("🚨 CRITICAL: Mining did not stop properly after template update!", "VELORA");
    }

    // Increment block template version to invalidate all ongoing work
    currentBlockTemplateVersion_.fetch_add(1);

    // 🎯 CRITICAL FIX: Use memory fence to ensure template version update is visible
    std::atomic_thread_fence(std::memory_order_seq_cst);
    
    // 🎯 ADDITIONAL SAFETY: Brief pause after version increment
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Update block header with new template
    blockHeader_ = newHeader;
    difficulty_ = newHeader.difficulty;
    
    // 🎯 CRITICAL FIX: Another memory fence to ensure header update is fully visible
    std::atomic_thread_fence(std::memory_order_seq_cst);
    
    // 🎯 FINAL SAFETY: Brief pause to ensure all updates are visible across threads
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // 🎯 CRITICAL BUG FIX: Reset shouldStop flag to allow new mining to start
    // This was the root cause of immediate mining stoppage
    shouldStop_ = false;
    blockFoundByAnySystem_.store(false);

    // Reset mining state for new block (but NOT batch timing data!)
    currentNonce_.store(0);
    // DON'T reset hashesProcessed_ or timing - we want to track cumulative performance

    LOG_DEBUG("🔒 COORDINATED MINING: Block template updated, ready for new mining session for block " +
             std::to_string(newHeader.index), "VELORA");
}

// Private methods
void VeloraMiner::miningWorker(u32 threadIndex) {
    try {
        LOG_DEBUG("Mining worker started - max nonces: " + std::to_string(maxNonces_), "VELORA");

        // Use a thread-local algorithm instance to avoid data races in shared state
        std::unique_ptr<VeloraAlgorithm> localAlgorithm = std::make_unique<VeloraAlgorithm>();
        // Propagate GPU config flag in case generateHash uses shared parameters (CPU path uses generateHash)
        localAlgorithm->setUseGPU(false);

        while (!shouldStop_) {
            // 🔒 COORDINATED MINING: IMMEDIATE STOP if block found by any system
            if (blockFoundByAnySystem_.load()) {
                LOG_DEBUG("🔒 CPU mining stopped - block found by another system", "VELORA");
                break;
            }

            // Debug logging every 1000 nonces to track progress
            if (currentNonce_.load() % 1000 == 0) {
                LOG_DEBUG("Mining progress - nonce: " + std::to_string(currentNonce_.load()) +
                         ", shouldStop: " + std::to_string(shouldStop_.load()), "VELORA");
            }

            u64 nonce = currentNonce_.fetch_add(1);
            
            // 🎯 ATOMIC CAPTURE: Capture template data atomically to avoid race conditions
            u64 currentTemplateVersion = currentBlockTemplateVersion_.load();
            
            // 🎯 CRITICAL FIX: Double-check template version before and after capture
            std::atomic_thread_fence(std::memory_order_seq_cst); // Ensure we see latest changes
            
            u32 cpuBlockIndex = blockHeader_.index;
            u64 cpuTimestamp = blockHeader_.timestamp;
            std::string cpuPrevHash = blockHeader_.previousHash;
            std::string cpuMerkleRoot = blockHeader_.merkleRoot;
            u32 cpuDifficulty = blockHeader_.difficulty;
            
            // 🎯 CRITICAL TEMPLATE VERSION CHECK: Skip if template changed during capture
            u64 finalTemplateVersion = currentBlockTemplateVersion_.load();
            if (finalTemplateVersion != currentTemplateVersion) {
                LOG_DEBUG("🎯 TEMPLATE CHANGED during CPU capture - skipping nonce " + std::to_string(nonce) + 
                         " (version changed from " + std::to_string(currentTemplateVersion) + " to " + std::to_string(finalTemplateVersion) + ")", "VELORA");
                continue; // Skip this nonce and try again with new template
            }
            
            // 🎯 VALIDATION: Log what timestamp we're actually going to use (only every 10000 nonces to avoid spam)
            if (nonce % 10000 == 0) {
                LOG_DEBUG("🎯 CPU USING VALIDATED TIMESTAMP: " + std::to_string(cpuTimestamp) + 
                         " (template version: " + std::to_string(currentTemplateVersion) + ", nonce: " + std::to_string(nonce) + ")", "VELORA");
            }
            
            // 🎯 STORE TEMPLATE VERSION: Store with nonce data for result filtering
            u64 nonceTemplateVersion = currentTemplateVersion;

            // Always use CPU path in this worker; GPU mining uses gpuMiningWorker()
            Hash256 hash = localAlgorithm->generateHash(
                cpuBlockIndex,
                nonce,
                cpuTimestamp,
                utils::CryptoUtils::hexToHash(cpuPrevHash),
                utils::CryptoUtils::hexToHash(cpuMerkleRoot),
                cpuDifficulty
            );
            
            // Get accumulator for debugging - use executeMemoryWalk from the same algorithm
            std::vector<u32> pattern = localAlgorithm->generateMemoryPattern(
                cpuBlockIndex, nonce, cpuTimestamp,
                utils::CryptoUtils::hexToHash(cpuPrevHash),
                utils::CryptoUtils::hexToHash(cpuMerkleRoot),
                cpuDifficulty
            );
            u32 accumulator = localAlgorithm->executeMemoryWalk(pattern, nonce, cpuTimestamp);

            hashesProcessed_.fetch_add(1);
            LOG_DEBUG("CPU mining hash completed - total hashes now: " + std::to_string(hashesProcessed_.load()), "VELORA");
            // Update per-thread counter
            if (threadIndex < perThreadHashes_.size()) {
                perThreadHashes_[threadIndex]++;
            }

            // Check if hash meets difficulty
            if (meetsDifficulty(hash)) {
                // 🎯 TEMPLATE VERSION CHECK: Filter out results from outdated template versions
                if (currentBlockTemplateVersion_.load() != nonceTemplateVersion) {
                    LOG_DEBUG("🎯 DISCARDING CPU RESULT: Template version changed from " + std::to_string(nonceTemplateVersion) + 
                             " to " + std::to_string(currentBlockTemplateVersion_.load()) + " for nonce " + std::to_string(nonce), "VELORA");
                    continue; // Discard this result and continue with next nonce
                }
                
                LOG_DEBUG("Valid hash found with nonce: " + std::to_string(nonce), "VELORA");

                MiningResult result;
                result.nonce = nonce;
                result.hash.assign(hash.begin(), hash.end());
                result.difficulty = difficulty_;
                result.isValid = true;
                // 🎯 CRITICAL FIX: Use the EXACT timestamp used during mining, not current time
                result.timestamp = cpuTimestamp;
                // 🎯 DEBUGGING: Add accumulator value for comparison with daemon
                result.accumulator = accumulator;

                // 🎯 MINING SYSTEM IDENTIFICATION: Mark as CPU-found block
                result.miningSystem = "CPU";
                result.gpuId = -1; // CPU mining

                if (hashFoundCallback_) {
                    hashFoundCallback_(result);
                }

                // 🔒 COORDINATED MINING: For daemon mining, stop all mining when block found
                // For pool mining, continue mining to find more shares
                if (daemonMining_) {
                    blockFoundByAnySystem_.store(true);
                    shouldStop_ = true;
                    LOG_DEBUG("🔒 CPU mining worker stopping due to block found - DAEMON MODE", "VELORA");
                    break;
                } else {
                    LOG_DEBUG("🔒 CPU mining worker found share - POOL MODE, continuing", "VELORA");
                    // Continue mining for more shares
                }
            }
        }

        LOG_DEBUG("Mining worker finished - shouldStop: " + std::to_string(shouldStop_.load()) +
            ", currentNonce: " + std::to_string(currentNonce_.load()), "VELORA");

        // Set mining to false when worker finishes
        mining_ = false;
        LOG_DEBUG("Mining worker set mining_ to false", "VELORA");

    } catch (const std::exception& e) {
        setError(ErrorCode::MINING_FAILED, "Mining worker failed", e.what());
        mining_ = false; // Also set to false on error
    }
}

void VeloraMiner::progressWorker() {
    LOG_DEBUG("Progress worker started", "VELORA");
    while (!shouldStop_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(PROGRESS_UPDATE_INTERVAL_MS));

        auto now = std::chrono::steady_clock::now();

        // 🎯 OPTIMIZED MULTI-GPU HASH RATE TRACKING: More efficient per-GPU hashrate updates
        if (useGPU_ && !multiGPUConfigs_.empty()) {
            u64 nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count();

            for (size_t gpuIndex = 0; gpuIndex < multiGPUConfigs_.size(); gpuIndex++) {
                if (gpuIndex < perGPUHashrateSamples_.size() && gpuIndex < perGPUHashrateMutexes_.size()) {
                    // Minimize lock time by reading hash count first
                    u64 currentHashes = perGPUHashesProcessed_[gpuIndex]->load();

                    {
                        std::lock_guard<std::mutex> lock(*perGPUHashrateMutexes_[gpuIndex]);
                        auto& samples = perGPUHashrateSamples_[gpuIndex];
                        samples.emplace_back(nowMs, currentHashes);

                        // Keep only recent samples for rolling window (more efficient cleanup)
                        while (samples.size() > MAX_SAMPLES) {
                            samples.erase(samples.begin());
                        }
                    }
                }
            }
        }

        if (progressCallback_) {
            auto hps = getHashesPerSecond();
            progressCallback_(hashesProcessed_.load(), hps);
        }
    }
    LOG_DEBUG("Progress worker finished", "VELORA");
}

void VeloraMiner::updateProgress() {
    // This method is called periodically to update progress
    // Implementation can be expanded for more detailed progress tracking
}

void VeloraMiner::gpuMiningWorker(size_t gpuIndex) {
    try {
        // 🎯 MULTI-GPU NONCE DISTRIBUTION: Each GPU gets a unique nonce range
        const u32 BATCH_SIZE = multiGPUConfigs_[gpuIndex].maxNonces;
        const size_t totalGPUs = multiGPUConfigs_.size();

        // Calculate nonce starting point for this GPU to prevent overlap
        u64 gpuStartNonce = startNonce_ + (gpuIndex * BATCH_SIZE);

        LOG_DEBUG("GPU " + std::to_string(gpuIndex) + " (Device " +
                 std::to_string(multiGPUConfigs_[gpuIndex].deviceId) +
                 ") mining nonces starting from " + std::to_string(gpuStartNonce), "VELORA");

        // 🚀 MAXIMUM PERFORMANCE: Use full batch size for maximum GPU utilization
        // Full batches = maximum work per kernel launch = 100% GPU usage
        const u32 MAX_PERFORMANCE_BATCH_SIZE = BATCH_SIZE; // Use full batch size
        std::vector<std::vector<u64>> nonceBatchQueue;
        nonceBatchQueue.reserve(16); // Keep 16 batches ready for continuous processing

        LOG_DEBUG("MAXIMUM PERFORMANCE: Using full batch size " + std::to_string(MAX_PERFORMANCE_BATCH_SIZE) +
                 " for 100% GPU utilization", "VELORA");

        while (!shouldStop_) {
            // 🔒 COORDINATED MINING: IMMEDIATE STOP if block found by any system
            if (blockFoundByAnySystem_.load()) {
                LOG_DEBUG("🔒 GPU " + std::to_string(gpuIndex) + " mining stopped - block found by another system", "VELORA");
                break;
            }

            // 🚀 MAXIMUM PERFORMANCE: Pre-generate full nonce batches for 100% GPU utilization
            while (nonceBatchQueue.size() < 16 && !shouldStop_) {
                std::vector<u64> nonceBatch;
                nonceBatch.resize(MAX_PERFORMANCE_BATCH_SIZE);

                // Calculate starting nonce for this batch
                u64 batchStartNonce = gpuStartNonce + gpuCurrentNonces_[gpuIndex] + (nonceBatchQueue.size() * MAX_PERFORMANCE_BATCH_SIZE * totalGPUs);

                // 🚀 VECTORIZED NONCE GENERATION: Use arithmetic progression for maximum speed
                u64 step = totalGPUs;
                u64 endNonce = batchStartNonce + (MAX_PERFORMANCE_BATCH_SIZE * step);

                // Fill the vector using pointer arithmetic for maximum performance
                u64* noncePtr = nonceBatch.data();
                for (u64 nonce = batchStartNonce; nonce < endNonce; nonce += step) {
                    *noncePtr++ = nonce;
                }

                nonceBatchQueue.push_back(std::move(nonceBatch));
            }

            // 🚀 ULTRA-PERFORMANCE OPTIMIZATION: Minimal logging for maximum mining speed
            static int batchLogCounter = 0;
            if (++batchLogCounter % 50 == 0) { // Log every 50th batch for maximum performance
                LOG_DEBUG("GPU " + std::to_string(gpuIndex) + " mining batch - currentNonce: " +
                         std::to_string(gpuCurrentNonces_[gpuIndex]) +
                         ", shouldStop: " + std::to_string(shouldStop_.load()) +
                         ", batch: " + std::to_string(batchLogCounter) +
                         ", performance: " + std::to_string(BATCH_SIZE) + " nonces/batch" +
                         ", queue: " + std::to_string(nonceBatchQueue.size()) + " batches ready", "VELORA");
            }

            // 🎯 MAXIMUM PERFORMANCE: Get next batch from queue for 100% GPU utilization
            if (nonceBatchQueue.empty()) {
                // Queue is empty, generate a single full batch immediately
                std::vector<u64> nonceBatch;
                nonceBatch.resize(MAX_PERFORMANCE_BATCH_SIZE);

                // Calculate starting nonce for this batch
                u64 batchStartNonce = gpuStartNonce + gpuCurrentNonces_[gpuIndex];

                // 🚀 VECTORIZED NONCE GENERATION: Use arithmetic progression for maximum speed
                u64 step = totalGPUs;
                u64 endNonce = batchStartNonce + (MAX_PERFORMANCE_BATCH_SIZE * step);

                // Fill the vector using pointer arithmetic for maximum performance
                u64* noncePtr = nonceBatch.data();
                for (u64 nonce = batchStartNonce; nonce < endNonce; nonce += step) {
                    *noncePtr++ = nonce;
                }

                nonceBatchQueue.push_back(std::move(nonceBatch));
            }

            // Get the next batch from the queue
            std::vector<u64> nonceBatch = std::move(nonceBatchQueue.front());
            nonceBatchQueue.erase(nonceBatchQueue.begin());

            if (nonceBatch.empty()) break;

            // 🔒 COORDINATED MINING: IMMEDIATE STOP if block found by any system
            if (blockFoundByAnySystem_.load()) {
                LOG_DEBUG("🔒 GPU " + std::to_string(gpuIndex) + " mining stopped - block found by another system", "VELORA");
                break;
            }

            // 🎯 ATOMIC CAPTURE: Capture the exact template data used for this batch atomically
            // Check template version to ensure we're using current data
            u64 currentTemplateVersion = currentBlockTemplateVersion_.load();
            
            // 🎯 CRITICAL FIX: Double-check template version before and after capture
            std::atomic_thread_fence(std::memory_order_seq_cst); // Ensure we see latest changes
            
            u32 batchBlockIndex = blockHeader_.index;
            u64 batchTimestamp = blockHeader_.timestamp;
            std::string batchPrevHash = blockHeader_.previousHash;
            std::string batchMerkleRoot = blockHeader_.merkleRoot;
            u32 batchDifficulty = blockHeader_.difficulty;
            
            // 🎯 CRITICAL TEMPLATE VERSION CHECK: Abort if template changed during capture
            u64 finalTemplateVersion = currentBlockTemplateVersion_.load();
            if (finalTemplateVersion != currentTemplateVersion) {
                LOG_DEBUG("🎯 TEMPLATE CHANGED during batch capture - aborting batch for GPU " + std::to_string(gpuIndex) + 
                         " (version changed from " + std::to_string(currentTemplateVersion) + " to " + std::to_string(finalTemplateVersion) + ")", "VELORA");
                continue; // Skip this batch and try again with new template
            }
            
            // 🎯 VALIDATION: Log what timestamp we're actually going to use
            LOG_DEBUG("🎯 GPU " + std::to_string(gpuIndex) + " USING VALIDATED TIMESTAMP: " + std::to_string(batchTimestamp) + 
                     " (template version: " + std::to_string(currentTemplateVersion) + ")", "VELORA");
            
            // 🎯 STORE TEMPLATE VERSION: Store with batch data for result filtering
            u64 batchTemplateVersion = currentTemplateVersion;

            // 🎯 PROPER BATCH TIMING: Start timing actual GPU work
            startBatchTiming(gpuIndex, MAX_PERFORMANCE_BATCH_SIZE);

            // 🚀 GPU-OPTIMIZED: Generate hashes with internal GPU nonce generation
            u64 batchStartNonce = gpuStartNonce + gpuCurrentNonces_[gpuIndex];
            u32 nonceStep = totalGPUs;
            u32 batchCount = MAX_PERFORMANCE_BATCH_SIZE;

            std::vector<GPUBatchResult> batchResults = gpuAlgorithms_[gpuIndex]->generateHashBatchGPUWithAccumulators(
                batchBlockIndex,
                batchStartNonce,  // 🚀 NEW: Starting nonce
                nonceStep,        // 🚀 NEW: Step between nonces
                batchCount,       // 🚀 NEW: Batch count
                batchTimestamp,
                utils::CryptoUtils::hexToHash(batchPrevHash),
                utils::CryptoUtils::hexToHash(batchMerkleRoot),
                batchDifficulty
            );

            // 🎯 PROPER BATCH TIMING: End timing actual GPU work  
            endBatchTiming(gpuIndex);

            // 🚀 GPU-OPTIMIZED: Update hashes processed count
            LOG_DEBUG("GPU mining batch completed - adding " + std::to_string(batchCount) + " hashes to counter", "VELORA");
            hashesProcessed_.fetch_add(batchCount);
            LOG_DEBUG("Total hashes processed now: " + std::to_string(hashesProcessed_.load()), "VELORA");
            
            // Record batch for accurate hashrate calculation
            recordBatchCompletion(batchCount);

            // 🎯 PER-GPU HASH COUNTING: Update hash count for this specific GPU
            if (!perGPUHashesProcessed_.empty()) {
                updateGPUHashCount(gpuIndex, batchCount);
            }

            // 🚀 MAXIMUM PERFORMANCE: Zero delay between batches for 100% GPU utilization
            // Immediately start next batch to maintain continuous 100% GPU usage
            static auto lastBatchTime = std::chrono::high_resolution_clock::now();
            auto now = std::chrono::high_resolution_clock::now();
            auto timeSinceLastBatch = std::chrono::duration_cast<std::chrono::microseconds>(now - lastBatchTime);

            // CRITICAL: Zero delay between batches for maximum performance
            if (timeSinceLastBatch.count() < 100) { // Less than 0.1ms
                // Force immediate batch continuation - MAXIMUM PERFORMANCE
                lastBatchTime = now;
            } else {
                lastBatchTime = now;
            }

            // 🔒 COORDINATED MINING: Double-check block hasn't been found during processing
            if (blockFoundByAnySystem_.load()) {
                LOG_DEBUG("🔒 GPU " + std::to_string(gpuIndex) + " mining stopped - block found during processing", "VELORA");
                break;
            }

            // 🎯 TEMPLATE VERSION CHECK: Filter out results from outdated template versions
            if (currentBlockTemplateVersion_.load() != batchTemplateVersion) {
                LOG_DEBUG("🎯 DISCARDING GPU BATCH: Template version changed from " + std::to_string(batchTemplateVersion) + 
                         " to " + std::to_string(currentBlockTemplateVersion_.load()) + " for GPU " + std::to_string(gpuIndex), "VELORA");
                continue; // Discard entire batch and start fresh
            }

            // Process batch results
            for (size_t i = 0; i < batchResults.size(); i++) {
                if (meetsDifficulty(batchResults[i].hash)) {
                    MiningResult result;
                    result.nonce = static_cast<u32>(batchResults[i].nonce);
                    result.hash.assign(batchResults[i].hash.begin(), batchResults[i].hash.end());
                    result.difficulty = difficulty_;
                    result.isValid = true;
                    // 🎯 CRITICAL FIX: Use the EXACT timestamp used during mining, not current time
                    result.timestamp = batchTimestamp;
                    // 🎯 DEBUGGING: Add accumulator value for comparison with daemon
                    result.accumulator = batchResults[i].accumulator;

                    // 🎯 MINING SYSTEM IDENTIFICATION: Mark as GPU-found block
                    result.miningSystem = "GPU #" + std::to_string(gpuIndex);
                    result.gpuId = static_cast<int>(gpuIndex);

                    // 🚀 CRITICAL FIX: Use GPU-calculated accumulator instead of recalculating
                    // This ensures the miner and daemon use the exact same 96-byte hash input
                    u32 gpuAccumulator = batchResults[i].accumulator;

                    // 🎯 GPU FINAL HASH DEBUG - Match daemon format exactly WITH CORRECT WINNING NONCE
                    LOG_DEBUG("=== 🎯 MINER FINAL HASH - 96-BYTE INPUT DEBUG ===", "BLOCK");

                    // Show detailed buffer debug for the WINNING nonce (not batch template)
                    {
                        std::vector<u8> nonceLE64 = pastella::utils::CryptoUtils::toLittleEndian(result.nonce);
                        std::vector<u8> timestampLE64 = pastella::utils::CryptoUtils::toLittleEndian(batchTimestamp);

                        std::stringstream debug;
                        debug << "=== BUFFER DEBUG ===";
                        LOG_DEBUG(debug.str(), "BLOCK");

                        debug.str("");
                        debug << "Nonce: " << result.nonce << ", Nonce Buffer (hex): ";
                        for (u8 byte : nonceLE64) {
                            debug << std::hex << std::setfill('0') << std::setw(2) << static_cast<u32>(byte);
                        }
                        LOG_DEBUG(debug.str(), "BLOCK");

                        debug.str("");
                        debug << std::dec << "Timestamp: " << batchTimestamp << ", Timestamp Buffer (hex): ";
                        for (u8 byte : timestampLE64) {
                            debug << std::hex << std::setfill('0') << std::setw(2) << static_cast<u32>(byte);
                        }
                        LOG_DEBUG(debug.str(), "BLOCK");

                        debug.str("");
                        debug << "=== END BUFFER DEBUG ===";
                        LOG_DEBUG(debug.str(), "BLOCK");

                        // Show first 10 mix operations with CORRECT nonce values
                        for (int mixIdx = 0; mixIdx < 10; mixIdx++) {
                            u32 nonceWordIndex = mixIdx % 4;
                            u32 timestampWordIndex = mixIdx % 4;

                            // Read 32-bit words from the buffers (with zero-padding if needed)
                            u32 nonceWord = 0;
                            u32 timestampWord = 0;

                            if (nonceWordIndex * 4 < nonceLE64.size()) {
                                for (int byteIdx = 0; byteIdx < 4 && (nonceWordIndex * 4 + byteIdx) < nonceLE64.size(); byteIdx++) {
                                    nonceWord |= static_cast<u32>(nonceLE64[nonceWordIndex * 4 + byteIdx]) << (byteIdx * 8);
                                }
                            }

                            if (timestampWordIndex * 4 < timestampLE64.size()) {
                                for (int byteIdx = 0; byteIdx < 4 && (timestampWordIndex * 4 + byteIdx) < timestampLE64.size(); byteIdx++) {
                                    timestampWord |= static_cast<u32>(timestampLE64[timestampWordIndex * 4 + byteIdx]) << (byteIdx * 8);
                                }
                            }

                            debug.str("");
                            debug << std::dec << "  Mix[" << mixIdx << "]: nonceIndex=" << nonceWordIndex
                                  << ", nonceWord=0x" << std::hex << std::setfill('0') << std::setw(8) << nonceWord
                                  << ", timestampIndex=" << std::dec << timestampWordIndex
                                  << ", timestampWord=0x" << std::hex << std::setfill('0') << std::setw(8) << timestampWord;
                            LOG_DEBUG(debug.str(), "BLOCK");
                        }
                    }

                    {
                        std::stringstream debug;
                        debug << std::dec << "Block number: " << batchBlockIndex;
                        LOG_DEBUG(debug.str(), "BLOCK");
                    }
                    {
                        std::stringstream debug;
                        debug << "Nonce: " << result.nonce;
                        LOG_DEBUG(debug.str(), "BLOCK");
                    }
                    {
                        std::stringstream debug;
                        debug << "Timestamp: " << batchTimestamp;
                        LOG_DEBUG(debug.str(), "BLOCK");
                    }
                    {
                        std::stringstream debug;
                        debug << "Previous hash: " << batchPrevHash;
                        LOG_DEBUG(debug.str(), "BLOCK");
                    }
                    {
                        std::stringstream debug;
                        debug << "Merkle root: " << batchMerkleRoot;
                        LOG_DEBUG(debug.str(), "BLOCK");
                    }
                    {
                        std::stringstream debug;
                        debug << "Difficulty: " << batchDifficulty;
                        LOG_DEBUG(debug.str(), "BLOCK");
                    }
                    {
                        std::stringstream debug;
                        debug << "Accumulator: " << gpuAccumulator;
                        LOG_DEBUG(debug.str(), "BLOCK");
                    }
                    {
                        std::stringstream debug;
                        debug << "Final data length: 96 bytes (should be 96)";
                        LOG_DEBUG(debug.str(), "BLOCK");
                    }
                    {
                        std::stringstream debug;
                        debug << "Miner computed hash: " << pastella::utils::CryptoUtils::hashToHex(result.hash);
                        LOG_DEBUG(debug.str(), "BLOCK");
                    }
                    LOG_DEBUG("=== END MINER FINAL HASH DEBUG ===", "BLOCK");

                    if (hashFoundCallback_) {
                        hashFoundCallback_(result);
                    }

                    // 🔒 COORDINATED MINING: For daemon mining, stop all mining when block found
                    // For pool mining, continue mining to find more shares
                    if (daemonMining_) {
                        blockFoundByAnySystem_.store(true);
                        shouldStop_ = true;
                        LOG_DEBUG("🔒 GPU " + std::to_string(gpuIndex) + " mining stopping due to block found - DAEMON MODE", "VELORA");
                        break;
                    } else {
                        LOG_DEBUG("🔒 GPU " + std::to_string(gpuIndex) + " found share - POOL MODE, continuing", "VELORA");
                        // Continue mining for more shares - don't break
                    }
                }
            }

            // 🚀 GPU-OPTIMIZED: Update GPU nonce counter
            gpuCurrentNonces_[gpuIndex] += batchCount;
        }

        LOG_DEBUG("GPU " + std::to_string(gpuIndex) + " mining worker finished - shouldStop: " +
                 std::to_string(shouldStop_.load()) +
                 ", currentNonce: " + std::to_string(gpuCurrentNonces_[gpuIndex]), "VELORA");

        // Set mining to false when worker finishes
        mining_ = false;
        LOG_DEBUG("GPU " + std::to_string(gpuIndex) + " mining worker set mining_ to false", "VELORA");

    } catch (const std::exception& e) {
        setError(ErrorCode::MINING_FAILED, "GPU " + std::to_string(gpuIndex) + " mining worker failed", e.what());
        mining_ = false; // Also set to false on error
    }
}

// 🎯 BACKWARD COMPATIBILITY: Old method signature for single GPU
void VeloraMiner::gpuMiningWorker() {
    // Delegate to multi-GPU version with GPU index 0
    gpuMiningWorker(0);
}

bool VeloraMiner::processGPUResults() {
    // TODO: Implement GPU result processing
    return false;
}

void VeloraMiner::cpuMiningWorker() {
    // TODO: Implement CPU mining worker
}

bool VeloraMiner::processCPUResults() {
    // TODO: Implement CPU result processing
    return false;
}

void VeloraMiner::setError(ErrorCode code, const std::string& message, const std::string& details) {
    LOG_DEBUG("Setting error - code: " + std::to_string(static_cast<int>(code)) +
             ", message: " + message +
             ", details: " + details, "VELORA");
    lastError_ = ErrorInfo{message, details, "VeloraMiner"};

    if (errorCallback_) {
        errorCallback_(lastError_);
    }
}

void VeloraMiner::updatePerformanceMetrics(u64 hashesProcessed, u64 timeMs) {
    LOG_DEBUG("Updating performance metrics - hashesProcessed: " + std::to_string(hashesProcessed) +
             ", timeMs: " + std::to_string(timeMs), "VELORA");
    metrics_.hashesProcessed = hashesProcessed;
    metrics_.totalTimeMs = timeMs;

    if (timeMs > 0) {
        metrics_.hashesPerSecond = (hashesProcessed * 1000) / timeMs;
        metrics_.averageTimePerHashMs = timeMs / hashesProcessed;
    }
}

bool VeloraMiner::meetsDifficulty(const Hash256& hash) const {
    // Use real Velora difficulty checking as per VELORA_ALGO.md specification
    std::string hashHex = utils::CryptoUtils::hashToHex(hash);

    // Use the proper MiningUtils function for Velora difficulty checking
    bool meetsDifficulty = MiningUtils::veloraHashMeetsDifficulty(hash, difficulty_);

    return meetsDifficulty;
}

// Daemon mining methods
bool VeloraMiner::isDaemonMining() const {
    bool result = daemonMining_;
    LOG_DEBUG("isDaemonMining() called - returning: " + std::to_string(result), "VELORA");
    return result;
}

void VeloraMiner::setDaemonMining(bool enabled) {
    LOG_DEBUG("Setting daemon mining mode: " + std::to_string(enabled), "VELORA");
    daemonMining_ = enabled;
}

u64 VeloraMiner::getCurrentBlockIndex() const {
    u64 result = currentBlockIndex_;
    LOG_DEBUG("getCurrentBlockIndex() called - returning: " + std::to_string(result), "VELORA");
    return result;
}

Hash256 VeloraMiner::getPreviousHash() const {
    LOG_DEBUG("getPreviousHash() called", "VELORA");
    return previousHash_;
}

Hash256 VeloraMiner::getMerkleRoot() const {
    LOG_DEBUG("getMerkleRoot() called", "VELORA");
    return merkleRoot_;
}

BlockHeader VeloraMiner::getCurrentBlockHeader() const {
    LOG_DEBUG("getCurrentBlockHeader() called - index: " + std::to_string(blockHeader_.index) +
             ", difficulty: " + std::to_string(blockHeader_.difficulty), "VELORA");
    return blockHeader_;
}

// 🎯 MULTI-GPU SUPPORT: Set multiple GPU configurations
void VeloraMiner::setMultiGPUConfig(const std::vector<GPUConfig>& configs) {
    LOG_DEBUG("setMultiGPUConfig() called with " + std::to_string(configs.size()) + " GPU(s)", "VELORA");
    multiGPUConfigs_ = configs;

    if (!configs.empty()) {
        // Use the first GPU config as the primary one for backward compatibility
        gpuConfig_ = configs[0];
        useGPU_ = true;

        // 🎯 MULTI-GPU: Create separate algorithm instance for each GPU to avoid context sharing
        gpuAlgorithms_.clear();
        
        // Initialize per-GPU share counters
        perGPUAcceptedShares_.clear();
        perGPURejectedShares_.clear();
        for (size_t i = 0; i < configs.size(); i++) {
            perGPUAcceptedShares_.emplace_back(std::make_unique<std::atomic<u64>>(0));
            perGPURejectedShares_.emplace_back(std::make_unique<std::atomic<u64>>(0));
        }
        
        // 🎯 Initialize accurate batch timing tracking (PRESERVE existing data!)
        if (perGPUTotalBatches_.empty()) {
            // Only initialize if arrays are empty - NEVER reset existing timing data
            perGPUBatchStartTimes_.assign(configs.size(), 0);
            perGPUTotalBatches_.assign(configs.size(), 0);
            perGPUTotalTimeMs_.assign(configs.size(), 0);
            perGPUNoncesPerBatch_.assign(configs.size(), 0);
            LOG_DEBUG("Initialized timing arrays for " + std::to_string(configs.size()) + " GPUs", "VELORA");
        } else {
            // Arrays already exist - NEVER modify them to preserve timing data
            LOG_DEBUG("Preserving existing timing data for " + std::to_string(perGPUTotalBatches_.size()) + " GPUs", "VELORA");
        }
        
        for (size_t i = 0; i < configs.size(); i++) {
            auto gpuAlgo = std::make_unique<VeloraAlgorithm>();
            gpuAlgo->setGPUConfig(configs[i]);
            gpuAlgo->setUseGPU(true);
            gpuAlgorithms_.push_back(std::move(gpuAlgo));
            LOG_DEBUG("Created VeloraAlgorithm instance for GPU " + std::to_string(i) +
                     " (Device " + std::to_string(configs[i].deviceId) + ")", "VELORA");
        }

        // Initialize per-GPU nonce counters
        gpuCurrentNonces_.clear();
        for (size_t i = 0; i < configs.size(); i++) {
            gpuCurrentNonces_.emplace_back(0);
        }

        // 🎯 PER-GPU HASH COUNTING: Initialize per-GPU hash counters
        perGPUHashesProcessed_.clear();
        for (size_t i = 0; i < configs.size(); i++) {
            perGPUHashesProcessed_.emplace_back(std::make_unique<std::atomic<u64>>(0)); // Initialize to 0 hashes
        }

        // 🎯 PER-GPU HASH RATE TRACKING: Initialize per-GPU hashrate tracking
        perGPUHashrateSamples_.clear();
        perGPUHashrateMutexes_.clear();
        for (size_t i = 0; i < configs.size(); i++) {
            perGPUHashrateSamples_.emplace_back(); // Empty vector for this GPU
            perGPUHashrateMutexes_.emplace_back(std::make_unique<std::mutex>()); // Create new mutex for this GPU
        }

        LOG_INFO_CAT("Multi-GPU configuration set - " + std::to_string(configs.size()) + " device(s) enabled", "VELORA");

        // Log each GPU configuration
        for (size_t i = 0; i < configs.size(); i++) {
            const auto& config = configs[i];
            LOG_INFO_CAT("  GPU #" + std::to_string(i) + " (Device " + std::to_string(config.deviceId) +
                    "): " + std::to_string(config.threadsPerBlock) + " threads, " +
                    std::to_string(config.blocksPerGrid) + " blocks, batch_size: " +
                    std::to_string(config.maxNonces), "VELORA");
        }

        // Log total combined batch size
        u64 totalBatch = getTotalBatchSize();
        LOG_INFO_CAT("Total combined batch size: " + std::to_string(totalBatch) + " nonces per round", "VELORA");
    } else {
        useGPU_ = false;
        gpuCurrentNonces_.clear();
        gpuAlgorithms_.clear();
        LOG_DEBUG("Multi-GPU configuration cleared - GPU mining disabled", "VELORA");
    }
}

std::vector<GPUConfig> VeloraMiner::getMultiGPUConfig() const {
    LOG_DEBUG("getMultiGPUConfig() called - returning " + std::to_string(multiGPUConfigs_.size()) + " GPU config(s)", "VELORA");
    return multiGPUConfigs_;
}

int VeloraMiner::getActiveGPUCount() const {
    int count = static_cast<int>(multiGPUConfigs_.size());
    LOG_DEBUG("getActiveGPUCount() called - returning: " + std::to_string(count), "VELORA");
    return count;
}

u64 VeloraMiner::getTotalBatchSize() const {
    u64 total = 0;
    for (const auto& config : multiGPUConfigs_) {
        total += config.maxNonces;
    }
    LOG_DEBUG("getTotalBatchSize() called - returning: " + std::to_string(total), "VELORA");
    return total;
}

// 🎯 PER-GPU HASH RATE TRACKING: Get individual GPU hashrates
std::vector<double> VeloraMiner::getIndividualGPUHashrates() const {
    std::vector<double> gpuHashrates;

    try {
        for (size_t gpuIndex = 0; gpuIndex < perGPUTotalBatches_.size(); gpuIndex++) {
            double gpuHashrate = 0.0;
            
            u64 completedBatches = perGPUTotalBatches_[gpuIndex];
            u64 totalRunningTimeMs = perGPUTotalTimeMs_[gpuIndex];
            u64 noncesPerBatch = perGPUNoncesPerBatch_[gpuIndex];
            
            if (completedBatches > 0 && totalRunningTimeMs > 0 && noncesPerBatch > 0) {
                // Calculate: (batches × nonces per batch) / running time in seconds
                u64 totalNonces = completedBatches * noncesPerBatch;
                gpuHashrate = static_cast<double>(totalNonces) * 1000.0 / static_cast<double>(totalRunningTimeMs);
            }
            
            gpuHashrates.push_back(gpuHashrate);
        }
        
    } catch (...) {
        LOG_WARNING("Exception in getIndividualGPUHashrates(), returning empty", "VELORA");
    }

    LOG_DEBUG("getIndividualGPUHashrates() called - returning " + std::to_string(gpuHashrates.size()) + " GPU hashrates", "VELORA");
    return gpuHashrates;
}

// 🎯 PER-GPU HASH COUNTING: Update hashes processed by specific GPU
void VeloraMiner::updateGPUHashCount(size_t gpuIndex, u64 hashesProcessed) {
    if (gpuIndex >= perGPUHashesProcessed_.size()) {
        LOG_DEBUG("updateGPUHashCount() called with invalid GPU index: " + std::to_string(gpuIndex), "VELORA");
        return;
    }

    // Update the hash counter for this GPU
    perGPUHashesProcessed_[gpuIndex]->fetch_add(hashesProcessed);

    // Update per-GPU hashrate samples
    if (gpuIndex < perGPUHashrateSamples_.size() && gpuIndex < perGPUHashrateMutexes_.size()) {
        std::lock_guard<std::mutex> lock(*perGPUHashrateMutexes_[gpuIndex]);

        auto now = std::chrono::steady_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

        // Add new sample: (timestamp, total hashes for this GPU)
        u64 totalHashesForGPU = perGPUHashesProcessed_[gpuIndex]->load();
        perGPUHashrateSamples_[gpuIndex].emplace_back(timestamp, totalHashesForGPU);

        // Maintain rolling window (keep only MAX_SAMPLES)
        if (perGPUHashrateSamples_[gpuIndex].size() > MAX_SAMPLES) {
            perGPUHashrateSamples_[gpuIndex].erase(perGPUHashrateSamples_[gpuIndex].begin());
        }

        LOG_DEBUG("GPU " + std::to_string(gpuIndex) + " hash count updated: +" + std::to_string(hashesProcessed) +
                 " = " + std::to_string(totalHashesForGPU) + " total, samples: " +
                 std::to_string(perGPUHashrateSamples_[gpuIndex].size()), "VELORA");
    }
}

// Share tracking methods
void VeloraMiner::incrementAcceptedShares(int gpuId) {
    acceptedShares_.fetch_add(1);
    if (gpuId >= 0 && gpuId < static_cast<int>(perGPUAcceptedShares_.size())) {
        perGPUAcceptedShares_[gpuId]->fetch_add(1);
    }
}

void VeloraMiner::incrementRejectedShares(int gpuId) {
    rejectedShares_.fetch_add(1);
    if (gpuId >= 0 && gpuId < static_cast<int>(perGPURejectedShares_.size())) {
        perGPURejectedShares_[gpuId]->fetch_add(1);
    }
}

u64 VeloraMiner::getAcceptedShares() const {
    return acceptedShares_.load();
}

u64 VeloraMiner::getRejectedShares() const {
    return rejectedShares_.load();
}

std::vector<u64> VeloraMiner::getPerGPUAcceptedShares() const {
    std::vector<u64> shares;
    for (const auto& counter : perGPUAcceptedShares_) {
        shares.push_back(counter->load());
    }
    return shares;
}

std::vector<u64> VeloraMiner::getPerGPURejectedShares() const {
    std::vector<u64> shares;
    for (const auto& counter : perGPURejectedShares_) {
        shares.push_back(counter->load());
    }
    return shares;
}

// Hashrate display methods
void VeloraMiner::displayHashrate() const {
    try {
        // Color constants for the hashrate display
        const std::string COLOR_DARK_YELLOW = "\033[33m";
        const std::string COLOR_YELLOW = "\033[93m";
        const std::string COLOR_GREEN = "\033[32m";
        const std::string COLOR_RED = "\033[31m";
        const std::string COLOR_WHITE = "\033[37m";
        const std::string COLOR_RESET = "\033[0m";
        
        // Header
        std::string header = COLOR_DARK_YELLOW + "[VeloraHash]" + COLOR_RESET;
        LOG_INFO_CAT(header, "MINER");
        
        // Safely get total hashrate
        double totalHashrate = 0.0;
        try {
            totalHashrate = getCurrentHashrate();
        } catch (...) {
            LOG_WARNING("Failed to get current hashrate", "MINER");
            totalHashrate = 0.0;
        }
        
        // Safely get share counts
        u64 totalAccepted = 0;
        u64 totalRejected = 0;
        try {
            totalAccepted = getAcceptedShares();
            totalRejected = getRejectedShares();
        } catch (...) {
            LOG_WARNING("Failed to get share counts", "MINER");
        }
        
        // Safely get GPU share counts
        std::vector<u64> gpuAccepted;
        std::vector<u64> gpuRejected;
        try {
            gpuAccepted = getPerGPUAcceptedShares();
            gpuRejected = getPerGPURejectedShares();
        } catch (...) {
            LOG_WARNING("Failed to get per-GPU share counts", "MINER");
        }
        
        // Safely get active GPU count
        int activeGPUCount = 0;
        try {
            activeGPUCount = getActiveGPUCount();
        } catch (...) {
            LOG_WARNING("Failed to get active GPU count", "MINER");
            activeGPUCount = 0;
        }
        
        if (activeGPUCount > 0) {
            // Get individual GPU hashrates instead of dividing total
            std::vector<double> gpuHashrates = getIndividualGPUHashrates();
            
            for (int i = 0; i < activeGPUCount && i < 8; ++i) { // Safety limit to max 8 GPUs
                u64 accepted = (i < gpuAccepted.size()) ? gpuAccepted[i] : 0;
                u64 rejected = (i < gpuRejected.size()) ? gpuRejected[i] : 0;
                
                // Get individual GPU hashrate or fallback to 0
                double perGPUHashrate = (i < static_cast<int>(gpuHashrates.size())) ? gpuHashrates[i] : 0.0;
                
                // Format hashrate with 2 decimal places and safety checks
                std::string hashrateStr;
                if (perGPUHashrate < 0 || perGPUHashrate > 1e12) {
                    hashrateStr = "0.00 H/s"; // Safety fallback for invalid values
                } else if (perGPUHashrate >= 1000000.0) {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << (perGPUHashrate / 1000000.0) << " MH/s";
                    hashrateStr = oss.str();
                } else if (perGPUHashrate >= 1000.0) {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << (perGPUHashrate / 1000.0) << " KH/s";
                    hashrateStr = oss.str();
                } else {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << perGPUHashrate << " H/s";
                    hashrateStr = oss.str();
                }
                
                // Create GPU line with colored numbers
                std::string gpuLine = "GPU" + std::to_string(i) + ": " + hashrateStr;
                std::string plainGpuLine = "GPU" + std::to_string(i) + ": " + hashrateStr; // For length calculation
                
                // Add padding to align share counters (use plain text length)
                while (plainGpuLine.length() < 25) {
                    gpuLine += " ";
                    plainGpuLine += " ";
                }
                
                gpuLine += "[ " + COLOR_GREEN + std::to_string(accepted) + COLOR_RESET + 
                           " | " + COLOR_RED + std::to_string(rejected) + COLOR_RESET + " ]";
                
                LOG_INFO_CAT(gpuLine, "MINER");
            }
        } else {
            // No GPU info available, show generic line
            std::string hashrateStr;
            if (totalHashrate < 0 || totalHashrate > 1e12) {
                hashrateStr = "0 H/s"; // Safety fallback for invalid values
            } else if (totalHashrate >= 1000000.0) {
                hashrateStr = std::to_string(static_cast<int>(totalHashrate / 1000000.0)) + " MH/s";
            } else if (totalHashrate >= 1000.0) {
                hashrateStr = std::to_string(static_cast<int>(totalHashrate / 1000.0)) + " KH/s";
            } else {
                hashrateStr = std::to_string(static_cast<int>(totalHashrate)) + " H/s";
            }
            
            std::string deviceLine = "Device: " + hashrateStr;
            std::string plainDeviceLine = "Device: " + hashrateStr;
            
            while (plainDeviceLine.length() < 25) {
                deviceLine += " ";
                plainDeviceLine += " ";
            }
            
            deviceLine += "[ " + COLOR_GREEN + std::to_string(totalAccepted) + COLOR_RESET + 
                          " | " + COLOR_RED + std::to_string(totalRejected) + COLOR_RESET + " ]";
            
            LOG_INFO_CAT(deviceLine, "MINER");
        }
        
        // Total line with 2 decimal places
        std::string totalHashrateStr;
        if (totalHashrate < 0 || totalHashrate > 1e12) {
            totalHashrateStr = "0.00 H/s"; // Safety fallback for invalid values
        } else if (totalHashrate >= 1000000.0) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << (totalHashrate / 1000000.0) << " MH/s";
            totalHashrateStr = oss.str();
        } else if (totalHashrate >= 1000.0) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << (totalHashrate / 1000.0) << " KH/s";
            totalHashrateStr = oss.str();
        } else {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << totalHashrate << " H/s";
            totalHashrateStr = oss.str();
        }
        
        std::string totalLine = COLOR_YELLOW + "Total: " + totalHashrateStr + COLOR_RESET;
        std::string plainTotal = "Total: " + totalHashrateStr;
        
        // Add padding using plain text length
        while (plainTotal.length() < 25) {
            totalLine += " ";
            plainTotal += " ";
        }
        
        totalLine += "[ " + COLOR_GREEN + std::to_string(totalAccepted) + COLOR_RESET + 
                     " | " + COLOR_RED + std::to_string(totalRejected) + COLOR_RESET + " ]";
        
        LOG_INFO_CAT(totalLine, "MINER");
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception in displayHashrate(): " + std::string(e.what()), "MINER");
    } catch (...) {
        LOG_ERROR("Unknown exception in displayHashrate()", "MINER");
    }
}

void VeloraMiner::startPeriodicHashrateDisplay() {
    if (displayHashrateActive_.load()) {
        return; // Already running
    }
    
    displayHashrateActive_.store(true);
    LOG_DEBUG("Starting periodic hashrate display thread", "VELORA");
    
    hashrateDisplayThread_ = std::thread([this]() {
        LOG_DEBUG("Periodic hashrate display thread started", "VELORA");
        
        // Initial delay before first display
        std::this_thread::sleep_for(std::chrono::seconds(2));
        
        while (displayHashrateActive_.load()) {
            if (mining_.load()) {
                LOG_DEBUG("Displaying periodic hashrate", "VELORA");
                displayHashrate();
            }
            
            // Sleep for 60 seconds, but check every second if we should stop
            for (int i = 0; i < 60 && displayHashrateActive_.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        
        LOG_DEBUG("Periodic hashrate display thread ended", "VELORA");
    });
}

void VeloraMiner::stopPeriodicHashrateDisplay() {
    displayHashrateActive_.store(false);
    if (hashrateDisplayThread_.joinable()) {
        hashrateDisplayThread_.join();
    }
}

} // namespace velora
} // namespace pastella
