#pragma once

#include "../types.h"
#include "../mining_types.h"
#include <memory>
#include <vector>
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

namespace pastella {
namespace velora {

/**
 * GPU Batch Result - contains hash and accumulator for each nonce
 */
struct GPUBatchResult {
    Hash256 hash;
    u32 accumulator;
    u64 nonce;
};

/**
 * CPU Hash Result - contains hash and accumulator for debugging
 */
struct CPUHashResult {
    Hash256 hash;
    u32 accumulator;
};

/**
 * Velora Algorithm Implementation
 * ASIC-resistant, GPU-friendly proof-of-work algorithm
 */
class VeloraAlgorithm {
public:
    VeloraAlgorithm();
    ~VeloraAlgorithm();

    // Disable copy constructor and assignment
    VeloraAlgorithm(const VeloraAlgorithm&) = delete;
    VeloraAlgorithm& operator=(const VeloraAlgorithm&) = delete;

    // Core algorithm methods (Enhanced v1.1)
    Hash256 generateHash(u64 blockNumber, u64 nonce, u64 timestamp, const Hash256& previousHash, const Hash256& merkleRoot, u32 difficulty);
    // Hash256 generateHashGPU(...) removed from per-nonce usage; GPU path is batched only
    // 🚀 GPU-OPTIMIZED: Batch GPU processing with internal nonce generation
    std::vector<Hash256> generateHashBatchGPU(u64 blockNumber, u64 start_nonce, u32 nonce_step, u32 batch_count, u64 timestamp, const Hash256& previousHash, const Hash256& merkleRoot, u32 difficulty);

    // 🚀 NEW: GPU batch processing with accumulator values for accurate hash reconstruction
    std::vector<GPUBatchResult> generateHashBatchGPUWithAccumulators(u64 blockNumber, u64 start_nonce, u32 nonce_step, u32 batch_count, u64 timestamp, const Hash256& previousHash, const Hash256& merkleRoot, u32 difficulty);
    bool verifyHash(u64 blockNumber, u64 nonce, u64 timestamp, const Hash256& previousHash, const Hash256& merkleRoot, u32 difficulty, const Hash256& targetHash);

    // Epoch management
    u64 getEpoch(u64 blockNumber) const;
    Hash256 generateEpochSeed(u64 blockNumber);

    // Scratchpad management
    void generateScratchpad(const Hash256& epochSeed);
    const std::vector<u32>& getScratchpad() const;

    // Memory pattern generation (Enhanced)
    std::vector<u32> generateMemoryPattern(u64 blockNumber, u64 nonce, u64 timestamp, const Hash256& previousHash, const Hash256& merkleRoot, u32 difficulty);

    // Memory walk execution (Enhanced)
    u32 executeMemoryWalk(const std::vector<u32>& pattern, u64 nonce, u64 timestamp);

    // Performance optimization
    void setUseGPU(bool useGPU);
    void setGPUConfig(const GPUConfig& config);

    // GPU management
    bool initializeGPU();
    void cleanupGPU();

    // Error handling
    ErrorInfo getLastError() const;
    void clearError();

    // Statistics
    PerformanceMetrics getPerformanceMetrics() const;
    void resetPerformanceMetrics();

private:
    // Internal state
    std::vector<u32> scratchpad_;
    Hash256 currentEpochSeed_;
    u64 currentEpoch_;
    bool useGPU_;
    GPUConfig gpuConfig_;

    // Performance tracking
    mutable PerformanceMetrics metrics_;
    mutable u64 startTime_;

    // Error tracking
    mutable ErrorInfo lastError_;

    // Internal methods
    void updatePerformanceMetrics(u64 hashesProcessed, u64 timeMs);
    void setError(ErrorCode code, const std::string& message, const std::string& details = "");

#ifdef HAVE_CUDA
        // 🚀 TIER 2: DOUBLE BUFFERING + MULTI-STREAM PIPELINE
    cudaStream_t cudaStream_ = 0;
    cudaStream_t cudaStreams_[2] = {0, 0}; // Dual streams for overlapping operations

    // 🚀 DOUBLE BUFFERING: Dual GPU buffers for zero-downtime processing
    u32* d_scratchpad_ = nullptr;
    u32* d_pattern_[2] = {nullptr, nullptr}; // Dual pattern buffers
    u64* d_nonces_[2] = {nullptr, nullptr};  // Dual nonce buffers
    u32* d_acc_out_[2] = {nullptr, nullptr}; // Dual accumulator buffers
    u8* d_hash_out_[2] = {nullptr, nullptr}; // Dual hash buffers
    u8* d_prev_ = nullptr;
    u8* d_merkle_ = nullptr;

    // 🚀 DOUBLE BUFFERING: Dual pinned host memory buffers for zero-copy transfers
    u32* h_pattern_pinned_[2] = {nullptr, nullptr};
    u64* h_nonces_pinned_[2] = {nullptr, nullptr};
    u32* h_acc_pinned_[2] = {nullptr, nullptr};
    u8* h_hash_pinned_[2] = {nullptr, nullptr}; // GPU-computed data for validation

    size_t cudaScratchpadSize_ = 0;
    size_t cudaPatternSize_ = 0;
    size_t cudaNoncesSize_ = 0;
    size_t cudaAccSize_ = 0;
    size_t cudaHashSize_ = 0;
    bool cudaInitialized_ = false;
    int currentStream_ = 0;
#endif

    // CPU fallback (Enhanced)
    Hash256 generateHashCPU(u64 blockNumber, u64 nonce, u64 timestamp, const Hash256& previousHash, const Hash256& merkleRoot, u32 difficulty);

    // Final hash generation using all 6 parameters
    Hash256 generateFinalHash(u64 blockNumber, u64 nonce, u64 timestamp, const Hash256& previousHash, const Hash256& merkleRoot, u32 difficulty, u32 accumulator);

    // Utility methods
    u32 xorshift32(u32 state);
    u32 seedFromHash(const Hash256& hash);
    void mixScratchpad(const Hash256& epochSeed);

    // Helper function to ensure 32-bit unsigned arithmetic like JavaScript's >>> 0
    u32 ensure32BitUnsigned(u64 value);

    // Constants
    static constexpr u32 MIXING_ROUNDS = 2;
    static constexpr u32 MIXING_CONSTANT = 0x5bd1e995;
};

} // namespace velora
} // namespace pastella
