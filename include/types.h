#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <memory>

namespace pastella {

// Basic types
using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i32 = int32_t;
using i64 = int64_t;

// Velora algorithm constants
namespace velora {
    constexpr u64 SCRATCHPAD_SIZE = 64 * 1024 * 1024;  // 64MB (as per specification)
    constexpr u32 MEMORY_READS = 65536;  // 262,144 reads per hash (256K reads as requested)
    constexpr u64 EPOCH_LENGTH = 2016;  // 2,016 blocks (Bitcoin-like difficulty adjustment period as per v1.0 specification)
    constexpr u32 SCRATCHPAD_WORDS = SCRATCHPAD_SIZE / 4;  // 16,777,216 words
}

// Hash types
using Hash256 = std::array<u8, 32>;
using Hash512 = std::array<u8, 64>;

// Note: BlockHeader is defined in mining_types.h

// Note: MiningResult is defined in mining_types.h

// GPU configuration
struct GPUConfig {
    u32 deviceId;
    u32 threadsPerBlock;
    u32 blocksPerGrid;
    u32 maxNonces;
    bool useDoublePrecision;

    GPUConfig() : deviceId(0), threadsPerBlock(256), blocksPerGrid(1024),
                  maxNonces(1000000), useDoublePrecision(false) {}
};

// Performance metrics
struct PerformanceMetrics {
    u64 hashesProcessed;
    u64 hashesPerSecond;
    u64 totalTimeMs;
    u64 averageTimePerHashMs;
    u32 gpuUtilization;

    PerformanceMetrics() : hashesProcessed(0), hashesPerSecond(0),
                          totalTimeMs(0), averageTimePerHashMs(0), gpuUtilization(0) {}
};

// Error codes
enum class ErrorCode {
    SUCCESS = 0,
    INVALID_PARAMETER,
    GPU_INIT_FAILED,
    GPU_MEMORY_ERROR,
    GPU_KERNEL_FAILED,
    MEMORY_ALLOCATION_FAILED,
    CRYPTO_ERROR,
    SCRATCHPAD_GENERATION_FAILED,
    PATTERN_GENERATION_FAILED,
    MINING_FAILED
};

// Note: ErrorInfo is defined in mining_types.h

// Utility functions
namespace utils {
    // Convert hash to hex string
    std::string hashToHex(const Hash256& hash);

    // Convert hex string to hash
    Hash256 hexToHash(const std::string& hex);

    // Convert to little-endian bytes
    std::vector<u8> toLittleEndian(u64 value);
    std::vector<u8> toLittleEndian(u32 value);

    // Convert from little-endian bytes
    u64 fromLittleEndian64(const std::vector<u8>& data, size_t offset = 0);
    u32 fromLittleEndian32(const std::vector<u8>& data, size_t offset = 0);

    // Time utilities
    u64 getCurrentTimestamp();
    u64 getCurrentTimestampMs();
}

} // namespace pastella




