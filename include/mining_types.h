#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace pastella {

// Type aliases for better readability
using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;

/**
 * Pool Configuration
 */
struct PoolConfig {
    std::string url = "stratum+tcp://pool.example.com";
    int port = 3333;
    std::string wallet = "wallet_address";
    std::string worker_name = "pastella-miner";
    bool daemon = false;
    std::string daemon_url = "http://localhost:22000";
    std::string daemon_api_key = "";
};

/**
 * GPU Device Configuration
 */
struct GPUDeviceConfig {
    int device_id = 0;
    int threads = 256;
    int blocks = 1024;
    int batch_size = 1000000;
    bool enabled = true;
    bool override_launch = false; // if true, use threads/blocks from config; otherwise auto-tune
};

/**
 * Main Miner Configuration
 */
struct MinerConfig {
    PoolConfig pool;
    std::vector<GPUDeviceConfig> cuda_devices;
    std::vector<GPUDeviceConfig> opencl_devices;
    int cpu_threads = 4;
    bool cpu_enabled = false;
    bool gpu_enabled = true;
    u64 max_nonces = 10000000000;
    bool verbose = false;
};

/**
 * Daemon Block Structure
 */
struct DaemonBlock {
    int index;
    u64 timestamp;
    std::string previousHash;
    std::string merkleRoot;
    u32 nonce;
    u32 difficulty;
    std::string hash;
    std::string algorithm;
    std::vector<std::string> transactions;
};

/**
 * Daemon Response Structure
 */
struct DaemonResponse {
    bool success;
    std::string message;
    std::string error;
};

/**
 * Mining Result Structure
 */
struct MiningResult {
    u32 nonce;
    std::vector<u8> hash;
    u32 difficulty;
    bool isValid;
    u64 timestamp;
    u32 accumulator; // For debugging hash calculation

    // 🎯 MINING SYSTEM IDENTIFICATION: Track which system found the block
    std::string miningSystem; // "GPU" or "CPU"
    int gpuId; // GPU ID for multi-GPU setups (-1 for CPU)
};

/**
 * Error Information Structure
 */
struct ErrorInfo {
    std::string message;
    std::string details;
    std::string location;
};

/**
 * Block Header Structure for Mining
 */
struct BlockHeader {
    int index;
    u64 timestamp;
    std::string previousHash;
    std::string merkleRoot;
    u32 nonce;
    u32 difficulty;
    std::string algorithm;
};

} // namespace pastella
