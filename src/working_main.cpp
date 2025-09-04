#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <signal.h>
#include <atomic>
#include "../include/mining_types.h"
#include "../include/utils/logger.h"
#include "../include/utils/crypto_utils.h"
#include "../include/config_manager.h"
#include "../include/velora/velora_miner.h"
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif
#include "../include/transactions/transaction_manager.h"
#include "../include/daemon/daemon_client.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

// Undefine Windows macros that conflict with std::min/max and RapidJSON
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

using namespace pastella;
using namespace pastella::utils;
using namespace pastella::velora;

// Global variables for signal handling
std::atomic<bool> g_running(true);
VeloraMiner* g_miner = nullptr;
MinerConfig* g_config = nullptr;

// Global instances for modular components
transactions::TransactionManager* g_transactionManager = nullptr;
daemon::DaemonClient* g_daemonClient = nullptr;

// Global variables for storing final transaction data used during mining
rapidjson::Value g_finalCoinbaseTransaction;
u64 g_finalMiningTimestamp = 0;
std::string g_finalMiningMerkleRoot;

// Flag to track when block submission is complete and ready for next block
std::atomic<bool> g_blockSubmissionComplete(false);

// Helper functions using modular components
bool checkDaemonConnectivity(const std::string& daemonUrl) {
    if (!g_daemonClient) {
        LOG_ERROR_CAT("Daemon client not initialized", "DAEMON");
        return false;
    }
    return g_daemonClient->checkConnectivity(daemonUrl);
}

DaemonBlock getLatestBlockFromDaemon(const std::string& daemonUrl, const std::string& apiKey) {
    if (!g_daemonClient) {
        LOG_ERROR_CAT("Daemon client not initialized", "DAEMON");
        return DaemonBlock{};
    }
    return g_daemonClient->getLatestBlock(daemonUrl, apiKey);
}

bool submitBlockToDaemon(const std::string& jsonPayload, const std::string& daemonUrl, const std::string& apiKey) {
    if (!g_daemonClient) {
        LOG_ERROR_CAT("Daemon client not initialized", "DAEMON");
        return false;
    }
    return g_daemonClient->submitBlock(jsonPayload, daemonUrl, apiKey);
}

// Signal handler
void signalHandler(int signal) {
    LOG_INFO_CAT("Received signal " + std::to_string(signal) + ", shutting down...", "CTRL");
    g_running = false;

    if (g_miner) {
        g_miner->stopMining();
    }
}


// Print hardware status
void printHeaderText(const MinerConfig& config) {
    LOG_STATUS(" * ABOUT:       PASTELLA-MINER/1.0.0 MSVC/2019", "STATUS");

    if (config.gpu_enabled) {
        LOG_STATUS(" * GPU:         CUDA enabled", "STATUS");
        if (!config.cuda_devices.empty()) {
            const auto& gpu = config.cuda_devices[0];
            const bool overrideLaunch = gpu.override_launch;
            int threadsOut = gpu.threads;
            int blocksOut = gpu.blocks;
            if (!overrideLaunch) {
                // Resolve auto values from device properties
                int deviceId = gpu.id;
                cudaDeviceProp prop;
                if (cudaGetDeviceProperties(&prop, deviceId) == cudaSuccess) {
                    threadsOut = 512;
                    blocksOut = prop.multiProcessorCount * 12;
                } else {
                    threadsOut = 512;
                    blocksOut = 0;
                }
            }
            LOG_STATUS(" * CUDA:          Device 0 enabled (threads: " + std::to_string(threadsOut) +
                     ", blocks: " + std::to_string(blocksOut) + ")", "STATUS");
        }
    } else {
        LOG_STATUS(" * GPU:         Disabled", "STATUS");
    }

    if (config.cpu_enabled) {
        LOG_STATUS(" * CPU:         " + std::to_string(config.cpu_threads) + " threads enabled", "STATUS");
    } else {
        LOG_STATUS(" * CPU:         Disabled", "STATUS");
    }

    LOG_STATUS(" * ALGO:        velora, donate=0%", "STATUS");
    if (config.pool.daemon) {
        LOG_STATUS(" * MODE:        Daemon mining", "STATUS");
        LOG_STATUS(" * DAEMON:      " + config.pool.daemon_url, "STATUS");
    } else {
        LOG_STATUS(" * POOL:        " + config.pool.url + ":" + std::to_string(config.pool.port), "STATUS");
        LOG_STATUS(" * WALLET:      " + config.pool.wallet, "STATUS");
        LOG_STATUS(" * WORKER:      " + config.pool.worker_name, "STATUS");
    }
    LOG_STATUS("", "STATUS");
}

// Print usage
void printUsage(const char* programName) {
    LOG_STATUS("Usage: " + std::string(programName) + " [OPTIONS]", "HELP");
    LOG_STATUS("", "HELP");
    LOG_STATUS("Options:", "HELP");
    LOG_STATUS("  -c, --config FILE       Configuration file (default: config.json)", "HELP");
    LOG_STATUS("  -o, --url URL           Pool URL", "HELP");
    LOG_STATUS("  -u, --user USER         Wallet address", "HELP");
    LOG_STATUS("  -w, --worker NAME       Worker name", "HELP");
    LOG_STATUS("  --cuda                  Enable CUDA mining", "HELP");
    LOG_STATUS("  --opencl                Enable OpenCL mining (future)", "HELP");
    LOG_STATUS("  --cpu                   Enable CPU mining", "HELP");
    LOG_STATUS("  --daemon                Enable daemon mining (instead of pool)", "HELP");
    LOG_STATUS("  --daemon-url URL        Daemon URL (default: http://localhost:3002)", "HELP");
    LOG_STATUS("  --daemon-api-key KEY    Daemon API key for authentication", "HELP");
    LOG_STATUS("  -t, --threads N         CPU threads", "HELP");
    LOG_STATUS("  --cuda-devices LIST     CUDA device list (e.g., 0,1,2)", "HELP");
    LOG_STATUS("  -h, --help              Show this help", "HELP");
    LOG_STATUS("", "HELP");
}



// Hash found callback
void onHashFound(const MiningResult& result) {
    LOG_CLEAR_PROGRESS();
    LOG_INFO_CAT("Block " + utils::CryptoUtils::hashToHex(result.hash) + " found! (Nonce: " + std::to_string(result.nonce) + ")", "RESULT");

    // If daemon mining is enabled, submit the block to the daemon
    if (g_miner && g_miner->isDaemonMining()) {
        LOG_INFO_CAT("Submitting block to daemon...", "DAEMON");

        // CRITICAL FIX: Use the EXACT same transaction data that was used for mining
        auto currentHeader = g_miner->getCurrentBlockHeader();

        // Use the stored final values that were used during mining
        u64 miningTimestamp = g_finalMiningTimestamp;
        std::string miningMerkleRoot = g_finalMiningMerkleRoot;

        // Use the EXACT same coinbase transaction that was used for mining
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();

        // Copy the final coinbase transaction to avoid any modifications
        rapidjson::Value coinbaseTx(rapidjson::kObjectType);
        coinbaseTx.CopyFrom(g_finalCoinbaseTransaction, allocator);

        // Create transactions array with the exact same coinbase transaction
        rapidjson::Value transactionsArray(rapidjson::kArrayType);
        transactionsArray.PushBack(coinbaseTx, allocator);

        // Create the block object using the EXACT same values used for mining
        rapidjson::Value blockObj(rapidjson::kObjectType);
        blockObj.AddMember("index", currentHeader.index, allocator);
        blockObj.AddMember("timestamp", miningTimestamp, allocator); // Use mining timestamp
        blockObj.AddMember("transactions", transactionsArray, allocator);
        blockObj.AddMember("hash", rapidjson::Value(utils::CryptoUtils::hashToHex(result.hash).c_str(), allocator), allocator);
        blockObj.AddMember("previousHash", rapidjson::Value(currentHeader.previousHash.c_str(), allocator), allocator);
        blockObj.AddMember("merkleRoot", rapidjson::Value(miningMerkleRoot.c_str(), allocator), allocator); // Use mining merkle root
        blockObj.AddMember("nonce", result.nonce, allocator);
        blockObj.AddMember("difficulty", currentHeader.difficulty, allocator);
        blockObj.AddMember("algorithm", "velora", allocator);

        doc.AddMember("block", blockObj, allocator);

        // Convert to JSON string
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        std::string jsonPayload = buffer.GetString();

        // Submit to daemon using the global config
        if (g_config && g_config->pool.daemon) {
            bool submitted = submitBlockToDaemon(jsonPayload, g_config->pool.daemon_url, g_config->pool.daemon_api_key);
                        if (submitted) {
                LOG_INFO_CAT("Block successfully submitted to daemon!", "DAEMON");
                std::this_thread::sleep_for(std::chrono::seconds(2));

                // CRITICAL: Signal that block submission is complete and ready for next block
                g_blockSubmissionComplete = true;

                LOG_INFO_CAT("Block submission complete, ready for next block", "DAEMON");
            } else {
                LOG_ERROR_CAT("Failed to submit block to daemon", "DAEMON");
            }
        }
    }
}

// Error callback
void onError(const ErrorInfo& error) {
    LOG_CLEAR_PROGRESS();
    LOG_ERROR_CAT(error.message, "ERROR");
    if (!error.details.empty()) {
        LOG_ERROR_CAT("Details: " + error.details, "ERROR");
    }
}



// Modern mining functions
bool startDaemonMining(const MinerConfig& config) {
    LOG_INFO_CAT("Starting daemon mining mode", "DAEMON");

    // Check daemon connectivity
    if (!checkDaemonConnectivity(config.pool.daemon_url)) {
        LOG_ERROR_CAT("Daemon is not reachable. Stopping mining.", "DAEMON");
        LOG_ERROR_CAT("Please check if the daemon is running at: " + config.pool.daemon_url, "DAEMON");
        return false;
    }

    // Create VeloraMiner instance
    velora::VeloraMiner veloraMiner;
    g_miner = &veloraMiner;

    // Set callbacks
    veloraMiner.setProgressCallback(nullptr); // No longer needed
    veloraMiner.setErrorCallback(onError);
    veloraMiner.setHashFoundCallback(onHashFound);

    // Configure miner
    veloraMiner.setMaxNonces(config.max_nonces);
    veloraMiner.setNumThreads(config.cpu_threads);
    veloraMiner.setDaemonMining(true);

    // GPU configuration
    if (config.gpu_enabled && !config.cuda_devices.empty()) {
        const auto& gpuDevice = config.cuda_devices[0];
        GPUConfig gpuConfig;
        gpuConfig.deviceId = gpuDevice.device_id;
        gpuConfig.threadsPerBlock = gpuDevice.threads;
        gpuConfig.blocksPerGrid = gpuDevice.blocks;
        gpuConfig.maxNonces = static_cast<u32>(std::min(static_cast<u64>(gpuDevice.batch_size), config.max_nonces));
        gpuConfig.useDoublePrecision = false;
        veloraMiner.setGPUConfig(gpuConfig);
        veloraMiner.setUseGPU(true);
    } else {
        veloraMiner.setUseGPU(false);
    }

    // Continuous mining loop - mine blocks one after another
    LOG_INFO_CAT("Daemon mining mode - continuous mining enabled", "DAEMON");
        while (g_running) {
        // Reset block submission flag for new cycle
        g_blockSubmissionComplete = false;

        // Get latest blockchain data
        DaemonBlock latestBlock = getLatestBlockFromDaemon(config.pool.daemon_url, config.pool.daemon_api_key);
        if (latestBlock.index == 0 && latestBlock.hash.empty()) {
            LOG_ERROR_CAT("Failed to get latest block from daemon", "ERROR");
            break;
        }

        // Prepare mining data
        u64 miningTimestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        // Create coinbase transaction
        rapidjson::Document tempDoc;
        tempDoc.SetObject();
        rapidjson::Document::AllocatorType& tempAllocator = tempDoc.GetAllocator();
        rapidjson::Value coinbaseTx = g_transactionManager->createCoinbaseTransaction(config.pool.wallet, miningTimestamp, tempAllocator);

        // Calculate transaction ID and merkle root
        std::string transactionId = g_transactionManager->calculateTransactionId(coinbaseTx);
        coinbaseTx["id"] = rapidjson::Value(transactionId.c_str(), tempAllocator);
        std::string merkleRoot = g_transactionManager->calculateMerkleRoot(transactionId);

        // Store final data for block submission
        g_finalCoinbaseTransaction = coinbaseTx;
        g_finalMiningTimestamp = miningTimestamp;
        g_finalMiningMerkleRoot = merkleRoot;

        // Create block header
        BlockHeader header;
        header.index = latestBlock.index + 1;
        header.timestamp = miningTimestamp;
        header.difficulty = latestBlock.difficulty;
        header.nonce = 0;
        header.previousHash = latestBlock.hash;
        header.merkleRoot = merkleRoot;
        header.algorithm = "velora";

        // Ensure miner is stopped before setting new block header
        veloraMiner.stopMining();

        // Set block header and start mining
        veloraMiner.setBlockHeader(header);
        veloraMiner.setDifficulty(latestBlock.difficulty);

        LOG_INFO_CAT("Mining block " + std::to_string(header.index) +
                " (Difficulty: " + std::to_string(header.difficulty) + ")", "DAEMON");

        LOG_DEBUG("Starting mining for block " + std::to_string(header.index), "DAEMON");
        if (!veloraMiner.startMining()) {
            LOG_ERROR_CAT("Failed to start mining for block " + std::to_string(header.index), "ERROR");
            break;
        }
        LOG_DEBUG("Mining started successfully for block " + std::to_string(header.index), "DAEMON");

        // Wait for block to be found and submitted
        while (g_running && veloraMiner.isMining()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Wait for block submission to complete
        if (g_running) {
            while (g_running && !g_blockSubmissionComplete) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            g_blockSubmissionComplete = false;
        }

        // Stop mining before preparing next block
        veloraMiner.stopMining();

        if (!g_running) {
            LOG_INFO_CAT("Mining stopped by user request", "DAEMON");
            break;
        }

        LOG_INFO_CAT("Block completed, preparing next block...", "DAEMON");

        // Small delay to ensure clean state transition
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    return true;
}

bool startPoolMining(const MinerConfig& config) {
    LOG_INFO_CAT("Starting pool mining mode", "POOL");

    // TODO: Implement pool mining using Stratum protocol
    // This will include:
    // - Connect to pool server
    // - Handle Stratum protocol messages
    // - Receive work from pool
    // - Submit shares to pool
    // - Handle difficulty adjustments

    LOG_ERROR_CAT("Pool mining not yet implemented. Please use daemon mining.", "POOL");
    return false;
}



int main(int argc, char* argv[]) {
    // Initialize modular components
    transactions::TransactionManager transactionManager;
    daemon::DaemonClient daemonClient;
    g_transactionManager = &transactionManager;
    g_daemonClient = &daemonClient;

    // Set up signal handling
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    try {
        // Use simple config path - will create in current directory if not exists
        std::string configPath = "config.json";
        ConfigManager configManager;
        MinerConfig config;

        // Parse command line arguments
        if (configManager.parseCommandLine(argc, argv, config, configPath)) {
            printUsage(argv[0]);
            return 0;
        }

        // Load or create configuration
        if (!configManager.loadOrCreateConfig(configPath, config)) {
            return 1;
        }

        // Validate configuration
        if (!configManager.validateConfig(config)) {
            return 0; // Exit gracefully for validation issues
        }

        // Set global config for use in callbacks
        g_config = &config;

        // Ensure at least one GPU device is configured if GPU is enabled
        if (config.gpu_enabled && config.cuda_devices.empty()) {
            GPUDeviceConfig defaultGPU;
            config.cuda_devices.push_back(defaultGPU);
        }

        // Print hardware status after config is loaded
        printHeaderText(config);

        // Start mining based on configuration
        if (config.pool.daemon) {
            LOG_INFO_CAT("Daemon mining enabled - will mine to: " + config.pool.daemon_url, "CONFIG");
            return startDaemonMining(config) ? 0 : 1;
        } else {
            LOG_INFO_CAT("Pool mining mode selected", "CONFIG");
            return startPoolMining(config) ? 0 : 1;
        }

    } catch (const std::exception& e) {
        LOG_FATAL(e.what(), "FATAL");

        // Cleanup CURL even on error
        curl_global_cleanup();

        return 1;
    }
}
