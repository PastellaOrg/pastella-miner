#include <iostream>
#include <memory>
#include <thread>
#include <atomic>
#include <chrono>
#include <signal.h>
#include <optional>

#ifdef _WIN32
#include <conio.h>
#else
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#endif

#include "../include/mining_types.h"
#include "../include/config_manager.h"
#include "../include/transactions/transaction_manager.h"
#include "../include/daemon/daemon_client.h"
#include "../include/velora/velora_miner.h"
#include "../include/pool_miner.h"
#include "../include/utils/logger.h"
#include "../include/utils/crypto_utils.h"
#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/istreamwrapper.h"

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

// Signal handling
std::atomic<bool> g_running(true);

// Global instances for modular components
transactions::TransactionManager* g_transactionManager = nullptr;
daemon::DaemonClient* g_daemonClient = nullptr;
velora::VeloraMiner* g_miner = nullptr;
MinerConfig* g_config = nullptr;

// Global variables for storing final transaction data used during mining
// Store coinbase as JSON string to avoid RapidJSON allocator lifetime issues
std::string g_finalCoinbaseJson;
u64 g_finalMiningTimestamp = 0;
std::string g_finalMiningMerkleRoot;

// Flag to track when block submission is complete and ready for next block
std::atomic<bool> g_blockSubmissionComplete(false);

// 🚀 MULTI-GPU COORDINATION: Prevent duplicate block submissions
std::atomic<bool> g_blockFound(false);

// Watchdog/phase tracking
enum class MiningPhase { Idle, FetchLatest, PrepareTx, PrepareHeader, StartMining, WaitMining, WaitSubmit };
std::atomic<int> g_phase(static_cast<int>(MiningPhase::Idle));
std::atomic<u64> g_phaseTsMs(0);

static inline u64 nowMs() {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

void setPhase(MiningPhase p, const char* note) {
    g_phase.store(static_cast<int>(p));
    g_phaseTsMs.store(nowMs());
    LOG_DEBUG(std::string("PHASE → ") + note, "WATCHDOG");
}

void signalHandler(int signal) {
    LOG_INFO_CAT("Received signal " + std::to_string(signal) + ", shutting down...", "CTRL");
    g_running = false;

    if (g_miner) {
        g_miner->stopMining();
    }
}

// Cross-platform keyboard input handling
#ifdef _WIN32
char getKeyPress() {
    if (_kbhit()) {
        return _getch();
    }
    return 0;
}
#else
char getKeyPress() {
    struct termios oldt, newt;
    char ch = 0;
    
    // Get current terminal settings
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    
    // Set non-blocking, non-canonical mode
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    
    // Set non-blocking mode
    int oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);
    
    // Try to read a character
    if (read(STDIN_FILENO, &ch, 1) != 1) {
        ch = 0; // No character available
    }
    
    // Restore terminal settings
    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);
    
    return ch;
}
#endif

// Keyboard monitoring function
void keyboardMonitor() {
    LOG_DEBUG("Keyboard monitor started - press 'h' to display hashrate", "KEYBOARD");
    
    static bool keyProcessed = false; // Static flag to prevent rapid triggering
    
    while (g_running) {
        char key = getKeyPress();
        
        if ((key == 'h' || key == 'H') && !keyProcessed) {
            keyProcessed = true; // Set flag immediately
            LOG_DEBUG("'h' key pressed - displaying hashrate", "KEYBOARD");
            if (g_miner) {
                g_miner->displayHashrate();
            } else {
                LOG_WARNING("No miner instance available", "KEYBOARD");
            }
        } else if (key == 0) {
            // No key pressed - reset the processed flag
            keyProcessed = false;
        }
        
        // Small sleep to prevent high CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    LOG_DEBUG("Keyboard monitor stopped", "KEYBOARD");
}

// Hash found callback
void onHashFound(const MiningResult& result) {
    // 🚀 MULTI-GPU COORDINATION: Use atomic compare-and-swap to prevent race conditions
    bool expected = false;
    if (!g_blockFound.compare_exchange_strong(expected, true)) {
        // Another GPU already found a block for this round - ignore this result
        LOG_INFO_CAT("MULTI-GPU: Block already found by another GPU (current: " + result.miningSystem + ", nonce: " + std::to_string(result.nonce) + "), ignoring duplicate", "DAEMON");
        return;
    }

    LOG_INFO_CAT("MULTI-GPU: First valid block found by " + result.miningSystem + " (nonce: " + std::to_string(result.nonce) + "), processing...", "DAEMON");

    // 🚀 STOP ALL MINING: Signal all GPUs to stop mining this block (non-blocking)
    if (g_miner) {
        // Don't call stopMining() as it might cause deadlock - let natural completion handle it
        // g_miner->stopMining();
    }

    LOG_CLEAR_PROGRESS();
    // 🎯 MINING SYSTEM IDENTIFICATION: Show which system found the block
    std::string systemInfo = result.miningSystem.empty() ? "Unknown" : result.miningSystem;
    LOG_INFO_CAT("Block " + utils::CryptoUtils::hashToHex(result.hash) + " found by " + systemInfo + "! (Nonce: " + std::to_string(result.nonce) + ")", "RESULT");

    // Show nonce/timestamp mixing for first 10 iterations (matching JavaScript format)
    for (int i = 0; i < 10; i++) {
        uint32_t nonceIndex = i % 4;
        uint32_t timestampIndex = i % 4;

        // 🎯 CRITICAL FIX: Extract nonce word using SPECIFICATION-COMPLIANT zero-padding behavior
        // This matches the updated VELORA_ALGO.md specification and daemon implementation
        // NOTE: result.nonce is u32 (4 bytes), not u64 (8 bytes)
        uint32_t nonceWord = 0;
        for (int j = 0; j < 4; j++) {
            uint32_t bytePos = nonceIndex + j;
            if (bytePos < 4) {  // Nonce is only 4 bytes, not 8
                nonceWord |= ((result.nonce >> (bytePos * 8)) & 0xFF) << (j * 8);
            }
            // If bytePos >= 4, leave as zero (SPECIFICATION-COMPLIANT zero-padding)
        }

        // 🎯 CRITICAL FIX: Extract timestamp word using SPECIFICATION-COMPLIANT zero-padding behavior
        uint32_t timestampWord = 0;
        for (int j = 0; j < 4; j++) {
            uint32_t bytePos = timestampIndex + j;
            if (bytePos < 8) {
                timestampWord |= ((result.timestamp >> (bytePos * 8)) & 0xFF) << (j * 8);
            }
            // If bytePos >= 8, leave as zero (SPECIFICATION-COMPLIANT zero-padding)
        }
    }

    // If daemon mining is enabled, submit the block to the daemon
    if (g_miner && g_miner->isDaemonMining()) {
        LOG_INFO_CAT("Submitting block to daemon...", "DAEMON");

        // Use the EXACT same transaction data that was used for mining
        auto currentHeader = g_miner->getCurrentBlockHeader();

        // Use the stored final values that were used during mining
        u64 miningTimestamp = g_finalMiningTimestamp;
        std::string miningMerkleRoot = g_finalMiningMerkleRoot;

        // Use the EXACT same coinbase transaction that was used for mining
        rapidjson::Document doc;
        doc.SetObject();
        rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();

        // Reconstruct the coinbase transaction from stored JSON string
        rapidjson::Document coinbaseDoc;
        coinbaseDoc.Parse(g_finalCoinbaseJson.c_str());
        if (coinbaseDoc.HasParseError() || !coinbaseDoc.IsObject()) {
            LOG_ERROR_CAT("Failed to parse stored coinbase JSON for submission", "DAEMON");
            return;
        }

        rapidjson::Value coinbaseTx(rapidjson::kObjectType);
        coinbaseTx.CopyFrom(coinbaseDoc, allocator);

        // Create transactions array with the exact same coinbase transaction
        rapidjson::Value transactionsArray(rapidjson::kArrayType);
        transactionsArray.PushBack(coinbaseTx, allocator);

        // Create the block object using the EXACT same values used for mining
        rapidjson::Value blockObj(rapidjson::kObjectType);
        blockObj.AddMember("index", currentHeader.index, allocator);
        // 🎯 CRITICAL FIX: Use result.timestamp to ensure EXACT timestamp consistency
        // This guarantees the same timestamp used during mining is submitted to daemon
        blockObj.AddMember("timestamp", result.timestamp, allocator);
        blockObj.AddMember("transactions", transactionsArray, allocator);
        // Order per daemon expectation: previousHash, nonce, difficulty, hash, merkleRoot, algorithm
        blockObj.AddMember("previousHash", rapidjson::Value(currentHeader.previousHash.c_str(), allocator), allocator);
        blockObj.AddMember("nonce", result.nonce, allocator);
        blockObj.AddMember("difficulty", currentHeader.difficulty, allocator);
        blockObj.AddMember("hash", rapidjson::Value(utils::CryptoUtils::hashToHex(result.hash).c_str(), allocator), allocator);
        blockObj.AddMember("merkleRoot", rapidjson::Value(miningMerkleRoot.c_str(), allocator), allocator); // Use mining merkle root
        blockObj.AddMember("algorithm", "velora", allocator);

        doc.AddMember("block", blockObj, allocator);

        // Convert to JSON string
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        doc.Accept(writer);
        std::string jsonPayload = buffer.GetString();

        // Submit to daemon using the global config
        if (g_config && g_config->pool.daemon) {
            bool submitted = g_daemonClient->submitBlock(jsonPayload, g_config->pool.daemon_url, g_config->pool.daemon_api_key);
            if (submitted) {
                LOG_INFO_CAT("Block successfully submitted to daemon!", "DAEMON");

                // Signal that block submission is complete and ready for next block
                g_blockSubmissionComplete = true;

                LOG_INFO_CAT("Block submission complete, ready for next block", "DAEMON");
            } else {
                LOG_ERROR_CAT("Failed to submit block to daemon", "DAEMON");
                // 🎯 CRITICAL FIX: Set submission complete to true to exit WaitSubmit phase
                // When submission fails, we still need to exit the WaitSubmit phase
                g_blockSubmissionComplete = true;
                LOG_INFO_CAT("Block submission failed - will continue mining for new block", "DAEMON");
            }
        }
    }
}

// Error callback
void onError(const ErrorInfo& error) {
    LOG_CLEAR_PROGRESS();
    LOG_ERROR_CAT(error.message, "ERROR");
                        setPhase(MiningPhase::Idle, "Idle: cycle complete");
    if (!error.details.empty()) {
        LOG_ERROR_CAT("Details: " + error.details, "ERROR");
    }
}

// Print hardware status
void printHeaderText(const MinerConfig& config) {
    LOG_STATUS(" * ABOUT:       PASTELLA-MINER/1.0.0 MSVC/2019", "STATUS");

    if (config.gpu_enabled) {
        LOG_STATUS(" * GPU:         CUDA enabled", "STATUS");
        if (!config.cuda_devices.empty()) {
            // 🎯 MULTI-GPU SUPPORT: Show all enabled GPU devices
            for (size_t i = 0; i < config.cuda_devices.size(); i++) {
                const auto& gpu = config.cuda_devices[i];
                if (!gpu.enabled) continue;

                int threadsOut = gpu.threads;
                int blocksOut = gpu.blocks;
                if (!gpu.override_launch) {
                    cudaDeviceProp prop;
                    if (cudaGetDeviceProperties(&prop, gpu.device_id) == cudaSuccess) {
                        threadsOut = 512;
                        blocksOut = prop.multiProcessorCount * 12;
                    } else {
                        threadsOut = 512;
                        blocksOut = 0;
                    }
                }
                LOG_STATUS(" * CUDA:          Device " + std::to_string(i) + " enabled (threads: " + std::to_string(threadsOut) +
                         ", blocks: " + std::to_string(blocksOut) + ")", "STATUS");
            }
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
    LOG_STATUS("🚀 QUICK START (Auto-Configuration):", "HELP");
    LOG_STATUS("  ./pastella-miner                    # Auto-detect GPU and mine", "HELP");
    LOG_STATUS("", "HELP");
    LOG_STATUS("📋 PRESETS:", "HELP");
    LOG_STATUS("  ./pastella-miner -c config-auto.json      # Auto-detection (recommended)", "HELP");
    LOG_STATUS("  ./pastella-miner -c config-maximum.json   # Maximum performance", "HELP");
    LOG_STATUS("  ./pastella-miner -c config-balanced.json  # Balanced (gaming PCs)", "HELP");
    LOG_STATUS("  ./pastella-miner -c config-lowpower.json  # Low power (24/7 mining)", "HELP");
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

// Modern mining functions
bool startDaemonMining(const MinerConfig& config) {
    LOG_INFO_CAT("Starting daemon mining mode", "DAEMON");

    // Check daemon connectivity
    if (!g_daemonClient->checkConnectivity(config.pool.daemon_url)) {
        LOG_ERROR_CAT("Daemon is not reachable. Stopping mining.", "DAEMON");
        LOG_ERROR_CAT("Please check if the daemon is running at: " + config.pool.daemon_url, "DAEMON");
        return false;
    }

    // Create a single VeloraMiner instance that we'll reuse for all blocks
    auto veloraMiner = std::make_unique<velora::VeloraMiner>();
    g_miner = veloraMiner.get();

    // Set callbacks once
    veloraMiner->setProgressCallback(nullptr); // No longer needed
    veloraMiner->setErrorCallback(onError);
    veloraMiner->setHashFoundCallback(onHashFound);

    // Configure miner once
    veloraMiner->setMaxNonces(config.max_nonces);
    veloraMiner->setNumThreads(config.cpu_threads);
    veloraMiner->setCPUEnabled(config.cpu_enabled);
    veloraMiner->setDaemonMining(true);

    // 🎯 MULTI-GPU SUPPORT: Configure multiple GPUs from config.json
    if (config.gpu_enabled && !config.cuda_devices.empty()) {
        std::vector<GPUConfig> multiGPUConfigs;

        for (const auto& gpuDevice : config.cuda_devices) {
            if (!gpuDevice.enabled) {
                LOG_DEBUG("Skipping disabled GPU device " + std::to_string(gpuDevice.device_id), "DAEMON");
                continue;
            }

            GPUConfig gpuConfig;
            gpuConfig.deviceId = gpuDevice.device_id;

            // 🎯 FIXED MULTI-GPU LOGIC: Always use config values, fallback to auto-tune only if not specified
            if (gpuDevice.override_launch && gpuDevice.threads > 0 && gpuDevice.blocks > 0) {
                // Use explicit config values
                gpuConfig.threadsPerBlock = gpuDevice.threads;
                gpuConfig.blocksPerGrid = gpuDevice.blocks;
                LOG_DEBUG("Using explicit GPU config - threads: " + std::to_string(gpuDevice.threads) +
                         ", blocks: " + std::to_string(gpuDevice.blocks), "DAEMON");
            } else {
                // Auto-tune: use config values if available, otherwise auto-calculate
                if (gpuDevice.threads > 0 && gpuDevice.blocks > 0) {
                    gpuConfig.threadsPerBlock = gpuDevice.threads;
                    gpuConfig.blocksPerGrid = gpuDevice.blocks;
                    LOG_DEBUG("Using config values for auto-tune - threads: " + std::to_string(gpuDevice.threads) +
                             ", blocks: " + std::to_string(gpuDevice.blocks), "DAEMON");
                } else {
                    gpuConfig.threadsPerBlock = 0; // trigger auto-tune in algorithm init
                    gpuConfig.blocksPerGrid = 0;   // trigger auto-tune in algorithm init
                    LOG_DEBUG("No config values - will auto-tune in algorithm init", "DAEMON");
                }
            }
                        // 🚀 AUTO-CONFIGURATION: Apply smart GPU detection for batch size if needed
            if (gpuDevice.batch_size == 0) {
                LOG_INFO_CAT("APPLYING AUTO-CONFIGURATION for GPU " + std::to_string(gpuDevice.device_id), "CONFIG");

                // Run auto-detection for batch size
                cudaDeviceProp prop;
                cudaError_t err = cudaGetDeviceProperties(&prop, gpuDevice.device_id);
                if (err == cudaSuccess) {
                    LOG_INFO_CAT(std::string("   VRAM: ") + std::to_string(prop.totalGlobalMem / (1024 * 1024)) +
                        " MB, Compute: " + std::to_string(prop.major) + "." + std::to_string(prop.minor) +
                        ", SMs: " + std::to_string(prop.multiProcessorCount), "VELORA");


                    // 🚀 PRESET-AWARE CONFIGURATION: Read preset from config file
                    std::string presetType = "auto"; // Default

                    // Read preset from config.json
                    std::ifstream configFile("config.json");
                    if (configFile.is_open()) {
                        rapidjson::Document doc;
                        rapidjson::IStreamWrapper isw(configFile);
                        doc.ParseStream(isw);

                        if (!doc.HasParseError() && doc.HasMember("_preset") && doc["_preset"].IsString()) {
                            presetType = doc["_preset"].GetString();
                            LOG_INFO_CAT("   PRESET FROM CONFIG: " + presetType, "VELORA");
                        }
                        configFile.close();
                    }

                    // Calculate optimal batch size based on VRAM and preset
                    size_t scratchpadMemory = 67108864; // 64MB scratchpad
                    size_t reservedMemory = 256 * 1024 * 1024; // Reserve 256MB

                    // 🎯 PRESET-BASED MEMORY USAGE: Apply preset-specific memory usage
                    float memoryUsage;
                    std::string presetName;

                    if (presetType == "maximum") {
                        memoryUsage = 0.85f;
                        presetName = "Maximum Performance";
                    } else if (presetType == "balanced") {
                        memoryUsage = 0.75f;
                        presetName = "Balanced Performance";
                    } else if (presetType == "lowpower") {
                        memoryUsage = 0.60f;
                        presetName = "Low Power";
                    } else { // "auto" or any other value
                        // Auto-detect based on GPU capability
                        if (prop.major >= 8) {
                            memoryUsage = 0.85f; // High-end: Maximum performance
                            presetName = "Maximum Performance (Auto-detected)";
                        } else if (prop.major >= 7) {
                            memoryUsage = 0.75f; // Mid-range: Balanced
                            presetName = "Balanced Performance (Auto-detected)";
                        } else {
                            memoryUsage = 0.65f; // Older: Conservative
                            presetName = "Conservative Performance (Auto-detected)";
                        }
                    }

                    LOG_INFO_CAT(std::string("   PRESET APPLIED: ") +
                        presetName +
                        " (" + std::to_string(memoryUsage * 100.0f) +
                        "% VRAM)", "VELORA");

                    size_t availableMemory = (size_t)(prop.totalGlobalMem * memoryUsage) - scratchpadMemory - reservedMemory;
                    size_t bufferMemory = availableMemory / 2; // Double buffering
                    size_t maxNonces = bufferMemory / sizeof(uint32_t);

                    // Set optimal batch size based on GPU tier and available memory
                    if (prop.major >= 8) {
                        gpuConfig.maxNonces = std::min(maxNonces, static_cast<size_t>(1024000)); // 1M for high-end
                    } else if (prop.major >= 7) {
                        gpuConfig.maxNonces = std::min(maxNonces, static_cast<size_t>(512000));  // 512K for mid-range
                    } else {
                        gpuConfig.maxNonces = std::min(maxNonces, static_cast<size_t>(256000));  // 256K for older
                    }

                    // Ensure minimum batch size
                    gpuConfig.maxNonces = std::max(static_cast<size_t>(gpuConfig.maxNonces), static_cast<size_t>(32000));

                    LOG_INFO_CAT("   AUTO-CONFIG" +
                        std::string("   APPLIED: Batch size: ") +
                        std::to_string(static_cast<u32>(gpuConfig.maxNonces)) +
                        " nonces (" +
                        std::to_string(memoryUsage * 100.0f) +
                        "% VRAM usage)", "VELORA");
                } else {
                    LOG_ERROR_CAT("Failed to get GPU properties for device " + std::to_string(gpuDevice.device_id), "CONFIG");
                    gpuConfig.maxNonces = 128000; // Conservative default
                }
            } else {
                gpuConfig.maxNonces = gpuDevice.batch_size; // Use configured batch size
            }

            gpuConfig.useDoublePrecision = false;
            multiGPUConfigs.push_back(gpuConfig);

            LOG_DEBUG("Added GPU device " + std::to_string(gpuDevice.device_id) +
                     " with batch_size " + std::to_string(gpuDevice.batch_size), "DAEMON");
        }

        if (!multiGPUConfigs.empty()) {
            // Set multi-GPU configuration
            veloraMiner->setMultiGPUConfig(multiGPUConfigs);
            LOG_INFO_CAT("Multi-GPU configuration applied - " + std::to_string(multiGPUConfigs.size()) + " device(s)", "DAEMON");

                        // Log total combined batch size
            u64 totalBatchSize = veloraMiner->getTotalBatchSize();
            LOG_INFO_CAT("Total combined batch size: " + std::to_string(totalBatchSize) + " nonces per round", "DAEMON");
        } else {
            LOG_WARNING("No enabled GPU devices found - GPU mining disabled", "DAEMON");
            veloraMiner->setUseGPU(false);
        }
    } else {
        veloraMiner->setUseGPU(false);
    }

    // Start watchdog thread to monitor phases
    std::atomic<bool> watchdogRun(true);
    std::thread watchdog([&]() {
        while (watchdogRun.load()) {
            auto phase = static_cast<MiningPhase>(g_phase.load());
            u64 age = nowMs() - g_phaseTsMs.load();
            std::string phaseName;
            switch (phase) {
                case MiningPhase::Idle: phaseName = "Idle"; break;
                case MiningPhase::FetchLatest: phaseName = "FetchLatest"; break;
                case MiningPhase::PrepareTx: phaseName = "PrepareTx"; break;
                case MiningPhase::PrepareHeader: phaseName = "PrepareHeader"; break;
                case MiningPhase::StartMining: phaseName = "StartMining"; break;
                case MiningPhase::WaitMining: phaseName = "WaitMining"; break;
                case MiningPhase::WaitSubmit: phaseName = "WaitSubmit"; break;
                default: phaseName = "Unknown"; break;
            }
            LOG_DEBUG("Watchdog phase=" + phaseName + ", ageMs=" + std::to_string(age), "WATCHDOG");
            // Warn if stuck in risky phases too long
            if ((phase == MiningPhase::StartMining || phase == MiningPhase::FetchLatest) && age > 10000) {
                LOG_WARNING("Watchdog: phase taking longer than 10s: " + phaseName, "WATCHDOG");
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });

    // Start keyboard monitoring thread
    std::thread keyboardThread(keyboardMonitor);

    // Start hashrate stats thread: logs instant (30s) and 5m/1h averages every 30s
    std::atomic<bool> statsRun(true);
    std::thread statsThread([&]() {
        struct Sample { u64 tMs; u64 cumHashes; };
        std::vector<Sample> samples;
        samples.reserve(256);
        u64 lastT = nowMs();
        u64 lastHashes = 0;
        // Initialize lastHashes from miner if available
        try { if (g_miner) lastHashes = g_miner->getHashesProcessed(); } catch (...) {}
        u64 accHashes = 0; // monotonically increasing across miner restarts
        const u64 intervalMs = 2000; // 2s - match GPU batch timing to avoid hashrate oscillation
        const u64 window1m = 60ull * 1000ull;
        const u64 window5m = 5ull * 60ull * 1000ull;
        const u64 window1h = 60ull * 60ull * 1000ull;
        double emaHps = 0.0; // smoothed instantaneous hashrate
        const double alpha = 0.3; // smoothing factor for EMA
        auto fmtHps = [](double hps) -> std::string {
            const char* units[] = {"H/s", "KH/s", "MH/s", "GH/s", "TH/s", "PH/s"};
            int idx = 0;
            while (hps >= 1000.0 && idx < 5) {
                hps /= 1000.0;
                ++idx;
            }
            char out[32];
            std::snprintf(out, sizeof(out), "%.2f %s", hps, units[idx]);
            return std::string(out);
        };
        while (statsRun.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs));
            if (!statsRun.load()) break;
            if (!g_miner) continue;
            u64 now = nowMs();
            u64 raw = 0;
            try { raw = g_miner->getHashesProcessed(); } catch (...) { continue; }
            bool miningState = false;
            try { miningState = g_miner->isMining(); } catch (...) { miningState = false; }

            // Record sample
            // Compute delta; if miner reset detected (raw < lastHashes), treat delta as raw in this interval
            u64 dHashes = (raw >= lastHashes) ? (raw - lastHashes) : raw;
            u64 dMs = (now > lastT) ? (now - lastT) : intervalMs;
            // If not mining, pause stats accumulation and preserve last averages
            static double lastEma = 0.0, lastAvg1m = 0.0, lastAvg5 = 0.0, lastAvg1h = 0.0;
            static std::string lastPerCoreStr;
            if (!miningState) {
                // Reset baselines so downtime doesn't penalize the next interval
                lastT = now;
                lastHashes = raw;
                // Log last known values (do not update EMA/averages)
                std::string line = std::string("Hashrate: ") + fmtHps(lastEma)
                    + " | 1m avg: " + fmtHps(lastAvg1m)
                    + " | 5m avg: " + fmtHps(lastAvg5)
                    + " | 1h avg: " + fmtHps(lastAvg1h)
                    + lastPerCoreStr;
                LOG_INFO_CAT(line, "STATS");
                continue;
            }

            accHashes += dHashes;
            samples.push_back({ now, accHashes });

            // Prune older than 1h
            while (!samples.empty() && now - samples.front().tMs > window1h) samples.erase(samples.begin());

            // 🎯 ROLLING WINDOW HASH RATE: Use miner's smooth hashrate instead of delta calculation
            // Get the rolling window hashrate directly from the miner
            double rollingWindowHps = 0.0;
            try {
                rollingWindowHps = g_miner->getCurrentHashrate();
            } catch (...) {
                rollingWindowHps = 0.0;
            }

            // Use rolling window hashrate for EMA calculation
            if (emaHps <= 0.0) emaHps = rollingWindowHps;
            else emaHps = alpha * rollingWindowHps + (1.0 - alpha) * emaHps;

            // 1m average
            double avg1m = 0.0;
            for (size_t i = 0; i < samples.size(); ++i) {
                if (now - samples[i].tMs <= window1m) {
                    u64 baseHashes = samples[i].cumHashes;
                    u64 baseTime = samples[i].tMs;
                    u64 wMs = (now > baseTime) ? (now - baseTime) : 1;
                    u64 wHashes = (accHashes >= baseHashes) ? (accHashes - baseHashes) : 0;
                    avg1m = static_cast<double>(wHashes) * 1000.0 / static_cast<double>(wMs);
                    break;
                }
            }
            if (avg1m == 0.0 && !samples.empty()) {
                u64 baseHashes = samples.back().cumHashes;
                u64 baseTime = samples.back().tMs;
                u64 wMs = (now > baseTime) ? (now - baseTime) : 1;
                u64 wHashes = (accHashes >= baseHashes) ? (accHashes - baseHashes) : 0;
                avg1m = static_cast<double>(wHashes) * 1000.0 / static_cast<double>(wMs);
            }

            // 5m average
            double avg5 = 0.0;
            for (size_t i = 0; i < samples.size(); ++i) {
                if (now - samples[i].tMs <= window5m) {
                    u64 baseHashes = samples[i].cumHashes;
                    u64 baseTime = samples[i].tMs;
                    u64 wMs = (now > baseTime) ? (now - baseTime) : 1;
                    u64 wHashes = (accHashes >= baseHashes) ? (accHashes - baseHashes) : 0;
                    avg5 = static_cast<double>(wHashes) * 1000.0 / static_cast<double>(wMs);
                    break;
                }
            }
            // If no sample within 5m, use oldest available
            if (avg5 == 0.0 && !samples.empty()) {
                u64 baseHashes = samples.front().cumHashes;
                u64 baseTime = samples.front().tMs;
                u64 wMs = (now > baseTime) ? (now - baseTime) : 1;
                u64 wHashes = (accHashes >= baseHashes) ? (accHashes - baseHashes) : 0;
                avg5 = static_cast<double>(wHashes) * 1000.0 / static_cast<double>(wMs);
            }

            // 1h average
            double avg1h = 0.0;
            if (!samples.empty()) {
                u64 baseHashes = samples.front().cumHashes;
                u64 baseTime = samples.front().tMs;
                u64 wMs = (now > baseTime) ? (now - baseTime) : 1;
                u64 wHashes = (accHashes >= baseHashes) ? (accHashes - baseHashes) : 0;
                avg1h = static_cast<double>(wHashes) * 1000.0 / static_cast<double>(wMs);
            }

            // Per-thread hashrate since last interval
            std::string perCoreStr;
            std::string gpuHashrateStr;
            try {
                if (g_miner) {
                    auto per = g_miner->getPerThreadHashesProcessed();
                    static std::vector<u64> lastPer;
                    if (lastPer.size() != per.size()) lastPer.assign(per.size(), 0);

                    // Always show CPU status - either active threads or idle
                    if (!per.empty()) {
                        // compute deltas
                        perCoreStr.reserve(128);
                        perCoreStr = " | threads:";
                        for (size_t i = 0; i < per.size(); ++i) {
                            u64 delta = (per[i] >= lastPer[i]) ? (per[i] - lastPer[i]) : per[i];
                            double coreHps = dMs ? (static_cast<double>(delta) * 1000.0 / static_cast<double>(dMs)) : 0.0;
                            perCoreStr += " - #" + std::to_string(i) + ":" + fmtHps(coreHps);
                            lastPer[i] = per[i];
                        }
                    } else {
                        // No active CPU threads - show idle status
                        perCoreStr = " | CPU: idle";
                    }

                                        // 🎯 MULTI-GPU SUPPORT: Display individual hashrate for each GPU
                    if (g_miner->isGPUAvailable()) {
                        if (emaHps > 0.0) {
                            // Get individual GPU hashrates
                            std::vector<double> individualGPUHashrates;
                            try {
                                individualGPUHashrates = g_miner->getIndividualGPUHashrates();
                            } catch (...) {
                                // Fallback: use total hashrate for all GPUs
                                individualGPUHashrates.clear();
                            }

                            // Build multi-GPU display string
                            std::string gpuDisplay = " | GPU:";
                            int activeGPUCount = g_miner->getActiveGPUCount();

                            if (activeGPUCount > 0) {
                                // Show individual hashrate for each GPU
                                for (int i = 0; i < activeGPUCount; i++) {
                                    if (i > 0) gpuDisplay += " -";

                                    double gpuHashrate = 0.0;
                                    if (i < individualGPUHashrates.size()) {
                                        gpuHashrate = individualGPUHashrates[i];
                                    } else {
                                        // Fallback to total hashrate if individual not available
                                        gpuHashrate = emaHps;
                                    }

                                    gpuDisplay += " #" + std::to_string(i) + ":" + fmtHps(gpuHashrate);
                                }
                            } else {
                                // Fallback to single GPU display
                                gpuDisplay += " #0:" + fmtHps(emaHps);
                            }

                            gpuHashrateStr = gpuDisplay;
                        } else {
                            // GPU available but no hashrate yet
                            int activeGPUCount = g_miner->getActiveGPUCount();
                            if (activeGPUCount > 0) {
                                gpuHashrateStr = " | GPU: #0: idle";
                                for (int i = 1; i < activeGPUCount; i++) {
                                    gpuHashrateStr += " - #" + std::to_string(i) + ": idle";
                                }
                            } else {
                                gpuHashrateStr = " #0: idle";
                            }
                        }
                    } else {
                        // GPU not available - show this information
                        gpuHashrateStr = " | GPU: disabled";
                    }
                } else {
                    // No miner available
                    perCoreStr = " | CPU: no miner";
                    gpuHashrateStr = " | GPU: no miner";
                }
            } catch (...) {
                // Error occurred - show error state
                perCoreStr = " | CPU: error";
                gpuHashrateStr = " | GPU: error";
            }

            // Log human-readable units with enhanced display
            std::string line = std::string("Hashrate: ") + fmtHps(emaHps)
                + " | 1m avg: " + fmtHps(avg1m)
                + " | 5m avg: " + fmtHps(avg5)
                + " | 1h avg: " + fmtHps(avg1h)
                + perCoreStr
                + gpuHashrateStr
                + " | Total: " + fmtHps(emaHps);
            LOG_INFO_CAT(line, "STATS");

            lastT = now;
            lastHashes = raw;
            // Update last-known values for paused periods
            lastEma = emaHps; lastAvg1m = avg1m; lastAvg5 = avg5; lastAvg1h = avg1h; lastPerCoreStr = perCoreStr;
        }
    });

    // Continuous mining loop - mine blocks one after another
    LOG_INFO_CAT("Daemon mining mode - continuous mining enabled", "DAEMON");
    while (g_running) {
        try {
            // Reset block submission flag for new cycle
            g_blockSubmissionComplete = false;

            // 🚀 MULTI-GPU COORDINATION: Reset block found flag for new block
            g_blockFound.store(false);

            // Get block template from daemon (replaces latest block fetch)
            setPhase(MiningPhase::FetchLatest, "FetchLatest: begin");
            LOG_INFO_CAT("Getting mining template from daemon...", "DAEMON");
            DaemonBlock latestBlock; // will be filled as template
            int retryCount = 0;
            const int maxRetries = 5;

            while (retryCount < maxRetries && g_running) {
                try {
                    LOG_DEBUG("Attempting to get mining template (attempt " + std::to_string(retryCount + 1) + ")", "DAEMON");
                    DaemonBlock tpl;
                    if (g_daemonClient->getMiningTemplate(config.pool.daemon_url, config.pool.daemon_api_key, config.pool.wallet, tpl)) {
                        latestBlock = tpl;
                        LOG_INFO_CAT("Template received - next index: " + std::to_string(latestBlock.index) + ", prev: " + latestBlock.previousHash.substr(0, 16) + "...", "DAEMON");
                        break;
                    } else {
                        retryCount++;
                        LOG_WARNING("Failed to get mining template, retry " + std::to_string(retryCount) + "/" + std::to_string(maxRetries), "DAEMON");

                        if (retryCount < maxRetries) {
                            // Wait before retry, with exponential backoff
                            int waitTime = std::min(1000 * retryCount, 5000); // 1s, 2s, 3s, 4s, 5s max
                            LOG_INFO_CAT("Waiting " + std::to_string(waitTime) + "ms before retry...", "DAEMON");
                            std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
                        }
                    }
                } catch (const std::exception& e) {
                    retryCount++;
                    LOG_ERROR_CAT("Exception getting latest block: " + std::string(e.what()), "DAEMON");
                    if (retryCount < maxRetries) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    }
                } catch (...) {
                    retryCount++;
                    LOG_ERROR_CAT("Unknown exception getting latest block", "DAEMON");
                    if (retryCount < maxRetries) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    }
                }
            }

            if (latestBlock.index <= 0 && latestBlock.previousHash.empty()) {
                LOG_ERROR_CAT("Failed to get mining template from daemon after " + std::to_string(maxRetries) + " retries", "ERROR");
                // Wait a bit before trying again
                std::this_thread::sleep_for(std::chrono::seconds(5));
                continue;
            }

            LOG_INFO_CAT("Successfully retrieved latest block, preparing mining data...", "DAEMON");
            setPhase(MiningPhase::PrepareTx, "PrepareTx: before coinbase");

            // Prepare mining data
            u64 miningTimestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            LOG_DEBUG("Creating coinbase transaction...", "DAEMON");
            // Create coinbase transaction
            rapidjson::Document tempDoc;
            tempDoc.SetObject();
            rapidjson::Document::AllocatorType& tempAllocator = tempDoc.GetAllocator();

            // Declare variables outside try block
            rapidjson::Value coinbaseTx;
            std::string transactionId;
            std::string merkleRoot;

            try {
                coinbaseTx = g_transactionManager->createCoinbaseTransaction(config.pool.wallet, miningTimestamp, tempAllocator);
                LOG_DEBUG("Coinbase transaction created successfully", "DAEMON");

                // Calculate transaction ID and merkle root
                transactionId = g_transactionManager->calculateTransactionId(coinbaseTx);
                coinbaseTx["id"] = rapidjson::Value(transactionId.c_str(), tempAllocator);
                merkleRoot = g_transactionManager->calculateMerkleRoot(transactionId);
                LOG_DEBUG("Transaction ID and merkle root calculated", "DAEMON");

                // Store final data for block submission (serialize coinbase object only)
                {
                    rapidjson::StringBuffer coinbaseBuf;
                    rapidjson::Writer<rapidjson::StringBuffer> coinbaseWriter(coinbaseBuf);
                    coinbaseTx.Accept(coinbaseWriter);
                    g_finalCoinbaseJson = coinbaseBuf.GetString();
                }
                g_finalMiningTimestamp = miningTimestamp;
                g_finalMiningMerkleRoot = merkleRoot;
                LOG_DEBUG("Final mining data stored", "DAEMON");
            } catch (const std::exception& e) {
                LOG_ERROR_CAT("Exception creating coinbase transaction: " + std::string(e.what()), "ERROR");
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            } catch (...) {
                LOG_ERROR_CAT("Unknown exception creating coinbase transaction", "ERROR");
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }

            LOG_DEBUG("Creating block header...", "DAEMON");
            setPhase(MiningPhase::PrepareHeader, "PrepareHeader: before header set");
            // Create block header
            BlockHeader header;
            header.index = latestBlock.index; // template provides next index
            header.timestamp = latestBlock.timestamp ? latestBlock.timestamp : miningTimestamp;
            header.difficulty = latestBlock.difficulty;
            header.nonce = 0;
            header.previousHash = latestBlock.previousHash;
            header.merkleRoot = merkleRoot.empty() ? latestBlock.merkleRoot : merkleRoot;
            header.algorithm = "velora";

            LOG_INFO_CAT("Block header created - index: " + std::to_string(header.index) +
                    ", difficulty: " + std::to_string(header.difficulty) +
                    ", previousHash: " + header.previousHash.substr(0, 16) + "...", "DAEMON");

            // Reuse the existing VeloraMiner instance - just reset it for the new block
            if (g_miner) {
                LOG_DEBUG("Stopping previous mining session...", "DAEMON");
                g_miner->stopMining(); // Stop previous mining before setting new header
                LOG_DEBUG("Previous mining session stopped", "DAEMON");
            }

            // 🔒 COORDINATED MINING: Update block template and reset mining state
            LOG_DEBUG("🔒 COORDINATED MINING: Updating block template and resetting mining state...", "DAEMON");
            g_miner->updateBlockTemplate(header);

            LOG_INFO_CAT("Mining block " + std::to_string(header.index) +
                    " (Difficulty: " + std::to_string(header.difficulty) + ")", "DAEMON");

            LOG_DEBUG("Starting mining for block " + std::to_string(header.index), "DAEMON");
            setPhase(MiningPhase::StartMining, "StartMining: calling startMining");
            bool miningStarted = false;
            try {
                miningStarted = g_miner->startMining();
            } catch (const std::exception& e) {
                LOG_ERROR_CAT("Exception starting mining: " + std::string(e.what()), "ERROR");
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            } catch (...) {
                LOG_ERROR_CAT("Unknown exception starting mining", "ERROR");
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }

            if (!miningStarted) {
                LOG_ERROR_CAT("Failed to start mining for block " + std::to_string(header.index), "ERROR");
                // Wait before retrying
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }
            LOG_DEBUG("Mining started successfully for block " + std::to_string(header.index), "DAEMON");

            // Verify mining actually started
            bool isMining = false;
            try {
                isMining = g_miner->isMining();
            } catch (const std::exception& e) {
                LOG_ERROR_CAT("Exception checking mining status: " + std::string(e.what()), "ERROR");
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            } catch (...) {
                LOG_ERROR_CAT("Unknown exception checking mining status", "ERROR");
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }

            if (!isMining) {
                LOG_ERROR_CAT("Mining failed to start - isMining() returned false", "ERROR");
                // Wait before retrying
                std::this_thread::sleep_for(std::chrono::seconds(2));
                continue;
            }
            LOG_INFO_CAT("Mining verification passed - isMining() = true", "DAEMON");
            setPhase(MiningPhase::WaitMining, "WaitMining: loop while isMining");
            // Wait for mining to complete first (hash found callback will set g_blockSubmissionComplete)
            LOG_INFO_CAT("Waiting for mining to complete...", "DAEMON");
            while (g_running && g_miner->isMining()) {
                // watchdog periodic log
                static int tick = 0; tick = (tick + 1) % 50;
                if (tick == 0) {
                    LOG_DEBUG("Watchdog: mining alive (isMining=true)", "WATCHDOG");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // 🎯 CRITICAL FIX: Skip WaitSubmit phase if block submission is already complete
            if (g_running && !g_blockSubmissionComplete) {
                setPhase(MiningPhase::WaitSubmit, "WaitSubmit: confirm daemon height advanced");
                const int maxPoll = 20; // up to ~2s
                for (int i = 0; i < maxPoll && g_running; ++i) {
                    try {
                        DaemonBlock latestAfter = g_daemonClient->getLatestBlock(config.pool.daemon_url, config.pool.daemon_api_key);
                        if (latestAfter.index >= header.index) {
                            g_blockSubmissionComplete = true;
                            break;
                        }
                    } catch (...) {
                        // ignore and retry
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
            }

            // 🎯 CRITICAL FIX: Handle block submission success vs failure
            if (g_blockSubmissionComplete) {
                LOG_INFO_CAT("Block " + std::to_string(header.index) + " completed, moving to next block", "DAEMON");

                // Reset the flag for the next block
                g_blockSubmissionComplete = false;
                setPhase(MiningPhase::Idle, "Idle: cycle complete");
                continue;
            } else {
                // 🎯 BLOCK SUBMISSION FAILED OR NO BLOCK FOUND: Get new template and continue
                LOG_INFO_CAT("Block submission failed or no block found for block " + std::to_string(header.index) + " - getting new template", "DAEMON");
                setPhase(MiningPhase::Idle, "Idle: getting new template");
                continue; // Go back to get new template
            }

            // If we get here, something went wrong
            LOG_ERROR_CAT("Unexpected state in mining loop", "ERROR");
            // Wait before retrying
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;

        } catch (const std::exception& e) {
            LOG_ERROR_CAT("CRITICAL ERROR in mining loop: " + std::string(e.what()), "FATAL");
            LOG_ERROR_CAT("Stack trace location: " + std::string(__FILE__) + ":" + std::to_string(__LINE__), "FATAL");
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        } catch (...) {
            LOG_ERROR_CAT("UNKNOWN CRITICAL ERROR in mining loop", "FATAL");
            LOG_ERROR_CAT("Stack trace location: " + std::string(__FILE__) + ":" + std::to_string(__LINE__), "FATAL");
            std::this_thread::sleep_for(std::chrono::seconds(5));
            continue;
        }
    }

    LOG_INFO_CAT("Mining loop ended normally", "DAEMON");
    watchdogRun = false;
    if (watchdog.joinable()) watchdog.join();
    statsRun = false;
    if (statsThread.joinable()) statsThread.join();
    
    // Stop keyboard monitoring thread
    if (keyboardThread.joinable()) keyboardThread.join();
    
    return true;
}

// Pool mining functions
bool startPoolMining(const MinerConfig& config) {
    LOG_INFO_CAT("Starting pool mining mode", "POOL");

        // Parse pool URL to extract host and port
    std::string poolUrl = config.pool.url;
    int poolPort = config.pool.port;

    // Remove protocol prefix if present
    if (poolUrl.find("stratum+tcp://") == 0) {
        poolUrl = poolUrl.substr(14); // Remove "stratum+tcp://" prefix
    }

    // If port is in URL (after protocol removal), extract it
    size_t colonPos = poolUrl.find(":");
    if (colonPos != std::string::npos) {
        try {
            int urlPort = std::stoi(poolUrl.substr(colonPos + 1));
            if (urlPort > 0) {
                poolPort = urlPort; // Use port from URL if valid
            }
            poolUrl = poolUrl.substr(0, colonPos); // Remove port from URL
        } catch (...) {
            // Keep original values if parsing fails
        }
    }

    LOG_INFO_CAT("Pool URL: " + poolUrl + ", Port: " + std::to_string(poolPort), "POOL");
    LOG_INFO_CAT("Wallet: " + config.pool.wallet + ", Worker: " + config.pool.worker_name, "POOL");

    // Create pool miner
    auto poolMiner = std::make_unique<PoolMiner>();

    // Initialize pool connection
    if (!poolMiner->initialize(poolUrl, poolPort, config.pool.wallet, config.pool.worker_name)) {
        LOG_ERROR_CAT("Failed to initialize pool miner", "POOL");
        return false;
    }

    // Create VeloraMiner instance
    auto veloraMiner = std::make_unique<velora::VeloraMiner>();

    // Configure miner
    veloraMiner->setMaxNonces(config.max_nonces);
    veloraMiner->setNumThreads(config.cpu_threads);
    veloraMiner->setCPUEnabled(config.cpu_enabled);
    veloraMiner->setDaemonMining(false); // Pool mining mode

    // Configure GPU if enabled
    if (config.gpu_enabled && !config.cuda_devices.empty()) {
        std::vector<GPUConfig> multiGPUConfigs;

        for (const auto& gpuDevice : config.cuda_devices) {
            if (!gpuDevice.enabled) {
                continue;
            }

            GPUConfig gpuConfig;
            gpuConfig.deviceId = gpuDevice.device_id;
            gpuConfig.threadsPerBlock = gpuDevice.threads > 0 ? gpuDevice.threads : 512;
            gpuConfig.blocksPerGrid = gpuDevice.blocks > 0 ? gpuDevice.blocks : 512;
            gpuConfig.maxNonces = gpuDevice.batch_size > 0 ? gpuDevice.batch_size : 128000;
            gpuConfig.useDoublePrecision = false;

            multiGPUConfigs.push_back(gpuConfig);
            LOG_INFO_CAT("Added GPU device " + std::to_string(gpuDevice.device_id) +
                     " for pool mining", "POOL");
        }

        if (!multiGPUConfigs.empty()) {
            veloraMiner->setMultiGPUConfig(multiGPUConfigs);
            LOG_INFO_CAT("Multi-GPU configuration applied for pool mining - " +
                     std::to_string(multiGPUConfigs.size()) + " device(s)", "POOL");
        } else {
            veloraMiner->setUseGPU(false);
        }
    } else {
        veloraMiner->setUseGPU(false);
    }

    // Start pool mining
    if (!poolMiner->startMining(*veloraMiner, config)) {
        LOG_ERROR_CAT("Failed to start pool mining", "POOL");
        return false;
    }

    // Set global miner reference for keyboard monitoring
    g_miner = veloraMiner.get();

    // Start keyboard monitoring thread
    std::thread keyboardThread(keyboardMonitor);

    // Mining loop with reconnection logic
    LOG_INFO_CAT("Press Ctrl+C to stop mining or 'h' to display hashrate", "POOL");

    int reconnectAttempts = 0;

    while (g_running) {
        if (poolMiner->isConnected()) {
            // Reset reconnect attempts on successful connection
            reconnectAttempts = 0;

            // Log status periodically
            static int statusCounter = 0;
            if (++statusCounter >= 30) { // Every 30 seconds
                // Pool status logging removed for cleaner output
                statusCounter = 0;
            }

            std::this_thread::sleep_for(std::chrono::seconds(1));
        } else {
            // Connection lost, try to reconnect indefinitely
            reconnectAttempts++;
            LOG_INFO_CAT("Connection lost. Reconnection attempt " + std::to_string(reconnectAttempts), "POOL");

            // Stop current mining
            veloraMiner->stopMining();

            // Wait before reconnecting (progressive backoff up to 30 seconds)
            int waitTime = std::min(5 + (reconnectAttempts - 1) * 2, 30);
            std::this_thread::sleep_for(std::chrono::seconds(waitTime));

            // Try to reconnect and restart mining
            if (poolMiner->startMining(*veloraMiner, config)) {
                LOG_INFO_CAT("Reconnected to pool successfully", "POOL");
            } else {
                LOG_ERROR_CAT("Failed to reconnect to pool - will retry", "POOL");
            }
        }
    }

    // Stop mining
    poolMiner->stopMining();
    veloraMiner->stopMining();

    // Stop keyboard monitoring thread
    if (keyboardThread.joinable()) keyboardThread.join();

    LOG_INFO_CAT("Pool mining completed", "POOL");
    return true;
}

int main(int argc, char* argv[]) {
    // Set up signal handling
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    try {
        // Initialize configuration
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

        // Move auto-configuration messages to after header display

        // Validate configuration
        if (!configManager.validateConfig(config)) {
            return 0; // Exit gracefully for validation issues
        }

        // Set log level based on verbosity
        if (config.verbose) {
            LOGGER->setLogLevel(pastella::utils::LogLevel::DEBUG);
        } else {
            LOGGER->setLogLevel(pastella::utils::LogLevel::INFO);
        }

        // Print hardware status first
        printHeaderText(config);

        // Show configuration status after hardware info
        LOG_INFO_CAT("Configuration loaded from " + configPath, "CONFIG");
        if (config.gpu_enabled) {
            LOG_INFO_CAT("GPU mining enabled in config", "CONFIG");
        }

        // 🚀 AUTO-CONFIGURATION GUIDANCE: Help users understand auto-config options
        if (config.cuda_devices.empty() ||
            (config.cuda_devices.size() == 1 && config.cuda_devices[0].batch_size == 0)) {

            LOG_INFO_CAT("AUTO-CONFIGURATION DETECTED", "CONFIG");
            LOG_INFO_CAT("   The miner will automatically detect your GPU and optimize settings", "CONFIG");
            LOG_INFO_CAT("   To use manual configuration, edit config.json with specific values", "CONFIG");
            LOG_INFO_CAT("   For maximum performance, consider using config-auto.json as a template", "CONFIG");
        } else {
            LOG_INFO_CAT("MANUAL CONFIGURATION ACTIVE", "CONFIG");
            LOG_INFO_CAT("   Using custom GPU settings from config.json", "CONFIG");
            LOG_INFO_CAT("   To enable auto-configuration, set batch_size=0 and id=-1 in config", "CONFIG");
        }

        // Initialize components with proper ownership
        auto transactionManager = std::make_unique<transactions::TransactionManager>();
        auto daemonClient = std::make_unique<daemon::DaemonClient>();

        // Set global pointers
        g_transactionManager = transactionManager.get();
        g_daemonClient = daemonClient.get();
        g_config = &config;

        // Check daemon connectivity
        if (config.pool.daemon) {
            LOG_INFO_CAT("Daemon mining enabled - will mine to: " + config.pool.daemon_url, "CONFIG");

            if (!daemonClient->checkConnectivity(config.pool.daemon_url)) {
                LOG_ERROR_CAT("Daemon is not reachable. Stopping mining.", "CONFIG");
                LOG_ERROR_CAT("Please check if the daemon is running at: " + config.pool.daemon_url, "CONFIG");
                return 1;
            }

            // Start daemon mining
            if (startDaemonMining(config)) {
                LOG_INFO_CAT("Daemon mining completed successfully", "MAIN");
            } else {
                LOG_ERROR_CAT("Daemon mining failed", "MAIN");
                return 1;
            }

        } else {
            LOG_INFO_CAT("Pool mining mode selected", "CONFIG");

            // Start pool mining
            if (startPoolMining(config)) {
                LOG_INFO_CAT("Pool mining completed successfully", "MAIN");
            } else {
                LOG_ERROR_CAT("Pool mining failed", "MAIN");
                return 1;
            }
        }

    } catch (const std::exception& e) {
        LOG_FATAL(e.what(), "FATAL");
        return 1;
    }

    LOG_INFO_CAT("Mining completed successfully", "MAIN");
    return 0;
}
