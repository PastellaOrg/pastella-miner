#include "../include/config_manager.h"
#include "../include/mining_types.h"
#include "../include/utils/logger.h"
#include "rapidjson/document.h"
#include "rapidjson/reader.h"
#include "rapidjson/filereadstream.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace pastella {

ConfigManager::ConfigManager() : defaultConfigPath("config.json") {}

MinerConfig ConfigManager::loadConfig(const std::string& configPath) {
    MinerConfig config;

    // Set defaults - both disabled by default
    config.gpu_enabled = false;
    config.cpu_enabled = false;

    std::ifstream file(configPath);
    if (!file.is_open()) {
        LOG_INFO_CAT("Configuration file not found, will create default", "CONFIG");
        return config;
    }

    // Read entire file content
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string jsonContent = buffer.str();
    file.close();

    // Parse JSON using RapidJSON
    rapidjson::Document doc;
    if (doc.Parse(jsonContent.c_str()).HasParseError()) {
        LOG_ERROR_CAT("Failed to parse JSON: Parse error at offset " + std::to_string(doc.GetErrorOffset()), "CONFIG");
        return config;
    }

    // Load different configuration sections
    loadPoolConfig(doc, config);
    loadGPUConfig(doc, config);
    loadCPUConfig(doc, config);

    // Load other settings
    if (doc.HasMember("max_nonces")) {
        config.max_nonces = doc["max_nonces"].GetUint64();
    }
    if (doc.HasMember("verbose")) {
        config.verbose = doc["verbose"].GetBool();
    }

    // LOG_INFO_CAT("Configuration loaded from " + configPath, "CONFIG"); // Moved to after header
    return config;
}

bool ConfigManager::saveDefaultConfig(const std::string& configPath) {
    std::ofstream file(configPath);
    if (file.is_open()) {
        file << "{\n";
        file << "  \"pool\": {\n";
        file << "    \"url\": \"stratum+tcp://pool.pastella.org:3333\",\n";
        file << "    \"wallet\": \"YOUR_WALLET_ADDRESS\",\n";
        file << "    \"worker\": \"pastella-miner\",\n";
        file << "    \"daemon\": false,\n";
        file << "    \"daemon_url\": \"http://localhost:22000\",\n";
        file << "    \"daemon_api_key\": \"\"\n";
        file << "  },\n";
        file << "  \"cuda\": {\n";
        file << "    \"devices\": [\n";
        file << "      {\n";
        file << "        \"id\": 0,\n";
        file << "        \"threads\": 256,\n";
        file << "        \"blocks\": 1024,\n";
        file << "        \"batch_size\": 128000,\n";
        file << "        \"override_launch\": true,\n";
        file << "        \"enabled\": false\n";
        file << "      }\n";
        file << "    ]\n";
        file << "  },\n";
        file << "  \"cpu\": {\n";
        file << "    \"enabled\": false,\n";
        file << "    \"threads\": 4\n";
        file << "  },\n";
        file << "  \"max_nonces\": 10000000000,\n";
        file << "  \"verbose\": false\n";
        file << "}\n";
        file.close();
        LOG_INFO_CAT("Default configuration saved to " + configPath, "CONFIG");
        return true;
    } else {
        LOG_ERROR_CAT("Failed to open config file for writing: " + configPath, "CONFIG");
        return false;
    }
}

bool ConfigManager::loadOrCreateConfig(const std::string& configPath, MinerConfig& config) {
    // Try to load configuration file
    if (std::ifstream(configPath)) {
        config = loadConfig(configPath);
        // LOG_INFO_CAT("Loaded configuration from " + configPath, "CONFIG"); // Moved to after header
        return true;
    } else {
        LOG_INFO_CAT("Configuration file not found, creating default...", "CONFIG");
        if (saveDefaultConfig(configPath)) {
            LOG_INFO_CAT("Default configuration created successfully at " + configPath, "CONFIG");
            config = loadConfig(configPath);
            return true;
        } else {
            LOG_ERROR_CAT("Failed to create default configuration file", "CONFIG");
            return false;
        }
    }
}

bool ConfigManager::validateConfig(const MinerConfig& config) {
    // Check if any mining mode is enabled
    if (!config.gpu_enabled && !config.cpu_enabled) {
        LOG_INFO_CAT("No mining mode enabled in configuration", "CONFIG");
        LOG_INFO_CAT("Please edit config.json to enable either CPU or GPU mining", "CONFIG");
        LOG_INFO_CAT("Or use command line arguments: --cpu or --cuda", "CONFIG");
        LOG_INFO_CAT("", "CONFIG");
        LOG_INFO_CAT("Example config.json:", "CONFIG");
        LOG_INFO_CAT("  \"cuda\": { \"devices\": [{ \"enabled\": true }] }", "CONFIG");
        LOG_INFO_CAT("  \"cpu\": { \"enabled\": true, \"threads\": 4 }", "CONFIG");
        return false;
    }

    // Check daemon mining requirements: allow CPU or GPU
    if (config.pool.daemon && !(config.cpu_enabled || config.gpu_enabled)) {
        LOG_ERROR_CAT("Daemon mining requires mining to be enabled (CPU or GPU)", "CONFIG");
        LOG_ERROR_CAT("Enable CPU with --cpu or GPU in config.json (cuda.devices[].enabled)", "CONFIG");
        return false;
    }

    return true;
}

bool ConfigManager::parseCommandLine(int argc, char* argv[], MinerConfig& config, std::string& configPath) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            return true; // Help requested
        } else if (arg == "-c" || arg == "--config") {
            if (++i < argc) {
                configPath = argv[i];
            }
        } else if (arg == "-o" || arg == "--url") {
            if (++i < argc) config.pool.url = argv[i];
        } else if (arg == "-u" || arg == "--user") {
            if (++i < argc) config.pool.wallet = argv[i];
        } else if (arg == "-w" || arg == "--worker") {
            if (++i < argc) config.pool.worker_name = argv[i];
        } else if (arg == "--cuda") {
            config.gpu_enabled = true;
            config.cpu_enabled = false;
            LOG_INFO_CAT("GPU mining enabled via command line", "CONFIG");
        } else if (arg == "--cpu") {
            config.cpu_enabled = true;
            config.gpu_enabled = false;
            LOG_INFO_CAT("CPU mining enabled via command line", "CONFIG");
        } else if (arg == "--daemon") {
            config.pool.daemon = true;
            LOG_INFO_CAT("Daemon mining enabled via command line", "CONFIG");
        } else if (arg == "--daemon-url") {
            if (++i < argc) {
                config.pool.daemon_url = argv[i];
                LOG_INFO_CAT("Daemon URL set to: " + config.pool.daemon_url, "CONFIG");
            }
        } else if (arg == "--daemon-api-key") {
            if (++i < argc) {
                config.pool.daemon_api_key = argv[i];
                LOG_INFO_CAT("Daemon API key set via command line", "CONFIG");
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (++i < argc) config.cpu_threads = std::stoi(argv[i]);
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
        }
    }

    return false; // No help requested
}

void ConfigManager::loadPoolConfig(const rapidjson::Document& doc, MinerConfig& config) {
    if (doc.HasMember("pool")) {
        const auto& pool = doc["pool"];
        if (pool.HasMember("url")) {
            config.pool.url = pool["url"].GetString();
        }
        if (pool.HasMember("wallet")) {
            config.pool.wallet = pool["wallet"].GetString();
        }
        if (pool.HasMember("worker")) {
            config.pool.worker_name = pool["worker"].GetString();
        }
        if (pool.HasMember("daemon")) {
            config.pool.daemon = pool["daemon"].GetBool();
        }
        if (pool.HasMember("daemon_url")) {
            config.pool.daemon_url = pool["daemon_url"].GetString();
        }
        if (pool.HasMember("daemon_api_key")) {
            config.pool.daemon_api_key = pool["daemon_api_key"].GetString();
        }
    }
}

void ConfigManager::loadGPUConfig(const rapidjson::Document& doc, MinerConfig& config) {
    if (doc.HasMember("cuda")) {
        const auto& cuda = doc["cuda"];
        if (cuda.HasMember("devices")) {
            const auto& devices = cuda["devices"];
            if (devices.IsArray()) {
                for (const auto& device : devices.GetArray()) {
                    if (device.HasMember("enabled") && device["enabled"].GetBool()) {
                        config.gpu_enabled = true;
                        GPUDeviceConfig gpuDevice;

                        if (device.HasMember("id")) {
                            gpuDevice.device_id = device["id"].GetInt();
                        }
                        if (device.HasMember("threads")) {
                            gpuDevice.threads = device["threads"].GetInt();
                        }
                        if (device.HasMember("blocks")) {
                            gpuDevice.blocks = device["blocks"].GetInt();
                        }
                        if (device.HasMember("batch_size")) {
                            gpuDevice.batch_size = device["batch_size"].GetUint64();
                        }
                        if (device.HasMember("override_launch")) {
                            gpuDevice.override_launch = device["override_launch"].GetBool();
                        }

                        config.cuda_devices.push_back(gpuDevice);
                        // LOG_INFO_CAT("GPU mining enabled in config", "CONFIG"); // Moved to after header
                    }
                }
            }
        }
    }
}

void ConfigManager::loadCPUConfig(const rapidjson::Document& doc, MinerConfig& config) {
    if (doc.HasMember("cpu")) {
        const auto& cpu = doc["cpu"];
        if (cpu.HasMember("enabled")) {
            config.cpu_enabled = cpu["enabled"].GetBool();
            if (config.cpu_enabled) {
                LOG_INFO_CAT("CPU mining enabled in config", "CONFIG");

                if (cpu.HasMember("threads")) {
                    config.cpu_threads = cpu["threads"].GetInt();
                    LOG_INFO_CAT("CPU threads set to: " + std::to_string(config.cpu_threads), "CONFIG");
                } else {
                    LOG_INFO_CAT("CPU threads not found in config, using default: " + std::to_string(config.cpu_threads), "CONFIG");
                }
            }
        }
    } else {
        LOG_INFO_CAT("CPU section not found in config", "CONFIG");
    }
}

} // namespace pastella
