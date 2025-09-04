#pragma once

#include <string>
#include <vector>
#include <memory>
#include "rapidjson/document.h"

namespace pastella {

// Forward declarations
struct PoolConfig;
struct GPUDeviceConfig;
struct MinerConfig;

/**
 * Configuration Manager for Pastella Miner
 * Handles loading, saving, and managing all configuration options
 */
class ConfigManager {
public:
    ConfigManager();
    ~ConfigManager() = default;

    // Load configuration from file
    MinerConfig loadConfig(const std::string& configPath);

    // Save default configuration
    bool saveDefaultConfig(const std::string& configPath);

    // Parse command line arguments
    // Returns true if help was requested, false otherwise
    // configPath is updated if --config is specified
    bool parseCommandLine(int argc, char* argv[], MinerConfig& config, std::string& configPath);

    // Load or create configuration (handles file creation if needed)
    // Returns true if config was loaded successfully, false otherwise
    bool loadOrCreateConfig(const std::string& configPath, MinerConfig& config);

    // Validate configuration
    // Returns true if config is valid, false otherwise
    bool validateConfig(const MinerConfig& config);

private:
    std::string defaultConfigPath;

    // Helper methods
    void loadPoolConfig(const rapidjson::Document& doc, MinerConfig& config);
    void loadGPUConfig(const rapidjson::Document& doc, MinerConfig& config);
    void loadCPUConfig(const rapidjson::Document& doc, MinerConfig& config);
};

} // namespace pastella
