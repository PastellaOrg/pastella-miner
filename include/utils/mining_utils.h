#pragma once

#include "../mining_types.h"
#include "../types.h"
#include <string>
#include <chrono>

namespace pastella {

/**
 * Mining Utilities - Common functions used across mining operations
 */
class MiningUtils {
public:
    // Velora-specific difficulty checking
    static bool veloraHashMeetsDifficulty(const Hash256& hash, u32 difficulty);

    // Hex conversion utilities (used by veloraHashMeetsDifficulty)
    static std::string bytesToHexString(const std::vector<u8>& bytes);

private:
    // Helper methods
    static std::string sha256(const std::string& data);
    static std::string doubleSha256(const std::string& data);
};

} // namespace pastella
