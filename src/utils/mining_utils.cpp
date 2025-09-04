#include "../../include/utils/mining_utils.h"
#include "../../include/utils/logger.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <random>
#include <chrono>
#include <cmath>
#include <bitset>

namespace pastella {

// Velora-specific difficulty checking - EXACTLY matches daemon logic
bool MiningUtils::veloraHashMeetsDifficulty(const Hash256& hash, u32 difficulty) {
    // Exact target compare: hash <= ((2^256 - 1) / difficulty)
    if (difficulty == 0) return true;

    // Compute target as big-endian byte array via division of 0xFF..FF by difficulty
    u8 target[32];
    u32 carry = 0;
    for (size_t i = 0; i < 32; ++i) {
        u32 word = (carry << 8) | 0xFFu;
        target[i] = static_cast<u8>(word / difficulty);
        carry = word % difficulty;
    }

    // Compare hash (big-endian) to target
    for (size_t i = 0; i < 32; ++i) {
        u8 hb = hash[i];
        u8 tb = target[i];
        if (hb < tb) return true;
        if (hb > tb) return false;
    }
    return true;
}

// Hex conversion utilities (used by veloraHashMeetsDifficulty)
std::string MiningUtils::bytesToHexString(const std::vector<u8>& bytes) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (u8 byte : bytes) {
        ss << std::setw(2) << static_cast<int>(byte);
    }
    return ss.str();
}

} // namespace pastella
