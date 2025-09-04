#pragma once
#include "../mining_types.h"
#include <rapidjson/document.h>
#include <string>
#include <random>

namespace pastella {
namespace transactions {

/**
 * Transaction Manager - Handles all transaction operations
 * Creates, validates, and manages transactions
 */
class TransactionManager {
public:
    TransactionManager();
    ~TransactionManager();

    // Create transactions
    rapidjson::Value createCoinbaseTransaction(const std::string& walletAddress, u64 timestamp, rapidjson::Document::AllocatorType& allocator);

    // Calculate transaction properties
    std::string calculateTransactionId(const rapidjson::Value& transaction);
    std::string calculateMerkleRoot(const std::string& transactionId);

    // Generate unique identifiers
    std::string generateNonce();
    std::string generateAtomicSequence();

    // Validate transactions
    bool isValidTransaction(const rapidjson::Value& transaction);
    bool isValidCoinbaseTransaction(const rapidjson::Value& transaction);

private:
    // Helper methods
    std::string convertToBase36(u64 value);
    u64 fromBase36(const std::string& base36);

    // Constants
    static constexpr u64 COINBASE_REWARD = 5000000000; // 50 PAS in atomic units
    static constexpr u64 COINBASE_EXPIRY_HOURS = 24;
    static constexpr const char* COINBASE_SCRIPT = "OP_DUP OP_HASH160 1Ggdo3kTzGvmh";

    // Base36 characters
    static constexpr const char* BASE36_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz";

    // Random number generation
    std::mt19937 randomGenerator;
    std::uniform_real_distribution<double> randomDistribution;
};

} // namespace transactions
} // namespace pastella
