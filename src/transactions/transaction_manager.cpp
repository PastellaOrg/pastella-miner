#include "../../include/transactions/transaction_manager.h"
#include "../../include/utils/crypto_utils.h"
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <chrono>
#include <random>

namespace pastella {
namespace transactions {

TransactionManager::TransactionManager()
    : randomGenerator(std::chrono::system_clock::now().time_since_epoch().count()),
      randomDistribution(0.0, 1.0) {
    // Constructor implementation
}

TransactionManager::~TransactionManager() {
    // Destructor implementation
}

rapidjson::Value TransactionManager::createCoinbaseTransaction(const std::string& walletAddress, u64 timestamp, rapidjson::Document::AllocatorType& allocator) {
    // Generate unique nonce and atomic sequence using the same logic as the daemon
    std::string nonce = generateNonce();
    std::string atomicSequence = generateAtomicSequence();

    // Create coinbase transaction exactly like daemon does
    rapidjson::Value coinbaseTx(rapidjson::kObjectType);

    // Create mining reward output FIRST
    rapidjson::Value rewardOutput(rapidjson::kObjectType);
    rewardOutput.AddMember("address", rapidjson::Value(walletAddress.c_str(), allocator), allocator);
    rewardOutput.AddMember("amount", COINBASE_REWARD, allocator);
    rewardOutput.AddMember("scriptPubKey", rapidjson::Value("OP_DUP OP_HASH160 1Ggdo3kTzGvmhSDuoZBuFZQeqDSg57FEf6 OP_EQUALVERIFY OP_CHECKSIG", allocator), allocator);

    // Create outputs array and add the reward output
    rapidjson::Value outputsArray(rapidjson::kArrayType);
    outputsArray.PushBack(rewardOutput, allocator);

    // Create inputs array (empty for coinbase)
    rapidjson::Value inputsArray(rapidjson::kArrayType);

    // Now build the complete transaction
    coinbaseTx.AddMember("id", rapidjson::Value("", allocator), allocator); // Empty - will be calculated
    coinbaseTx.AddMember("inputs", inputsArray, allocator);
    coinbaseTx.AddMember("outputs", outputsArray, allocator);
    coinbaseTx.AddMember("fee", 0, allocator);
    coinbaseTx.AddMember("timestamp", timestamp, allocator);
    coinbaseTx.AddMember("isCoinbase", true, allocator);
    coinbaseTx.AddMember("tag", "COINBASE", allocator);
    coinbaseTx.AddMember("nonce", rapidjson::Value(nonce.c_str(), allocator), allocator);
    coinbaseTx.AddMember("expiresAt", timestamp + (COINBASE_EXPIRY_HOURS * 60 * 60 * 1000), allocator); // 24 hours
    coinbaseTx.AddMember("sequence", 0, allocator);
    coinbaseTx.AddMember("atomicSequence", rapidjson::Value(atomicSequence.c_str(), allocator), allocator);

    // Calculate transaction ID from the complete transaction data
    std::string transactionId = calculateTransactionId(coinbaseTx);
    coinbaseTx["id"] = rapidjson::Value(transactionId.c_str(), allocator);

    return coinbaseTx;
}

std::string TransactionManager::calculateTransactionId(const rapidjson::Value& transaction) {
    // Convert the entire transaction to JSON string for hashing
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    transaction.Accept(writer);
    std::string jsonString = buffer.GetString();

    // Calculate SHA256 hash of the JSON string
    Hash256 hash = utils::CryptoUtils::sha256(jsonString);
    std::string transactionId = utils::CryptoUtils::hashToHex(hash);

    return transactionId;
}

std::string TransactionManager::calculateMerkleRoot(const std::string& transactionId) {
    // For single transaction (coinbase), merkle root is the hash of the transaction ID
    Hash256 singleHash = utils::CryptoUtils::sha256(transactionId);
    return utils::CryptoUtils::hashToHex(singleHash);
}

std::string TransactionManager::generateNonce() {
    // CRITICAL: Match the daemon's generateNonce() logic exactly
    // From daemon: return Date.now().toString(36) + Math.random().toString(36).substr(2) + Math.random().toString(36).substr(2);

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    // Convert timestamp to base36
    std::string timestampBase36 = convertToBase36(timestamp);

    // Generate two random values and convert to base36
    double rand1 = randomDistribution(randomGenerator);
    double rand2 = randomDistribution(randomGenerator);

    // Convert to base36 string (like daemon's .toString(36).substr(2))
    u64 rand1Int = static_cast<u64>(rand1 * 1000000000);
    u64 rand2Int = static_cast<u64>(rand2 * 1000000000);

    std::string rand1Base36 = convertToBase36(rand1Int);
    std::string rand2Base36 = convertToBase36(rand2Int);

    // Ensure minimum length (like daemon's .substr(2))
    if (rand1Base36.length() < 2) rand1Base36 = "00" + rand1Base36;
    if (rand2Base36.length() < 2) rand2Base36 = "00" + rand2Base36;

    // Concatenate like daemon: timestamp + rand1 + rand2
    return timestampBase36 + rand1Base36 + rand2Base36;
}

std::string TransactionManager::generateAtomicSequence() {
    // CRITICAL: Match the daemon's generateAtomicSequence() logic exactly
    // From daemon: return `${timestamp}-${random}-${processId}-${threadId}`;

    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

    // Generate random value in base36
    double random = randomDistribution(randomGenerator);
    u64 randomInt = static_cast<u64>(random * 1000000000);
    std::string randomBase36 = convertToBase36(randomInt);

    // Ensure minimum length and truncate to 9 characters (like daemon's .substr(2, 9))
    if (randomBase36.length() < 9) {
        randomBase36 = std::string(9 - randomBase36.length(), '0') + randomBase36;
    } else if (randomBase36.length() > 9) {
        randomBase36 = randomBase36.substr(0, 9);
    }

    // Process ID and Thread ID (simplified - use random numbers)
    u32 processId = static_cast<u32>(randomDistribution(randomGenerator) * 100000);
    u32 threadId = static_cast<u32>(randomDistribution(randomGenerator) * 1000000);

    // Format: timestamp-random-processId-threadId
    return std::to_string(timestamp) + "-" + randomBase36 + "-" + std::to_string(processId) + "-" + std::to_string(threadId);
}

bool TransactionManager::isValidTransaction(const rapidjson::Value& transaction) {
    // Basic transaction validation
    if (!transaction.IsObject()) return false;
    if (!transaction.HasMember("id") || !transaction["id"].IsString()) return false;
    if (!transaction.HasMember("timestamp") || !transaction["timestamp"].IsUint64()) return false;
    if (!transaction.HasMember("inputs") || !transaction["inputs"].IsArray()) return false;
    if (!transaction.HasMember("outputs") || !transaction["outputs"].IsArray()) return false;

    return true;
}

bool TransactionManager::isValidCoinbaseTransaction(const rapidjson::Value& transaction) {
    if (!isValidTransaction(transaction)) return false;

    // Coinbase-specific validation
    if (!transaction.HasMember("isCoinbase") || !transaction["isCoinbase"].IsBool() || !transaction["isCoinbase"].GetBool()) return false;
    if (!transaction.HasMember("tag") || !transaction["tag"].IsString() || std::string(transaction["tag"].GetString()) != "COINBASE") return false;
    if (!transaction["inputs"].GetArray().Empty()) return false; // Coinbase should have no inputs

    return true;
}

std::string TransactionManager::convertToBase36(u64 value) {
    if (value == 0) return "0";

    std::string result = "";
    while (value > 0) {
        result = BASE36_CHARS[value % 36] + result;
        value /= 36;
    }
    return result;
}

u64 TransactionManager::fromBase36(const std::string& base36) {
    u64 result = 0;
    u64 power = 1;

    for (int i = base36.length() - 1; i >= 0; i--) {
        char c = base36[i];
        u64 digit = 0;

        if (c >= '0' && c <= '9') {
            digit = c - '0';
        } else if (c >= 'a' && c <= 'z') {
            digit = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'Z') {
            digit = c - 'A' + 10;
        }

        result += digit * power;
        power *= 36;
    }

    return result;
}

} // namespace transactions
} // namespace pastella
