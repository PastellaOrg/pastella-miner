#include "../../include/utils/crypto_utils.h"
#include "../../include/sha256.h"
#include <iomanip>
#include <sstream>
#include <random>
#include <cstring>

namespace pastella {
namespace utils {

// Static member initialization
bool CryptoUtils::openSSLInitialized_ = false;

// Internal: External SHA-256 implementation
static Hash256 sha256_external(const u8* data, size_t length) {
    Hash256 out{};
    std::fill(out.begin(), out.end(), 0); // Ensure clean initialization

    SHA256_CTX ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, length);
    sha256_final(&ctx, out.data());

    return out;
}

// SHA-256 implementations
Hash256 CryptoUtils::sha256(const std::vector<u8>& data) {
    return sha256_external(data.data(), data.size());
}

Hash256 CryptoUtils::sha256(const std::string& data) {
    return sha256_external(reinterpret_cast<const u8*>(data.data()), data.length());
}

Hash256 CryptoUtils::sha256(const Hash256& data) {
    return sha256_external(data.data(), data.size());
}

Hash256 CryptoUtils::sha256(const u8* data, size_t length) {
    return sha256_external(data, length);
}

// Double hash implementation matching daemon's hex-based approach
std::string CryptoUtils::doubleHash(const std::string& data) {
    // First hash: SHA256(data) -> hex string
    Hash256 firstHash = sha256(data);
    std::string firstHashHex = hashToHex(firstHash);

    // Second hash: SHA256(firstHashHex) -> hex string
    Hash256 secondHash = sha256(firstHashHex);
    std::string secondHashHex = hashToHex(secondHash);

    return secondHashHex;
}

// Utility functions
std::string CryptoUtils::hashToHex(const Hash256& hash) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (u8 byte : hash) {
        ss << std::setw(2) << static_cast<u32>(byte);
    }
    return ss.str();
}

std::string CryptoUtils::hashToHex(const std::vector<u8>& hash) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (u8 byte : hash) {
        ss << std::setw(2) << static_cast<u32>(byte);
    }
    return ss.str();
}

Hash256 CryptoUtils::hexToHash(const std::string& hex) {
    Hash256 result{};
    if (hex.length() != 64) return result;
    for (size_t i = 0; i < 32; i++) {
        std::string byteStr = hex.substr(i * 2, 2);
        try {
            result[i] = static_cast<u8>(std::stoi(byteStr, nullptr, 16));
        } catch (...) {
            return Hash256{};
        }
    }
    return result;
}

std::vector<u8> CryptoUtils::hexToBytes(const std::string& hex) {
    std::vector<u8> result;
    if (hex.length() % 2 != 0) return result; // Hex string must have even length

    result.reserve(hex.length() / 2);
    for (size_t i = 0; i < hex.length(); i += 2) {
        std::string byteStr = hex.substr(i, 2);
        try {
            result.push_back(static_cast<u8>(std::stoi(byteStr, nullptr, 16)));
        } catch (...) {
            return std::vector<u8>{}; // Return empty vector on error
        }
    }
    return result;
}

// Byte manipulation
std::vector<u8> CryptoUtils::toLittleEndian(u64 value) {
    std::vector<u8> result(8);
    for (int i = 0; i < 8; i++) { result[i] = static_cast<u8>(value & 0xFF); value >>= 8; }
    return result;
}

std::vector<u8> CryptoUtils::toLittleEndian(u32 value) {
    std::vector<u8> result(4);
    for (int i = 0; i < 4; i++) { result[i] = static_cast<u8>(value & 0xFF); value >>= 8; }
    return result;
}

u64 CryptoUtils::fromLittleEndian64(const std::vector<u8>& data, size_t offset) {
    if (offset + 8 > data.size()) return 0;
    u64 result = 0; for (int i = 0; i < 8; i++) result |= static_cast<u64>(data[offset + i]) << (i * 8);
    return result;
}

u32 CryptoUtils::fromLittleEndian32(const std::vector<u8>& data, size_t offset) {
    if (offset + 4 > data.size()) return 0;
    u32 result = 0; for (int i = 0; i < 4; i++) result |= static_cast<u32>(data[offset + i]) << (i * 8);
    return result;
}

#ifdef HAVE_OPENSSL
EVP_MD_CTX* CryptoUtils::createSha256Context() { return EVP_MD_CTX_new(); }
void CryptoUtils::destroySha256Context(EVP_MD_CTX* ctx) { if (ctx) EVP_MD_CTX_free(ctx); }
#endif

void CryptoUtils::initializeOpenSSL() {
#ifdef HAVE_OPENSSL
    if (!openSSLInitialized_) {
        // OpenSSL 1.1+ initializes itself; nothing required here
        openSSLInitialized_ = true;
    }
#else
    (void)openSSLInitialized_;
#endif
}

} // namespace utils
} // namespace pastella
