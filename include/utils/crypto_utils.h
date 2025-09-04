#pragma once

#include "../types.h"
#include <string>
#include <vector>

// Conditional OpenSSL includes
#if defined(HAVE_OPENSSL)
  #if defined(__has_include)
    #if __has_include(<openssl/sha.h>) && __has_include(<openssl/evp.h>) && __has_include(<openssl/hmac.h>)
      #include <openssl/sha.h>
      #include <openssl/evp.h>
      #include <openssl/hmac.h>
    #else
      #undef HAVE_OPENSSL
    #endif
  #else
    // Fallback: assume headers exist if HAVE_OPENSSL set
    #include <openssl/sha.h>
    #include <openssl/evp.h>
    #include <openssl/hmac.h>
  #endif
#endif

namespace pastella {
namespace utils {

/**
 * Cryptographic utilities for Velora algorithm
 */
class CryptoUtils {
public:
    // SHA-256 hashing
    static Hash256 sha256(const std::vector<u8>& data);
    static Hash256 sha256(const std::string& data);
    static Hash256 sha256(const Hash256& data);
    static Hash256 sha256(const u8* data, size_t length);

    // Double hash (hex-based like daemon)
    static std::string doubleHash(const std::string& data);

    // Utility functions
    static std::string hashToHex(const Hash256& hash);
    static std::string hashToHex(const std::vector<u8>& hash);
    static Hash256 hexToHash(const std::string& hex);
    static std::vector<u8> hexToBytes(const std::string& hex);

    // Byte manipulation
    static std::vector<u8> toLittleEndian(u64 value);
    static std::vector<u8> toLittleEndian(u32 value);
    static u64 fromLittleEndian64(const std::vector<u8>& data, size_t offset = 0);
    static u32 fromLittleEndian32(const std::vector<u8>& data, size_t offset = 0);

    // Constants
    static constexpr u32 SHA256_BLOCK_SIZE = 64;
    static constexpr u32 SHA256_DIGEST_SIZE = 32;

private:
    // OpenSSL context management (only when OpenSSL is available)
#ifdef HAVE_OPENSSL
    static EVP_MD_CTX* createSha256Context();
    static void destroySha256Context(EVP_MD_CTX* ctx);
#endif

    // Internal helper methods
    static void initializeOpenSSL();
    static bool openSSLInitialized_;
};

} // namespace utils
} // namespace pastella
