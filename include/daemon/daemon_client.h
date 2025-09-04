#pragma once
#include "../mining_types.h"
#include <string>
#include <curl/curl.h>

namespace pastella {
namespace daemon {

/**
 * Daemon Client - Handles communication with the blockchain daemon
 * Manages daemon connectivity, data retrieval, and block submission
 */
class DaemonClient {
public:
    DaemonClient();
    ~DaemonClient();

    // Daemon operations
    bool checkConnectivity(const std::string& daemonUrl);
    DaemonBlock getLatestBlock(const std::string& daemonUrl, const std::string& apiKey);
    // New: fetch mining template
    bool getMiningTemplate(const std::string& daemonUrl,
                           const std::string& apiKey,
                           const std::string& address,
                           DaemonBlock& outBlock);
    bool submitBlock(const std::string& jsonPayload, const std::string& daemonUrl, const std::string& apiKey);

    // URL and configuration
    void setDaemonUrl(const std::string& url);
    void setApiKey(const std::string& apiKey);

private:
    // Configuration
    std::string daemonUrl_;
    std::string apiKey_;

    // CURL options
    static constexpr u32 CURL_TIMEOUT = 10L;
    static constexpr u32 CURL_CONNECT_TIMEOUT = 5L;
    static constexpr u32 SUBMISSION_TIMEOUT = 30L;

    // Internal methods
    void setCurlOptions(CURL* curl);
    void setCurlHeaders(CURL* curl, bool isPost);

    // Error handling
    bool handleCurlError(CURLcode res, const std::string& operation);
    bool handleHttpError(long httpCode, const std::string& operation);

    // CURL callbacks
    static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp);

    // CURL global initialization
    static bool curlInitialized;
};

} // namespace daemon
} // namespace pastella
