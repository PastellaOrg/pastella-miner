#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <mutex>

// Undefine Windows macros that conflict with our enum
#ifdef ERROR
#undef ERROR
#endif
#ifdef FATAL
#undef FATAL
#endif

namespace pastella {
namespace utils {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    FATAL = 4,
    STATUS = 5
};

class Logger {
private:
    static Logger* instance_;
    static std::mutex mutex_;

    std::ofstream logFile_;
    LogLevel currentLevel_;
    bool enableConsole_;
    bool enableFile_;
    std::string logFilePath_;

    // ANSI color codes for Windows (using Windows 10+ ANSI support)
    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* BLACK = "\033[30m";
    static constexpr const char* RED = "\033[31m";
    static constexpr const char* GREEN = "\033[32m";
    static constexpr const char* YELLOW = "\033[33m";
    static constexpr const char* BLUE = "\033[34m";
    static constexpr const char* MAGENTA = "\033[35m";
    static constexpr const char* CYAN = "\033[36m";
    static constexpr const char* WHITE = "\033[37m";
    static constexpr const char* BRIGHT_RED = "\033[91m";
    static constexpr const char* BRIGHT_GREEN = "\033[92m";
    static constexpr const char* BRIGHT_YELLOW = "\033[93m";
    static constexpr const char* BRIGHT_BLUE = "\033[94m";
    static constexpr const char* BRIGHT_MAGENTA = "\033[95m";
    static constexpr const char* BRIGHT_CYAN = "\033[96m";
    static constexpr const char* BRIGHT_WHITE = "\033[97m";
    static constexpr const char* GRAY = "\033[37m";        // Light gray
    static constexpr const char* DARK_GRAY = "\033[90m";   // Dark gray

    // Background color codes
    static constexpr const char* BG_GREEN = "\033[42m";
    static constexpr const char* BG_LIGHT_BLUE = "\033[104m";
    static constexpr const char* BG_DARK_BLUE = "\033[44m";
    static constexpr const char* BG_MAGENTA = "\033[45m";
    static constexpr const char* BG_YELLOW = "\033[43m";
    static constexpr const char* BG_RED = "\033[41m";
    static constexpr const char* BG_CYAN = "\033[46m";

    Logger() : currentLevel_(LogLevel::INFO), enableConsole_(true), enableFile_(true), logFilePath_("pastella-miner.log") {}

    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }

    std::string getColoredTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

        std::stringstream ss;
        // Main timestamp in gray
        ss << GRAY << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << RESET;
        // Milliseconds in dark gray
        ss << DARK_GRAY << '.' << std::setfill('0') << std::setw(3) << ms.count() << RESET;
        return ss.str();
    }

                std::string getLevelString(LogLevel level) {
                switch (level) {
                    case LogLevel::DEBUG: return "DEBUG";
                    case LogLevel::INFO: return "INFO ";
                    case LogLevel::WARNING: return "WARN ";
                    case LogLevel::ERROR: return "ERROR";
                    case LogLevel::FATAL: return "FATAL";
                    case LogLevel::STATUS: return "STATUS";
                    default: return "UNKNW";
                }
            }

                const char* getLevelColor(LogLevel level) {
                switch (level) {
                    case LogLevel::DEBUG: return CYAN;
                    case LogLevel::INFO: return GREEN;
                    case LogLevel::WARNING: return YELLOW;
                    case LogLevel::ERROR: return RED;
                    case LogLevel::FATAL: return BRIGHT_RED;
                    case LogLevel::STATUS: return BRIGHT_WHITE;
                    default: return WHITE;
                }
            }

    std::string getCategoryDisplay(const std::string& category) {
        // Fixed width based on longest category (7 chars: "CONFIG" + 1 space)
        const int fixedWidth = 8;
        const char* bgColor;
        
        if (category == "CUDA") {
            bgColor = BG_GREEN;
        } else if (category == "CPU") {
            bgColor = BG_LIGHT_BLUE;
        } else if (category == "POOL") {
            bgColor = BG_DARK_BLUE;
        } else if (category == "VELORA") {
            bgColor = BG_MAGENTA;
        } else if (category == "CONFIG") {
            bgColor = BG_YELLOW;
        } else if (category == "MINER") {
            bgColor = BG_CYAN;
        } else {
            bgColor = BG_RED; // Default for unknown categories
        }
        
        // Create padded category string
        std::string paddedCategory = " " + category;
        while (paddedCategory.length() < fixedWidth) {
            paddedCategory += " ";
        }
        
        return std::string(bgColor) + BRIGHT_WHITE + paddedCategory + RESET;
    }

    void writeToFile(const std::string& message) {
        if (enableFile_ && logFile_.is_open()) {
            logFile_ << message << std::endl;
            logFile_.flush();
        }
    }

public:
    static Logger* getInstance() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (instance_ == nullptr) {
            instance_ = new Logger();
        }
        return instance_;
    }

    void setLogLevel(LogLevel level) { currentLevel_ = level; }
    void setLogFile(const std::string& filePath) {
        logFilePath_ = filePath;
        if (logFile_.is_open()) logFile_.close();
        logFile_.open(logFilePath_, std::ios::app);
    }
    void enableConsole(bool enable) { enableConsole_ = enable; }
    void enableFile(bool enable) { enableFile_ = enable; }

    void log(LogLevel level, const std::string& message, const std::string& component = "") {
        if (level < currentLevel_) return;

        std::string timestamp = std::string(GRAY) + "[" + RESET + getColoredTimestamp() + std::string(GRAY) + "]" + RESET;
        std::string levelStr = getLevelString(level);

                // Special handling for STATUS level - no timestamp, no level, no component
        if (level == LogLevel::STATUS) {
            // Console output with colors (just the message)
            if (enableConsole_) {
                std::string coloredMessage = message;
                // Check if message starts with " *" and colorize the star
                if (message.length() >= 2 && message.substr(0, 2) == " *") {
                    coloredMessage = " " + std::string(GREEN) + "*" + std::string(RESET) + message.substr(2);
                }
                std::cout << getLevelColor(level) << coloredMessage << RESET << std::endl;
            }

            // File output (just the message)
            writeToFile(message);
            return;
        }

        // Normal handling for other levels
        std::string componentStr = component.empty() ? "" : "[" + component + "] ";

        // Console output with colors
        if (enableConsole_) {
            std::cout << timestamp << " "
                      << getLevelColor(level) << levelStr << RESET << " "
                      << componentStr << message << std::endl;
        }

        // File output without colors
        std::string plainTimestamp = "[" + getCurrentTimestamp() + "]";
        std::string fileMessage = plainTimestamp + " " + levelStr + " " + componentStr + message;
        writeToFile(fileMessage);
    }

    void logWithCategory(LogLevel level, const std::string& message, const std::string& category = "") {
        if (level < currentLevel_) return;

        std::string timestamp = std::string(GRAY) + "[" + RESET + getColoredTimestamp() + std::string(GRAY) + "]" + RESET;
        std::string levelStr = getLevelString(level);

        // Special handling for STATUS level - no timestamp, no level, no component
        if (level == LogLevel::STATUS) {
            // Console output with colors (just the message)
            if (enableConsole_) {
                std::string coloredMessage = message;
                // Check if message starts with " *" and colorize the star
                if (message.length() >= 2 && message.substr(0, 2) == " *") {
                    coloredMessage = " " + std::string(GREEN) + "*" + std::string(RESET) + message.substr(2);
                }
                std::cout << getLevelColor(level) << coloredMessage << RESET << std::endl;
            }

            // File output (just the message)
            writeToFile(message);
            return;
        }

        // Normal handling with colored background categories
        std::string categoryStr = category.empty() ? "" : getCategoryDisplay(category) + " ";

        // Console output with colors
        if (enableConsole_) {
            std::cout << timestamp << " "
                      << getLevelColor(level) << levelStr << RESET << " "
                      << categoryStr << message << std::endl;
        }

        // File output without colors (plain category in brackets)
        std::string plainTimestamp = "[" + getCurrentTimestamp() + "]";
        std::string plainCategory = category.empty() ? "" : "[" + category + "] ";
        std::string fileMessage = plainTimestamp + " " + levelStr + " " + plainCategory + message;
        writeToFile(fileMessage);
    }

    void debug(const std::string& message, const std::string& component = "") {
        log(LogLevel::DEBUG, message, component);
    }

    void info(const std::string& message, const std::string& component = "") {
        log(LogLevel::INFO, message, component);
    }

    void warning(const std::string& message, const std::string& component = "") {
        log(LogLevel::WARNING, message, component);
    }

    void error(const std::string& message, const std::string& component = "") {
        log(LogLevel::ERROR, message, component);
    }

                void fatal(const std::string& message, const std::string& component = "") {
                log(LogLevel::FATAL, message, component);
            }

            void status(const std::string& message, const std::string& component = "") {
                log(LogLevel::STATUS, message, component);
            }

    // Category-based logging methods
    void debugCategory(const std::string& message, const std::string& category = "") {
        logWithCategory(LogLevel::DEBUG, message, category);
    }

    void infoCategory(const std::string& message, const std::string& category = "") {
        logWithCategory(LogLevel::INFO, message, category);
    }

    void warningCategory(const std::string& message, const std::string& category = "") {
        logWithCategory(LogLevel::WARNING, message, category);
    }

    void errorCategory(const std::string& message, const std::string& category = "") {
        logWithCategory(LogLevel::ERROR, message, category);
    }

    void fatalCategory(const std::string& message, const std::string& category = "") {
        logWithCategory(LogLevel::FATAL, message, category);
    }

    void progress(const std::string& message) {
        if (enableConsole_) {
            std::cout << "\r" << message << std::flush;
        }
    }

    void clearProgress() {
        if (enableConsole_) {
            std::cout << std::endl;
        }
    }

    ~Logger() {
        if (logFile_.is_open()) {
            logFile_.close();
        }
    }
};

// Global logger instance
#define LOGGER pastella::utils::Logger::getInstance()

// Convenience macros
#define LOG_DEBUG(msg, comp) LOGGER->debug(msg, comp)
#define LOG_INFO(msg, comp) LOGGER->info(msg, comp)
#define LOG_WARNING(msg, comp) LOGGER->warning(msg, comp)
#define LOG_ERROR(msg, comp) LOGGER->error(msg, comp)
#define LOG_FATAL(msg, comp) LOGGER->fatal(msg, comp)
#define LOG_STATUS(msg, comp) LOGGER->status(msg, comp)
#define LOG_PROGRESS(msg) LOGGER->progress(msg)
#define LOG_CLEAR_PROGRESS() LOGGER->clearProgress()

// New category-based logging macros
#define LOG_DEBUG_CAT(msg, cat) LOGGER->debugCategory(msg, cat)
#define LOG_INFO_CAT(msg, cat) LOGGER->infoCategory(msg, cat)
#define LOG_WARNING_CAT(msg, cat) LOGGER->warningCategory(msg, cat)
#define LOG_ERROR_CAT(msg, cat) LOGGER->errorCategory(msg, cat)
#define LOG_FATAL_CAT(msg, cat) LOGGER->fatalCategory(msg, cat)

} // namespace utils
} // namespace pastella
