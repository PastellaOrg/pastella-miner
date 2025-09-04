#include "../../include/utils/logger.h"

namespace pastella {
namespace utils {

// Static member definitions
Logger* Logger::instance_ = nullptr;
std::mutex Logger::mutex_;

} // namespace utils
} // namespace pastella
