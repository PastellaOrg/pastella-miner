cmake_minimum_required(VERSION 3.20)

# Use traditional CUDA support instead of modern CUDA language
project(pastella-miner LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA using traditional method (optional for testing)
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "✅ CUDA found: ${CUDA_VERSION}")
    message(STATUS "   CUDA Compiler: ${CUDA_NVCC_EXECUTABLE}")

    # Define HAVE_CUDA to enable CUDA code paths
    add_definitions(-DHAVE_CUDA)

    # 🎯 MULTI-ARCHITECTURE: Support wide range of GPUs from GTX 1060 to RTX 40 series
    set(CMAKE_CUDA_ARCHITECTURES 61 70 75 86 89)
    set(CUDA_NVCC_FLAGS_DEBUG "-g -G")
    set(CUDA_NVCC_FLAGS_RELEASE "-O3")

    # Note: CUDA include directories will be added AFTER project includes
    # to prevent CUDA headers from hijacking standard library includes
    link_directories(${CUDA_LIBRARY_DIRS})

    # Platform-specific CUDA configuration
    if(WIN32)
        # 🔧 VISUAL STUDIO 2022 COMPATIBILITY: Allow CUDA 12.1 to work with VS 2022
        if(MSVC_VERSION GREATER_EQUAL 1930)
            list(APPEND CUDA_NVCC_FLAGS "-allow-unsupported-compiler")
            list(APPEND CUDA_NVCC_FLAGS "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
            message(STATUS "⚙️  VS 2022 detected - adding compatibility flags for CUDA 12.1")
        endif()

        # Set CUDA library directories explicitly for Windows
        set(CUDA_LIBRARY_DIRS "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64")
        link_directories(${CUDA_LIBRARY_DIRS})
    endif()
else()
    message(STATUS "⚠️  CUDA not available - building CPU-only version")
    add_definitions(-DNO_CUDA)
endif()

# Set compiler flags
if(MSVC)
    add_compile_options(/W3 /O2)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)

    # Suppress RapidJSON warnings
    add_definitions(-D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING)
    add_definitions(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
    add_definitions(-DNOMINMAX)  # Prevent Windows min/max macro conflicts

    # Additional warning suppressions for RapidJSON
    add_compile_options(/wd4996)  # Suppress C4996 (deprecation warnings)
    add_compile_options(/wd4003)  # Suppress C4003 (macro invocation warnings)

    # Global RapidJSON warning suppressions
    add_compile_options(/wd4244)  # Suppress C4244 (conversion warnings)
    add_compile_options(/wd4267)  # Suppress C4267 (size_t conversion warnings)

    # Force static runtime to match CUDA compilation
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    add_compile_options(/MT$<$<CONFIG:Debug>:d>)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

# Add C++-specific compile options (not for C files)
if(NOT MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-class-memaccess>)
endif()

# Find required packages
# Use local CURL from externals directory
set(CURL_ROOT_DIR "${CMAKE_SOURCE_DIR}/externals/curl")
set(CURL_INCLUDE_DIR "${CURL_ROOT_DIR}/include")

message(STATUS "✅ Using local CURL from externals")
message(STATUS "   CURL Root: ${CURL_ROOT_DIR}")
message(STATUS "   CURL Include: ${CURL_INCLUDE_DIR}")

# Platform-specific OpenSSL configuration
if(WIN32)
    # Windows: Use specific OpenSSL path
    set(OPENSSL_ROOT_DIR "C:/Program Files/OpenSSL-Win64")
    set(OPENSSL_INCLUDE_DIR "${OPENSSL_ROOT_DIR}/include")
    set(OPENSSL_LIBRARIES "${OPENSSL_ROOT_DIR}/lib/VC/x64/MT")

    # Manually find OpenSSL libraries on Windows
    find_library(OPENSSL_SSL_LIBRARY
        NAMES libssl_static libssl ssl
        PATHS "${OPENSSL_ROOT_DIR}/lib/VC/x64/MT"
        NO_DEFAULT_PATH
    )

    find_library(OPENSSL_CRYPTO_LIBRARY
        NAMES libcrypto_static libcrypto crypto
        PATHS "${OPENSSL_ROOT_DIR}/lib/VC/x64/MT"
        NO_DEFAULT_PATH
    )

    if(OPENSSL_SSL_LIBRARY AND OPENSSL_CRYPTO_LIBRARY)
        message(STATUS "✅ OpenSSL found manually")
        message(STATUS "   OpenSSL Root: ${OPENSSL_ROOT_DIR}")
        message(STATUS "   OpenSSL SSL: ${OPENSSL_SSL_LIBRARY}")
        message(STATUS "   OpenSSL Crypto: ${OPENSSL_CRYPTO_LIBRARY}")
        set(OpenSSL_FOUND TRUE)
        add_definitions(-DHAVE_OPENSSL)
    else()
        message(STATUS "⚠️  OpenSSL not found - using fallback crypto")
        add_definitions(-DNO_OPENSSL)
    endif()
else()
    # Linux: Use standard OpenSSL finding
    find_package(OpenSSL)
    if(OPENSSL_FOUND)
        message(STATUS "✅ OpenSSL found")
        message(STATUS "   OpenSSL Version: ${OPENSSL_VERSION}")
        message(STATUS "   OpenSSL Include: ${OPENSSL_INCLUDE_DIR}")
        add_definitions(-DHAVE_OPENSSL)
    else()
        message(STATUS "⚠️  OpenSSL not found - using fallback crypto")
        add_definitions(-DNO_OPENSSL)
    endif()
endif()

find_package(Threads REQUIRED)

# For JSON support, we'll use a simple header-only approach
# Download jsoncpp header if not available
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/include/json/json.h")
    message(STATUS "JSON library not found, using built-in JSON parser")
    add_definitions(-DUSE_SIMPLE_JSON)
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/externals/rapidjson/include)
include_directories(${CMAKE_SOURCE_DIR}/externals/rapidjson/include/rapidjson)
include_directories(${CURL_INCLUDE_DIR})

# Configure CURL with platform-specific settings
set(BUILD_SHARED_LIBS OFF)
set(BUILD_STATIC_LIBS ON)
set(BUILD_CURL_EXE OFF)
set(BUILD_TESTING OFF)
set(BUILD_LIBCURL_DOCS OFF)
set(BUILD_MISC_DOCS OFF)
set(CURL_STATIC_CRT ON)
set(HTTP_ONLY ON)
set(CURL_USE_LIBPSL OFF)
set(USE_LIBIDN2 OFF)
set(CURL_ZLIB OFF)
set(CURL_BROTLI OFF)
set(CURL_ZSTD OFF)
set(USE_NGHTTP2 OFF)
set(ENABLE_MANUAL OFF)
set(ENABLE_IPV6 ON)
set(ENABLE_UNIX_SOCKETS OFF)
set(CURL_WERROR OFF)

# Platform-specific SSL configuration for CURL
if(WIN32)
    set(CURL_USE_OPENSSL OFF)
    set(CURL_USE_SCHANNEL ON)
    message(STATUS "✅ CURL configured to use Windows native SSL (Schannel)")
else()
    set(CURL_USE_OPENSSL ON)
    set(CURL_USE_SCHANNEL OFF)
    message(STATUS "✅ CURL configured to use OpenSSL")
endif()

# Add CURL as a subdirectory
add_subdirectory(externals/curl)

# Sources
set(SOURCES
    src/main.cpp
    src/config_manager.cpp
    src/utils/mining_utils.cpp
    src/daemon_miner.cpp
    src/pool_miner.cpp
    src/velora/velora_algorithm.cpp
    src/velora/velora_miner.cpp
    src/utils/crypto_utils.cpp
    src/utils/logger.cpp
    src/transactions/transaction_manager.cpp
    src/daemon/daemon_client.cpp
)

# Add custom SHA256 implementation only if OpenSSL is not available
if(NOT OpenSSL_FOUND AND NOT OPENSSL_FOUND)
    set(SOURCES ${SOURCES} src/sha256.c)
    message(STATUS "   Using custom SHA256 implementation (no OpenSSL)")
else()
    message(STATUS "   Using OpenSSL SHA256 implementation")
endif()

# Add CUDA kernel if CUDA is available
if(CUDA_FOUND)
    set(SOURCES ${SOURCES} src/velora/velora_kernel.cu src/sha256n.cu)
    message(STATUS "   CUDA kernel: src/velora/velora_kernel.cu")
endif()

# Headers
set(HEADERS
    include/velora/velora_algorithm.h
    include/velora/velora_miner.h
    include/utils/crypto_utils.h
    include/utils/logger.h
    include/types.h
)

# Create executable
# Set CUDA properties for the target
if(CUDA_FOUND)
    # Use traditional CUDA compilation approach for CMake 3.20 compatibility
    message(STATUS "   CUDA compilation enabled")

    # Use cuda_add_executable for proper CUDA compilation
    cuda_add_executable(pastella-miner ${SOURCES} ${HEADERS})

    # Set CUDA properties
    set_target_properties(pastella-miner PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )

    # Add CUDA include directories as SYSTEM includes (lower priority than standard library)
    # This prevents CUDA from hijacking standard library includes like <atomic>
    if(WIN32)
        target_include_directories(pastella-miner SYSTEM PRIVATE
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include"
        )
    else()
        target_include_directories(pastella-miner SYSTEM PRIVATE
            ${CUDA_INCLUDE_DIRS}
        )
    endif()

    # Link CUDA runtime libraries
    target_link_libraries(pastella-miner ${CUDA_LIBRARIES})
    
    # Platform-specific CUDA library linking
    if(WIN32)
        target_link_libraries(pastella-miner "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/cudart_static.lib")
        target_link_libraries(pastella-miner "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/lib/x64/cuda.lib")
    else()
        # Linux: Use standard CUDA libraries
        find_library(CUDART_LIBRARY cudart_static PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
        if(CUDART_LIBRARY)
            target_link_libraries(pastella-miner ${CUDART_LIBRARY})
            message(STATUS "   CUDA runtime library linked: ${CUDART_LIBRARY}")
        else()
            target_link_libraries(pastella-miner cudart_static)
            message(STATUS "   CUDA runtime library linked (default)")
        endif()
    endif()
else()
    # Create regular executable without CUDA
    add_executable(pastella-miner ${SOURCES} ${HEADERS})
endif()

# Link libraries
if(OpenSSL_FOUND OR OPENSSL_FOUND)
    if(WIN32)
        # Windows: Link manually found OpenSSL
        target_link_libraries(pastella-miner ${OPENSSL_SSL_LIBRARY})
        target_link_libraries(pastella-miner ${OPENSSL_CRYPTO_LIBRARY})
        message(STATUS "   OpenSSL static libraries linked")
    else()
        # Linux: Link standard OpenSSL
        target_link_libraries(pastella-miner OpenSSL::SSL OpenSSL::Crypto)
        message(STATUS "   OpenSSL libraries linked")
    endif()
else()
    message(STATUS "   Using fallback crypto (no OpenSSL)")
endif()

target_link_libraries(pastella-miner Threads::Threads)

# Link CURL
target_link_libraries(pastella-miner libcurl)
message(STATUS "   CURL library linked")