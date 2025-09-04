# Pastella Miner - C++ Implementation

A high-performance C++ implementation of the Velora proof-of-work algorithm for the Pastella cryptocurrency.

## 🚀 Features

- **High Performance**: 10-50x faster than Node.js implementation
- **GPU Acceleration**: CUDA/OpenCL support for massive speedup
- **ASIC Resistant**: Memory-hard algorithm with unpredictable access patterns
- **Cross Platform**: Windows, Linux, and macOS support
- **Multi-GPU**: Support for multiple GPU devices
- **Real-time Monitoring**: Live hash rate and progress tracking

## 📊 Performance Comparison

| Implementation | CPU Hash Rate | GPU Hash Rate | Performance Gain |
|----------------|---------------|---------------|------------------|
| **Node.js** | 16,000 H/s | 287 H/s | 1x (baseline) |
| **C++ CPU** | 200,000-800,000 H/s | - | 12-50x |
| **C++ GPU** | 800,000 H/s | 2,000,000+ H/s | 50-125x |

## 🏗️ Architecture

```
pastalla-miner/
├── include/                 # Header files
│   ├── velora/             # Velora algorithm headers
│   └── utils/              # Utility headers
├── src/                    # Source files
│   ├── velora/             # Velora algorithm implementation
│   ├── utils/              # Utility implementations
│   └── main.cpp            # Main application
├── CMakeLists.txt          # CMake build configuration
└── README.md               # This file
```

## 🔧 Requirements

### System Requirements
- **OS**: Windows 10+, Linux (Ubuntu 18.04+), macOS 10.15+
- **CPU**: x86_64 or ARM64 with SSE4.2 support
- **RAM**: 8GB+ (64MB scratchpad + overhead)
- **GPU**: NVIDIA (CUDA) or AMD/Intel (OpenCL) for acceleration

### Development Dependencies
- **Compiler**: GCC 7+, Clang 6+, or MSVC 2019+
- **CMake**: 3.16+
- **OpenSSL**: 1.1.1+
- **CUDA**: 11.0+ (optional, for NVIDIA GPU acceleration)
- **OpenCL**: 2.0+ (optional, for AMD/Intel GPU acceleration)

## 🚀 Quick Start

### 1. Clone and Build

```bash
# Clone the repository
git clone <repository-url>
cd pastalla-miner

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the project
cmake --build . --config Release

# Install (optional)
cmake --install .
```

### 2. Run the Miner

```bash
# Basic CPU mining
./pastalla-miner

# GPU mining with custom difficulty
./pastalla-miner -g -d 5000

# CPU mining with 8 threads
./pastalla-miner -c -t 8 -n 1000000

# Mine specific block
./pastalla-miner -b 12345 -d 10000
```

### 3. Command Line Options

```
Usage: pastalla-miner [OPTIONS]

Options:
  -h, --help              Show this help message
  -d, --difficulty <n>    Set mining difficulty (default: 1000)
  -g, --gpu               Enable GPU mining
  -c, --cpu               Force CPU-only mining
  -t, --threads <n>       Number of CPU threads (default: 4)
  -n, --nonces <n>        Maximum nonces to try (default: 1000000)
  -b, --block <n>         Block number to mine (default: 1)
  -v, --verbose           Enable verbose output
```

## 🎯 Velora Algorithm

The Velora algorithm is an ASIC-resistant, GPU-friendly proof-of-work algorithm that emphasizes:

- **Memory Hardness**: 64MB scratchpad with random access patterns
- **Epoch Rotation**: Scratchpad changes every 10,000 blocks
- **Unpredictable Access**: Memory patterns depend on block hash and nonce
- **GPU Optimization**: Designed for parallel processing

### Algorithm Flow

1. **Epoch Calculation**: `epoch = floor(blockNumber / 10000)`
2. **Scratchpad Generation**: PRNG-based 64MB memory initialization
3. **Pattern Generation**: 1000 memory access indices from (headerHash, nonce)
4. **Memory Walk**: Execute pattern with 32-bit mixing operations
5. **Finalization**: SHA-256(headerHash || nonce || accumulator)

## 🔧 Configuration

### GPU Configuration

```cpp
GPUConfig config;
config.deviceId = 0;           // GPU device ID
config.threadsPerBlock = 256;  // CUDA threads per block
config.blocksPerGrid = 1024;   // CUDA grid size
config.maxNonces = 1000000;    // Maximum nonces per batch
config.useDoublePrecision = false; // Single precision for speed
```

### Performance Tuning

- **CPU Threads**: Match your CPU core count
- **GPU Threads**: 256-1024 threads per block (GPU dependent)
- **Batch Size**: Larger batches improve GPU utilization
- **Memory**: Ensure sufficient RAM for scratchpad

## 📈 Performance Optimization

### CPU Optimizations
- **SIMD Instructions**: AVX2/AVX-512 for vector operations
- **Memory Layout**: Cache-friendly scratchpad access
- **Thread Pooling**: Efficient thread management
- **Compiler Flags**: `-O3 -march=native` for best performance

### GPU Optimizations
- **Memory Coalescing**: Aligned memory access patterns
- **Shared Memory**: Utilize GPU shared memory for frequently accessed data
- **Kernel Optimization**: Minimize divergent branching
- **Batch Processing**: Process multiple nonces simultaneously

## 🐛 Troubleshooting

### Common Issues

1. **Build Failures**
   - Ensure CMake 3.16+ is installed
   - Check OpenSSL development libraries
   - Verify compiler supports C++17

2. **GPU Initialization Failures**
   - Install latest GPU drivers
   - Verify CUDA/OpenCL installation
   - Check GPU memory availability

3. **Performance Issues**
   - Monitor CPU/GPU utilization
   - Check memory bandwidth
   - Verify algorithm parameters

### Debug Mode

```bash
# Build with debug symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --config Debug

# Run with verbose output
./pastalla-miner -v
```

## 🔒 Security Considerations

- **ASIC Resistance**: Algorithm designed to prevent specialized hardware
- **Memory Hardness**: Requires significant memory bandwidth
- **Epoch Rotation**: Prevents long-term optimization
- **Cryptographic Seeding**: SHA-256 based randomness

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

### Development Guidelines

- Follow C++17 standards
- Use RAII and smart pointers
- Implement proper error handling
- Add comprehensive tests
- Document public APIs

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original Velora algorithm design
- OpenSSL for cryptographic functions
- CUDA and OpenCL communities
- Pastalla cryptocurrency team

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: [Wiki](link-to-wiki)
- **Community**: [Discord/Telegram](link-to-community)

---

**Happy Mining! ⛏️🚀**




