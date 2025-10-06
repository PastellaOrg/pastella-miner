# Pastella Miner

A high-performance C++ implementation of the Velora proof-of-work algorithm for the Pastella cryptocurrency.

## Features

- **High Performance**: NVIDIA GPU-accelerated mining with CUDA 12.1
- **Multi-GPU Support**: Mine with multiple NVIDIA GPUs simultaneously
- **Pool Mining**: Connect to mining pools (GPU only)
- **Daemon Mining**: Direct blockchain mining (GPU and CPU)
- **ASIC Resistant**: Memory-hard algorithm with unpredictable access patterns
- **Real-time Monitoring**: Live hashrate and performance statistics

## Requirements

### System Requirements
- **OS**: Windows 10+ or Linux (Ubuntu 22.04+)
- **RAM**: 8GB minimum (256MB scratchpad + overhead)
- **GPU**: NVIDIA GPU with CUDA Compute Capability 5.0+ (GTX 900 series or newer)

### Build Dependencies
- **CMake**: 3.20+
- **Compiler**:
  - Windows: Visual Studio 2022
  - Linux: GCC 11
- **CUDA Toolkit**: 12.1 (required)
- **OpenSSL**: 1.1.1+ (optional, Linux only)

## Quick Start

### Windows

```bash
# Clone the repository
git clone https://github.com/PastellaOrg/pastella-miner.git
cd pastella-miner

# Build using batch script
build.bat

# Or build manually
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

### Linux

```bash
# Clone the repository
git clone https://github.com/PastellaOrg/pastella-miner.git
cd pastella-miner

# Install dependencies
sudo apt-get update
sudo apt-get install -y build-essential cmake libssl-dev

# Install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-1

# Build the miner
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## Usage

### Configuration

Edit `config.json` to configure your mining setup:

```json
{
  "_preset": "balanced",

  "pool": {
    "url": "stratum+tcp://pool.pastella.org",
    "port": 3333,
    "wallet": "your_wallet_address",
    "worker": "pastella-miner",
    "daemon": false,
    "daemon_url": "http://localhost:22000",
    "daemon_api_key": ""
  },

  "cuda": {
    "devices": [
      {
        "id": 0,
        "threads": 512,
        "blocks": 512,
        "batch_size": 128000,
        "enabled": true,
        "override_launch": false
      }
    ]
  },

  "cpu": {
    "enabled": false,
    "threads": 4
  },

  "verbose": false
}
```

### Presets

Available GPU memory presets in `config.json`:

- **auto**: Automatically detect GPU and apply optimal settings (recommended)
- **maximum**: 85% VRAM usage, highest hashrate
- **balanced**: 75% VRAM usage, good for gaming PCs
- **lowpower**: 60% VRAM usage, 24/7 mining

### Running the Miner

```bash
# Pool mining (GPU only)
./pastella-miner

# Daemon mining with GPU
./pastella-miner --daemon

# Daemon mining with CPU (fallback)
./pastella-miner --daemon --cpu

# Benchmark mode
./pastella-miner --benchmark
```

### Command Line Options

```
Options:
  --help              Show help message
  --benchmark         Run performance benchmark
  --daemon            Enable daemon mining mode
  --cpu               Enable CPU mining (daemon mode only)
  -g, --gpu           Enable GPU mining (default)
  -t, --threads <n>   Number of CPU threads (default: auto)
  -d, --difficulty <n> Mining difficulty
```

## Mining Modes

### Pool Mining
- **GPU Support**: Full support with optimized batching
- **CPU Support**: Not currently supported (in development)
- **Features**: Stratum protocol, automatic failover, share validation

### Daemon Mining
- **GPU Support**: Full support with multi-GPU capability
- **CPU Support**: Full support with multi-threading
- **Features**: Direct blockchain connection, block template updates

## Architecture

```
pastella-miner/
├── include/                    # Header files
│   ├── velora/                # Velora algorithm (CPU/GPU)
│   ├── utils/                 # Utilities (crypto, logging)
│   ├── daemon/                # Daemon client
│   └── mining_types.h         # Core mining data structures
├── src/                       # Source files
│   ├── velora/                # Algorithm implementation
│   │   ├── velora_algorithm.cpp
│   │   ├── velora_miner.cpp
│   │   ├── velora_kernel.cu   # CUDA GPU kernels
│   │   └── sha256n.cu         # GPU SHA-256
│   ├── pool_miner.cpp         # Pool mining client
│   ├── daemon_miner.cpp       # Daemon mining client
│   ├── config_manager.cpp     # Configuration management
│   └── main.cpp               # Entry point
├── externals/                 # Third-party libraries
│   ├── curl/                  # HTTP/Stratum networking
│   └── rapidjson/             # JSON parsing
├── CMakeLists.txt             # Build configuration
└── config.json                # Runtime configuration
```

## Velora Algorithm

The Velora algorithm is an ASIC-resistant, GPU-friendly proof-of-work algorithm designed for the Pastella cryptocurrency.

### Key Parameters

- **Scratchpad Size**: 256MB (2,016 blocks of 131,072 bytes)
- **Memory Accesses**: 65,536 reads per hash
- **Epoch Length**: 10,000 blocks (scratchpad regeneration)
- **Finalization**: SHA-256 based

### Algorithm Flow

1. **Epoch Calculation**: Determine current epoch from block number
2. **Scratchpad Generation**: Initialize 256MB memory buffer with PRNG
3. **Pattern Generation**: Generate 65,536 memory access indices from block header and nonce
4. **Memory Walk**: Execute pattern with XOR mixing operations
5. **Finalization**: SHA-256(block_header || nonce || timestamp || accumulator)

### GPU Optimization

- **Parallel Processing**: Process thousands of nonces simultaneously
- **Memory Coalescing**: Optimized memory access patterns for GPU architecture
- **Batch Processing**: Efficient workload distribution across CUDA cores
- **Double Buffering**: Overlap computation and data transfer

## Troubleshooting

### CUDA Not Found
- Ensure CUDA Toolkit 12.1 is installed
- Verify `nvcc` is in your PATH
- Check GPU driver version (minimum 525.60.11 for Linux, 527.41 for Windows)

### Build Failures
- Windows: Ensure Visual Studio 2022 with C++ tools is installed
- Linux: Install GCC 11 (CUDA 12.1 requirement)
- Verify CMake version is 3.20 or higher

### Low Hashrate
- Check GPU utilization with `nvidia-smi`
- Try different presets in config.json
- Ensure GPU has sufficient power and cooling
- Update NVIDIA drivers to latest version

### Pool Connection Issues
- Verify pool URL and port are correct
- Check firewall settings
- Ensure wallet address is valid

## License

This project is licensed under the MIT License - see the LICENSE file for details.
