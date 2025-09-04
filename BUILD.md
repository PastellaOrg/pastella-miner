# Building Pastalla Miner

## Quick Start

### Windows (Recommended)

1. **Double-click** `build.bat` or run `build.ps1` in PowerShell
2. Wait for the build to complete
3. Run the miner: `cd build\Release && pastalla-miner.exe --help`

### Manual Build

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release
```

## Requirements

- **CMake 3.16+** - [Download here](https://cmake.org/download/)
- **Visual Studio 2019/2022** with C++ tools OR **Build Tools**
- **CUDA Toolkit 12.1+** (for GPU mining)
- **OpenSSL** (optional, for cryptographic functions)

## What Gets Built

- **CPU-only version**: If CUDA is not available
- **CUDA version**: If CUDA 12.1+ is detected
- **OpenSSL support**: If OpenSSL is found

## Build Output

```
build/
└── Release/
    └── pastalla-miner.exe  ← Your miner executable
```

## Running the Miner

```bash
# Show help
pastalla-miner.exe --help

# CPU mining
pastalla-miner.exe --benchmark

# GPU mining (if CUDA available)
pastalla-miner.exe -g --benchmark

# Custom settings
pastalla-miner.exe -t 8 -d 100 --max-nonces 1000000
```

## Troubleshooting

### "CMake not found"
- Install CMake and add to PATH
- Or use the full path to cmake.exe

### "Visual Studio not found"
- Install Visual Studio 2019/2022 Community (free)
- Or install Build Tools only
- Make sure C++ development tools are selected

### "CUDA not found"
- Install CUDA Toolkit 12.1+
- Make sure nvcc is in PATH
- Restart after installation

### Build errors
- Delete the `build` folder and try again
- Check that all dependencies are installed
- Ensure you have enough disk space

## Performance

- **CPU**: ~100-500 H/s (depends on CPU)
- **GPU (RTX 3070)**: ~3,000+ H/s
- **GPU (GTX 1060)**: ~1,500+ H/s

## Support

If you encounter issues:
1. Check the error messages carefully
2. Ensure all requirements are met
3. Try a clean build (delete build folder)
4. Check that your GPU supports CUDA 12.1+

