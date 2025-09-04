@echo off
echo ============================================
echo    PASTELLA MINER - CMAKE BUILD
echo    (CUDA 12.1 + VS 2019 Enterprise)
echo ============================================

REM Check if CMake is available
cmake --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ❌ CMake not found in PATH
    echo Please install CMake from: https://cmake.org/download/
    exit /b 1
)

REM Check if Visual Studio 2019 Enterprise is available
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    echo ✅ Found Visual Studio 2019 Enterprise
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat" (
    echo ✅ Found Visual Studio 2019 Enterprise
    call "C:\Program Files\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo ❌ Visual Studio 2019 Enterprise not found
    echo Please install Visual Studio 2019 Enterprise with C++ and CUDA development tools
    echo Download from: https://visualstudio.microsoft.com/vs/older-downloads/
    exit /b 1
)

echo.
echo 🔧 CMake version:
cmake --version

echo.
echo 🔧 Configuring project...
if exist build rmdir /s /q build
mkdir build
cd build

REM Configure with CMake
cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release
if %ERRORLEVEL% neq 0 (
    echo ❌ CMake configuration failed
    cd ..
    exit /b 1
)

echo.
echo 🔧 Building project...
cmake --build . --config Release
if %ERRORLEVEL% neq 0 (
    echo ❌ Build failed
    cd ..
    exit /b 1
)

echo.
echo ✅ Build completed successfully!
echo 📁 Executable: build\Release\pastella-miner.exe

cd ..

echo.
echo 🚀 To run the miner:
echo    cd build\Release
echo    pastella-miner.exe --help
echo.
echo 🎯 For GPU mining:
echo    pastella-miner.exe -g --benchmark
echo.



