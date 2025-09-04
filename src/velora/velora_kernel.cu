#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda/std/cstdint>
#include <cstdio>

// 🚀 GPU SHA-256 IMPLEMENTATION for exact CPU pattern matching
// SHA-256 constants
__constant__ uint32_t gpu_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA-256 helper functions
__device__ uint32_t gpu_rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

__device__ uint32_t gpu_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

__device__ uint32_t gpu_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ uint32_t gpu_sigma0(uint32_t x) {
    return gpu_rotr(x, 2) ^ gpu_rotr(x, 13) ^ gpu_rotr(x, 22);
}

__device__ uint32_t gpu_sigma1(uint32_t x) {
    return gpu_rotr(x, 6) ^ gpu_rotr(x, 11) ^ gpu_rotr(x, 25);
}

__device__ uint32_t gpu_gamma0(uint32_t x) {
    return gpu_rotr(x, 7) ^ gpu_rotr(x, 18) ^ (x >> 3);
}

__device__ uint32_t gpu_gamma1(uint32_t x) {
    return gpu_rotr(x, 17) ^ gpu_rotr(x, 19) ^ (x >> 10);
}

// GPU SHA-256 implementation
__device__ void gpu_sha256(const unsigned char* data, int len, unsigned char* hash) {
    // Initial hash values
    uint32_t h0 = 0x6a09e667;
    uint32_t h1 = 0xbb67ae85;
    uint32_t h2 = 0x3c6ef372;
    uint32_t h3 = 0xa54ff53a;
    uint32_t h4 = 0x510e527f;
    uint32_t h5 = 0x9b05688c;
    uint32_t h6 = 0x1f83d9ab;
    uint32_t h7 = 0x5be0cd19;

    // Pre-processing: add padding
    unsigned char padded[128]; // Max 2 blocks for our 92-byte input
    int padded_len = 0;

    // Copy original data
    for (int i = 0; i < len; i++) {
        padded[padded_len++] = data[i];
    }

    // Add the '1' bit (plus zero padding to make it a byte)
    padded[padded_len++] = 0x80;

    // Add zero padding until message length is 64 bits less than a multiple of 512 bits
    while ((padded_len % 64) != 56) {
        padded[padded_len++] = 0x00;
    }

    // Add length in bits as 64-bit big-endian
    uint64_t bit_len = len * 8;
    for (int i = 7; i >= 0; i--) {
        padded[padded_len++] = (bit_len >> (i * 8)) & 0xFF;
    }

    // Process message in 512-bit chunks
    for (int chunk_start = 0; chunk_start < padded_len; chunk_start += 64) {
        uint32_t w[64];

        // Break chunk into sixteen 32-bit big-endian words
        for (int i = 0; i < 16; i++) {
            w[i] = (padded[chunk_start + i*4] << 24) |
                   (padded[chunk_start + i*4 + 1] << 16) |
                   (padded[chunk_start + i*4 + 2] << 8) |
                   (padded[chunk_start + i*4 + 3]);
        }

        // Extend the first 16 words into the remaining 48 words
        for (int i = 16; i < 64; i++) {
            w[i] = gpu_gamma1(w[i-2]) + w[i-7] + gpu_gamma0(w[i-15]) + w[i-16];
        }

        // Initialize working variables
        uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;

        // Main loop
        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h + gpu_sigma1(e) + gpu_ch(e, f, g) + gpu_sha256_k[i] + w[i];
            uint32_t t2 = gpu_sigma0(a) + gpu_maj(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        // Add this chunk's hash to result
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
        h5 += f;
        h6 += g;
        h7 += h;
    }

    // Produce the final hash value as big-endian
    hash[0] = (h0 >> 24) & 0xFF; hash[1] = (h0 >> 16) & 0xFF; hash[2] = (h0 >> 8) & 0xFF; hash[3] = h0 & 0xFF;
    hash[4] = (h1 >> 24) & 0xFF; hash[5] = (h1 >> 16) & 0xFF; hash[6] = (h1 >> 8) & 0xFF; hash[7] = h1 & 0xFF;
    hash[8] = (h2 >> 24) & 0xFF; hash[9] = (h2 >> 16) & 0xFF; hash[10] = (h2 >> 8) & 0xFF; hash[11] = h2 & 0xFF;
    hash[12] = (h3 >> 24) & 0xFF; hash[13] = (h3 >> 16) & 0xFF; hash[14] = (h3 >> 8) & 0xFF; hash[15] = h3 & 0xFF;
    hash[16] = (h4 >> 24) & 0xFF; hash[17] = (h4 >> 16) & 0xFF; hash[18] = (h4 >> 8) & 0xFF; hash[19] = h4 & 0xFF;
    hash[20] = (h5 >> 24) & 0xFF; hash[21] = (h5 >> 16) & 0xFF; hash[22] = (h5 >> 8) & 0xFF; hash[23] = h5 & 0xFF;
    hash[24] = (h6 >> 24) & 0xFF; hash[25] = (h6 >> 16) & 0xFF; hash[26] = (h6 >> 8) & 0xFF; hash[27] = h6 & 0xFF;
    hash[28] = (h7 >> 24) & 0xFF; hash[29] = (h7 >> 16) & 0xFF; hash[30] = (h7 >> 8) & 0xFF; hash[31] = h7 & 0xFF;
}

// Include proven GPU SHA-256 library types
extern "C" {
#include "../config.h"
}

// 🎯 SIMPLE VELORA KERNEL: Focus only on accumulator calculation
__device__ uint32_t ensure32BitUnsigned(uint64_t value) {
    return static_cast<uint32_t>(value & 0xFFFFFFFFULL);
}

// Simple kernel removed - only using ultra-optimized kernel for cleaner codebase


// 🚀 FULL GPU ACCELERATION: Complete GPU SHA-256 Implementation
__device__ void gpu_sha256(const unsigned char* data, size_t len, unsigned char* hash) {
    // SHA-256 constants
    const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    // Initial hash values
    uint32_t h[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    // Pre-processing: padding for exactly 92 bytes input
    unsigned char padded[128] = {0};

    // Copy input data (92 bytes)
    for (size_t i = 0; i < len && i < 92; i++) {
        padded[i] = data[i];
    }

    // Add padding bit
    padded[len] = 0x80;

    // Add length in bits (big-endian 64-bit) at positions 120-127
    uint64_t bit_len = len * 8;
    padded[120] = (bit_len >> 56) & 0xFF;
    padded[121] = (bit_len >> 48) & 0xFF;
    padded[122] = (bit_len >> 40) & 0xFF;
    padded[123] = (bit_len >> 32) & 0xFF;
    padded[124] = (bit_len >> 24) & 0xFF;
    padded[125] = (bit_len >> 16) & 0xFF;
    padded[126] = (bit_len >> 8) & 0xFF;
    padded[127] = (bit_len >> 0) & 0xFF;

    // Process 512-bit chunk
    uint32_t w[64];

    // Copy chunk into first 16 words w[0..15] (big-endian)
    for (int i = 0; i < 16; i++) {
        w[i] = (padded[i*4] << 24) | (padded[i*4+1] << 16) | (padded[i*4+2] << 8) | padded[i*4+3];
    }

    // Extend the first 16 words into the remaining 48 words w[16..63]
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = __funnelshift_r(w[i-15], w[i-15], 7) ^ __funnelshift_r(w[i-15], w[i-15], 18) ^ (w[i-15] >> 3);
        uint32_t s1 = __funnelshift_r(w[i-2], w[i-2], 17) ^ __funnelshift_r(w[i-2], w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    // Initialize working variables
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], h_var = h[7];

    // Main loop
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = __funnelshift_r(e, e, 6) ^ __funnelshift_r(e, e, 11) ^ __funnelshift_r(e, e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h_var + S1 + ch + k[i] + w[i];
        uint32_t S0 = __funnelshift_r(a, a, 2) ^ __funnelshift_r(a, a, 13) ^ __funnelshift_r(a, a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h_var = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    // Add the compressed chunk to the current hash value
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += h_var;

    // Produce the final hash value (big-endian)
    for (int i = 0; i < 8; i++) {
        hash[i*4] = (h[i] >> 24) & 0xFF;
        hash[i*4+1] = (h[i] >> 16) & 0xFF;
        hash[i*4+2] = (h[i] >> 8) & 0xFF;
        hash[i*4+3] = h[i] & 0xFF;
    }
}

// 🚀 ULTRA-OPTIMIZED VELORA KERNEL: GPU pattern generation + accumulator calculation
// This eliminates 64,000 CPU pattern generations per batch!
__global__ void velora_ultra_optimized_kernel(
    const uint32_t* scratchpad,
    uint64_t start_nonce,
    uint32_t nonce_step,
    uint32_t* acc_out,
    const uint64_t blockNumber,
    const uint32_t difficulty,
    uint32_t total_nonces,
    uint32_t pattern_size,
    uint32_t scratchpad_size,
    uint64_t timestamp,
    const unsigned char* previousHash,
    const unsigned char* merkleRoot
) {
    int nonce_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (nonce_idx >= total_nonces) return;

    // 🚀 GENERATE NONCE ON GPU
    uint64_t nonce = start_nonce + (nonce_idx * nonce_step);

    // 🎯 DEBUG: Print first few nonces for verification
    if (nonce_idx < 3) {
        // printf("GPU nonce %d: Ultra-optimized with GPU pattern generation\\n", nonce_idx); // Disabled for clean output
    }

        // Initialize accumulator
    uint32_t accumulator = 0;

        // 🚀 FULL GPU ACCELERATION: Exact CPU Algorithm Implementation
    // ELIMINATES ALL CPU COMPUTATION - 100% GPU-based pattern generation!

    // 🎯 STEP 1: Build seed data exactly like CPU (EXACT MATCH)
    unsigned char seedData[8 + 8 + 8 + 32 + 32 + 4]; // Total: 92 bytes
    int offset = 0;

    // blockNumber (8 bytes, little endian) - EXACT CPU MATCH
    seedData[offset++] = (blockNumber >> 0) & 0xFF;
    seedData[offset++] = (blockNumber >> 8) & 0xFF;
    seedData[offset++] = (blockNumber >> 16) & 0xFF;
    seedData[offset++] = (blockNumber >> 24) & 0xFF;
    seedData[offset++] = (blockNumber >> 32) & 0xFF;
    seedData[offset++] = (blockNumber >> 40) & 0xFF;
    seedData[offset++] = (blockNumber >> 48) & 0xFF;
    seedData[offset++] = (blockNumber >> 56) & 0xFF;

    // nonce (8 bytes, little endian) - EXACT CPU MATCH
    seedData[offset++] = (nonce >> 0) & 0xFF;
    seedData[offset++] = (nonce >> 8) & 0xFF;
    seedData[offset++] = (nonce >> 16) & 0xFF;
    seedData[offset++] = (nonce >> 24) & 0xFF;
    seedData[offset++] = (nonce >> 32) & 0xFF;
    seedData[offset++] = (nonce >> 40) & 0xFF;
    seedData[offset++] = (nonce >> 48) & 0xFF;
    seedData[offset++] = (nonce >> 56) & 0xFF;

    // timestamp (8 bytes, little endian) - EXACT CPU MATCH
    seedData[offset++] = (timestamp >> 0) & 0xFF;
    seedData[offset++] = (timestamp >> 8) & 0xFF;
    seedData[offset++] = (timestamp >> 16) & 0xFF;
    seedData[offset++] = (timestamp >> 24) & 0xFF;
    seedData[offset++] = (timestamp >> 32) & 0xFF;
    seedData[offset++] = (timestamp >> 40) & 0xFF;
    seedData[offset++] = (timestamp >> 48) & 0xFF;
    seedData[offset++] = (timestamp >> 56) & 0xFF;

    // previousHash (32 bytes) - EXACT CPU MATCH
    for (int i = 0; i < 32; i++) {
        seedData[offset++] = previousHash[i];
    }

    // merkleRoot (32 bytes) - EXACT CPU MATCH
    for (int i = 0; i < 32; i++) {
        seedData[offset++] = merkleRoot[i];
    }

    // difficulty (4 bytes, little endian) - EXACT CPU MATCH
    seedData[offset++] = (difficulty >> 0) & 0xFF;
    seedData[offset++] = (difficulty >> 8) & 0xFF;
    seedData[offset++] = (difficulty >> 16) & 0xFF;
    seedData[offset++] = (difficulty >> 24) & 0xFF;

    // 🎯 STEP 2: GPU SHA-256 computation (EXACT CPU MATCH)
    unsigned char hash[32];
    gpu_sha256(seedData, 92, hash);

        // 🎯 STEP 3: Extract state using EXACT CPU seedFromHash logic
    // Match the EXACT CPU implementation that works correctly
    uint32_t state = 0;

    // Process hash in 4-byte chunks - EXACT CPU MATCH
    for (uint32_t i = 0; i < 32; i += 4) {
        // EXACT CPU logic: i % (buf.length - (buf.length % 4 || 4))
        uint32_t bufLength = 32;
        uint32_t remainder = bufLength % 4;  // remainder = 0 for 32 bytes
        uint32_t adjustedLength = (remainder == 0) ? 4 : remainder;  // adjustedLength = 4
        uint32_t readPos = i % (bufLength - adjustedLength);  // i % 28

        // Ensure we don't read past the end (EXACT CPU MATCH)
        if (readPos + 4 <= 32) {
            // Read 4 bytes in little-endian order (EXACT CPU MATCH)
            uint32_t word = (hash[readPos] | (hash[readPos+1] << 8) | (hash[readPos+2] << 16) | (hash[readPos+3] << 24));

            // XOR with current state (EXACT CPU MATCH)
            state = state ^ word;

            // xorshift32 step (EXACT CPU MATCH)
            state = state ^ ((state << 13) & 0xFFFFFFFF);
            state = state ^ (state >> 17);
            state = state ^ ((state << 5) & 0xFFFFFFFF);
        }
    }

    // Ensure non-zero (EXACT CPU MATCH)
    if (state == 0) state = 0x9e3779b9;

    // 🎯 DEBUG: Print first few seeds for verification
    if (nonce_idx < 3) {
        // printf("GPU nonce %d: FULL GPU SHA-256 seed 0x%08x (exact CPU match)\\n", nonce_idx, state); // Disabled for clean output
    }

    // 🚀 MEMORY WALK with GPU-generated patterns (NO TRANSFER NEEDED!)
    for (uint32_t i = 0; i < pattern_size; i++) {
        // Generate pattern using xorshift32 (deterministic sequence)
        state = state ^ ((state << 13) & 0xFFFFFFFF);
        state = state ^ (state >> 17);
        state = state ^ ((state << 5) & 0xFFFFFFFF);

        // 🚀 VECTORIZED MODULO: Fast modulo using bitwise operations when scratchpad size is power of 2
        uint32_t pattern_val = state;
        uint32_t readPos;
        if ((scratchpad_size & (scratchpad_size - 1)) == 0) {
            // Power of 2: use bitwise AND (much faster than modulo)
            readPos = pattern_val & (scratchpad_size - 1);
        } else {
            readPos = pattern_val % scratchpad_size;
        }

        // 🎯 CRITICAL FIX: Ensure scratchpad access is always within bounds
        if (readPos >= scratchpad_size) {
            readPos = readPos % scratchpad_size; // Fallback bounds check
        }

        uint32_t value = scratchpad[readPos];

        // 🎯 EXACT CPU ALGORITHM: 6-step accumulator calculation (PERFECT MATCH)

        // Step 1: XOR with scratchpad value (EXACT CPU MATCH)
        accumulator = (accumulator ^ value) & 0xFFFFFFFF;

        // Step 2: Add shifted value (EXACT CPU MATCH: i % 32, not i % 8!)
        uint64_t shiftedValue = (uint64_t)value << (i % 32);
        accumulator = ensure32BitUnsigned(accumulator + ensure32BitUnsigned(shiftedValue));

        // Step 3: XOR with right-shifted accumulator (EXACT CPU MATCH)
        accumulator = (accumulator ^ (accumulator >> 13)) & 0xFFFFFFFF;

        // Step 4: Multiply by CPU constant (EXACT CPU MATCH: 0x5bd1e995, not 31!)
        uint64_t multiplied = (uint64_t)accumulator * 0x5bd1e995;
        accumulator = ensure32BitUnsigned(multiplied);

        // Step 5: Mix in nonce and timestamp (EXACT JAVASCRIPT MATCH)
        // JavaScript: nonceBuffer.readUInt32LE(i % 4) reads 4 bytes starting at position (i % 4)
        // This ALLOWS overlapping reads - positions 0,1,2,3 can read beyond 8-byte buffer
        uint32_t nonceIndex = i % 4; // Positions 0, 1, 2, 3 (byte offsets)
        uint32_t timestampIndex = i % 4; // Positions 0, 1, 2, 3 (byte offsets)

        // Extract nonce word (little-endian, 4 bytes starting at nonceIndex)
        // JavaScript behavior: zero-pad when reading beyond 8-byte buffer
        uint32_t nonceWord = 0;
        for (int j = 0; j < 4; j++) {
            uint32_t bytePos = nonceIndex + j;
            if (bytePos < 8) {
                nonceWord |= ((nonce >> (bytePos * 8)) & 0xFF) << (j * 8);
            }
            // If bytePos >= 8, leave as zero (zero-padding)
        }

        // Extract timestamp word (little-endian, 4 bytes starting at timestampIndex)
        // JavaScript behavior: zero-pad when reading beyond 8-byte buffer
        uint32_t timestampWord = 0;
        for (int j = 0; j < 4; j++) {
            uint32_t bytePos = timestampIndex + j;
            if (bytePos < 8) {
                timestampWord |= ((timestamp >> (bytePos * 8)) & 0xFF) << (j * 8);
            }
            // If bytePos >= 8, leave as zero (zero-padding)
        }

        // Final XOR mix (EXACT CPU MATCH)
        accumulator = (accumulator ^ nonceWord ^ timestampWord) & 0xFFFFFFFF;

        // 🎯 DEBUG: First few iterations for verification (disabled for clean output)
        // if (nonce_idx < 3 && i < 3) {
        //     printf("GPU nonce %d iter %d: pattern=0x%08x, value=0x%08x, acc=0x%08x\\n",
        //            nonce_idx, i, pattern_val, value, accumulator);
        // }

        // Debug disabled for clean output
    }

    // 🎯 DEBUG: Final accumulator for first few nonces (disabled for clean output)
    // if (nonce_idx < 3) {
    //     printf("GPU nonce %d: FINAL ultra-optimized accumulator = 0x%08x\\n", nonce_idx, accumulator);
    // }

    // Store result
    if (nonce_idx < total_nonces) {
        acc_out[nonce_idx] = accumulator;
    }
}

// 🚀 ULTRA-OPTIMIZED KERNEL LAUNCHER
extern "C" bool launch_velora_ultra_optimized_kernel(
    const uint32_t* d_scratchpad,
    uint64_t start_nonce,
    uint32_t nonce_step,
    uint32_t* d_acc_out,
    const uint64_t blockNumber,
    const uint32_t difficulty,
    uint32_t total_nonces,
    uint32_t pattern_size,
    uint32_t scratchpad_size,
    uint64_t timestamp,
    const unsigned char* d_previousHash,
    const unsigned char* d_merkleRoot,
    cudaStream_t stream,
    int blocks_per_grid,
    int threads_per_block
) {
    // Launch the ultra-optimized kernel
    velora_ultra_optimized_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        d_scratchpad, start_nonce, nonce_step, d_acc_out,
        blockNumber, difficulty, total_nonces, pattern_size, scratchpad_size, timestamp,
        d_previousHash, d_merkleRoot
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA ultra-optimized kernel launch error: %s\n", cudaGetErrorString(err));
        return false;
    }

    return true;
}

