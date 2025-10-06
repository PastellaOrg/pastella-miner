# 🚀 Pastella Miner - Configuration Presets

The Pastella miner includes several preset modes in a single `config.json` file to make mining setup
simple and user-friendly.

## 📋 Available Presets

### 1. **Auto-Detection** (`"_preset": "auto"`)

- **Best for**: First-time users, unknown hardware
- **Memory Usage**: 65-85% (adaptive based on GPU)
- **Power**: Adaptive
- **Description**: Automatically detects your GPU and applies optimal settings

### 2. **Maximum Performance** (`"_preset": "maximum"`)

- **Best for**: Dedicated mining rigs, high-end GPUs (RTX 30xx/40xx)
- **Memory Usage**: 85%
- **Power**: High
- **Description**: Highest hashrate possible - uses most VRAM and system resources

### 3. **Balanced Performance** (`"_preset": "balanced"`)

- **Best for**: Gaming PCs, workstations, general use while mining
- **Memory Usage**: 75%
- **Power**: Medium
- **Description**: Good hashrate with moderate resource usage - leaves room for other apps

### 4. **Low Power Mining** (`"_preset": "lowpower"`)

- **Best for**: 24/7 mining, laptops, older GPUs, power-limited systems
- **Memory Usage**: 60%
- **Power**: Low
- **Description**: Lower power consumption for continuous operation

## 🎯 How to Use Presets

### Simple Method: Edit config.json

1. Open `config.json` in any text editor
2. Change the `"_preset"` value at the top:
   ```json
   {
     "_preset": "maximum", // Change this to: auto, maximum, balanced, or lowpower
     "_available_presets": {
       "auto": "Automatically detect GPU and apply optimal settings (recommended)",
       "maximum": "Maximum performance - uses 85% VRAM, highest hashrate",
       "balanced": "Balanced performance - uses 75% VRAM, good for gaming PCs",
       "lowpower": "Low power mode - uses 60% VRAM, good for 24/7 mining"
     }
   }
   ```
3. Save the file and run `./pastella-miner`

### Examples:

```bash
# For maximum performance (edit config.json: "_preset": "maximum")
./pastella-miner

# For balanced performance (edit config.json: "_preset": "balanced")
./pastella-miner

# For low power mining (edit config.json: "_preset": "lowpower")
./pastella-miner

# For auto-detection (edit config.json: "_preset": "auto") - DEFAULT
./pastella-miner
```

## ⚙️ Preset Details

| Preset    | VRAM Usage | Batch Size (High-end) | Batch Size (Mid-range) | Batch Size (Older) |
| --------- | ---------- | --------------------- | ---------------------- | ------------------ |
| Maximum   | 85%        | 1M nonces             | 512K nonces            | 256K nonces        |
| Balanced  | 75%        | 850K nonces           | 400K nonces            | 200K nonces        |
| Low Power | 60%        | 600K nonces           | 300K nonces            | 150K nonces        |
| Auto      | Adaptive   | Varies by GPU         | Varies by GPU          | Varies by GPU      |

## 🔧 Manual Configuration

If you prefer manual configuration, set specific values in config.json:

```json
{
  "cuda": {
    "devices": [
      {
        "id": 1,
        "threads": 256,
        "blocks": 512,
        "batch_size": 256000,
        "enabled": true
      }
    ]
  }
}
```

## 🚀 Quick Start Guide

1. **New User**: Use `config-auto.json` - it will detect everything automatically
2. **Gaming PC**: Use `config-balanced.json` - good performance, leaves room for games
3. **Mining Rig**: Use `config-maximum.json` - squeeze every bit of performance
4. **Laptop/24-7**: Use `config-lowpower.json` - lower heat and power consumption

The miner will automatically detect your GPU capabilities and apply the appropriate settings within
each preset's parameters.
