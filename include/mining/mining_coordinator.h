#pragma once
#include "../mining_types.h
#include "mining_manager.h
#include <memory>

namespace pastella {
namespace mining {

/**
 * Mining Coordinator - Coordinates between different mining modes
 * Manages daemon mining and pool mining operations
 */
class MiningCoordinator {
public:
    MiningCoordinator();
    ~MiningCoordinator();

    // Mining operations
    bool startDaemonMining(const MinerConfig& config);
    bool startPoolMining(const MinerConfig& config;

    // Status
    bool isMining() const;
    bool isDa
