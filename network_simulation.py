#!/usr/bin/env python3
"""
Network Simulation System
Simulates federated learning across distributed locations with latency and bandwidth
"""

import os
import json
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
import statistics

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class NetworkConditions:
    """Network condition parameters"""
    latency_ms: float  # One-way latency in milliseconds
    bandwidth_mbps: float  # Available bandwidth in Mbps
    packet_loss_rate: float  # Probability of packet loss (0-1)
    jitter_ms: float  # Latency variation
    location_name: str  # Location identifier


class NetworkSimulator:
    """Simulate federated learning communication over distributed networks"""

    def __init__(self, num_clients: int = 4):
        """Initialize network simulator"""
        self.num_clients = num_clients
        self.network_configs = self._create_network_configs()
        self.communication_logs = []

        logger.info(f"Initialized network simulator: {num_clients} clients")
        for config in self.network_configs:
            logger.info(
                f"  {config.location_name}: {config.latency_ms}ms, {config.bandwidth_mbps}Mbps")

    def _create_network_configs(self) -> List[NetworkConditions]:
        """Create realistic network conditions for distributed locations"""
        configs = [
            NetworkConditions(
                latency_ms=10.0,
                bandwidth_mbps=100.0,
                packet_loss_rate=0.001,
                jitter_ms=2.0,
                location_name="Local_Network"
            ),
            NetworkConditions(
                latency_ms=50.0,
                bandwidth_mbps=50.0,
                packet_loss_rate=0.005,
                jitter_ms=5.0,
                location_name="Regional_Data_Center"
            ),
            NetworkConditions(
                latency_ms=100.0,
                bandwidth_mbps=25.0,
                packet_loss_rate=0.01,
                jitter_ms=10.0,
                location_name="Geographic_Remote"
            ),
            NetworkConditions(
                latency_ms=150.0,
                bandwidth_mbps=10.0,
                packet_loss_rate=0.02,
                jitter_ms=15.0,
                location_name="IoT_Edge_Device"
            ),
        ]
        return configs[:self.num_clients]

    def calculate_model_upload_time(self, model_size_mb: float,
                                    client_id: int) -> Dict:
        """Calculate time to upload model update from client to server"""
        config = self.network_configs[client_id]

        # Calculate transmission time
        transmission_time_s = (model_size_mb * 8) / \
            config.bandwidth_mbps  # seconds

        # Add latency (client -> server)
        base_latency = config.latency_ms / 1000.0

        # Simulate jitter (random variation)
        jitter = np.random.normal(0, config.jitter_ms / 1000.0)
        total_latency = base_latency + jitter

        # Simulate packet loss impact (retransmission)
        if config.packet_loss_rate > 0:
            # Expected number of transmissions needed
            expected_retransmissions = 1.0 / (1.0 - config.packet_loss_rate)
            transmission_time_s *= expected_retransmissions

        total_time_s = transmission_time_s + total_latency

        return {
            'client_id': client_id,
            'location': config.location_name,
            'model_size_mb': model_size_mb,
            'bandwidth_mbps': config.bandwidth_mbps,
            'latency_ms': config.latency_ms,
            'packet_loss_rate': config.packet_loss_rate,
            'transmission_time_s': transmission_time_s,
            'total_latency_s': total_latency,
            'total_upload_time_s': total_time_s,
        }

    def calculate_model_download_time(self, model_size_mb: float,
                                      client_id: int) -> Dict:
        """Calculate time to download aggregated model from server to client"""
        config = self.network_configs[client_id]

        # Same calculation as upload
        transmission_time_s = (model_size_mb * 8) / config.bandwidth_mbps
        base_latency = config.latency_ms / 1000.0
        jitter = np.random.normal(0, config.jitter_ms / 1000.0)
        total_latency = base_latency + jitter

        if config.packet_loss_rate > 0:
            expected_retransmissions = 1.0 / (1.0 - config.packet_loss_rate)
            transmission_time_s *= expected_retransmissions

        total_time_s = transmission_time_s + total_latency

        return {
            'client_id': client_id,
            'location': config.location_name,
            'model_size_mb': model_size_mb,
            'transmission_time_s': transmission_time_s,
            'total_latency_s': total_latency,
            'total_download_time_s': total_time_s,
        }

    def simulate_training_round(self, model_size_mb: float = 2.08) -> Dict:
        """Simulate one complete federated learning round with network conditions"""

        logger.info(
            f"\nSimulating training round (Model size: {model_size_mb}MB)...")

        # Client uploads
        upload_times = []
        for client_id in range(self.num_clients):
            upload_info = self.calculate_model_upload_time(
                model_size_mb, client_id)
            upload_times.append(upload_info['total_upload_time_s'])
            logger.info(f"  Client {client_id} ({upload_info['location']}): "
                        f"Upload {upload_info['total_upload_time_s']:.3f}s")

        # Server aggregation (assume instant)
        aggregation_time_s = 0.1

        # Server broadcasts to all clients
        download_times = []
        for client_id in range(self.num_clients):
            download_info = self.calculate_model_download_time(
                model_size_mb, client_id)
            download_times.append(download_info['total_download_time_s'])
            logger.info(f"  Client {client_id} ({download_info['location']}): "
                        f"Download {download_info['total_download_time_s']:.3f}s")

        # Round completion time (max of all operations)
        round_time_s = max(upload_times) + \
            aggregation_time_s + max(download_times)

        return {
            'round_type': 'full_sync',
            'model_size_mb': model_size_mb,
            'max_upload_time_s': max(upload_times),
            'max_download_time_s': max(download_times),
            'aggregation_time_s': aggregation_time_s,
            'total_round_time_s': round_time_s,
            'avg_upload_time_s': statistics.mean(upload_times),
            'avg_download_time_s': statistics.mean(download_times),
            'upload_time_variance': statistics.variance(upload_times) if len(upload_times) > 1 else 0,
            'download_time_variance': statistics.variance(download_times) if len(download_times) > 1 else 0,
            'upload_times': upload_times,
            'download_times': download_times,
        }

    def simulate_multiple_rounds(self, num_rounds: int = 10,
                                 model_size_mb: float = 2.08) -> Dict:
        """Simulate multiple federated learning rounds"""

        logger.info("\n" + "="*70)
        logger.info(f"SIMULATING {num_rounds} FEDERATED TRAINING ROUNDS")
        logger.info("="*70)

        round_results = []
        total_time = 0

        for round_num in range(1, num_rounds + 1):
            logger.info(f"\n📡 Round {round_num}/{num_rounds}")
            result = self.simulate_training_round(model_size_mb)
            round_results.append(result)
            total_time += result['total_round_time_s']

        # Aggregate statistics
        round_times = [r['total_round_time_s'] for r in round_results]

        summary = {
            'num_rounds': num_rounds,
            'model_size_mb': model_size_mb,
            'total_training_time_s': total_time,
            'total_training_time_hours': total_time / 3600,
            'avg_round_time_s': statistics.mean(round_times),
            'min_round_time_s': min(round_times),
            'max_round_time_s': max(round_times),
            'round_time_stddev': statistics.stdev(round_times) if len(round_times) > 1 else 0,
            'round_results': round_results
        }

        return summary

    def analyze_bandwidth_bottleneck(self, model_size_mb: float = 2.08) -> Dict:
        """Analyze which clients/locations are bandwidth bottlenecks"""

        logger.info("\n" + "="*70)
        logger.info("BANDWIDTH BOTTLENECK ANALYSIS")
        logger.info("="*70)

        bottleneck_analysis = []

        for client_id in range(self.num_clients):
            config = self.network_configs[client_id]

            # Upload time is limited by bandwidth
            upload_time = (model_size_mb * 8) / config.bandwidth_mbps

            # Score: higher score = more bottleneck
            # Based on combination of low bandwidth and high latency
            bandwidth_score = 100 / config.bandwidth_mbps  # Inverse bandwidth
            latency_score = config.latency_ms / 10  # Scaled latency
            bottleneck_score = bandwidth_score + latency_score

            analysis = {
                'client_id': client_id,
                'location': config.location_name,
                'bandwidth_mbps': config.bandwidth_mbps,
                'latency_ms': config.latency_ms,
                'upload_time_s': upload_time,
                'bottleneck_score': bottleneck_score,
                'is_critical_bottleneck': bottleneck_score > 15,
                'optimization_needed': upload_time > 5.0,
            }
            bottleneck_analysis.append(analysis)

            logger.info(f"\nClient {client_id} ({config.location_name}):")
            logger.info(f"  Bandwidth: {config.bandwidth_mbps}Mbps")
            logger.info(f"  Latency: {config.latency_ms}ms")
            logger.info(f"  Upload Time: {upload_time:.3f}s")
            logger.info(f"  Bottleneck Score: {bottleneck_score:.2f}")
            if analysis['is_critical_bottleneck']:
                logger.warning("  ⚠️  CRITICAL BOTTLENECK")
            if analysis['optimization_needed']:
                logger.warning("  ⚠️  Optimization recommended")

        return {
            'model_size_mb': model_size_mb,
            'analysis': bottleneck_analysis,
            'critical_bottlenecks': sum(1 for a in bottleneck_analysis if a['is_critical_bottleneck']),
        }

    def simulate_asynchronous_updates(self, num_rounds: int = 10) -> Dict:
        """Simulate asynchronous federated learning (clients don't wait for each other)"""

        logger.info("\n" + "="*70)
        logger.info("ASYNCHRONOUS UPDATE SIMULATION")
        logger.info("="*70)

        client_round_times = [[] for _ in range(self.num_clients)]

        for round_num in range(num_rounds):
            logger.info(f"\nRound {round_num + 1}:")

            # Each client uploads independently
            for client_id in range(self.num_clients):
                upload_info = self.calculate_model_upload_time(2.08, client_id)
                client_round_times[client_id].append(
                    upload_info['total_upload_time_s'])
                logger.info(
                    f"  Client {client_id}: {upload_info['total_upload_time_s']:.3f}s")

        # Stats per client
        client_stats = []
        for client_id in range(self.num_clients):
            times = client_round_times[client_id]
            stats = {
                'client_id': client_id,
                'location': self.network_configs[client_id].location_name,
                'avg_update_time_s': statistics.mean(times),
                'min_update_time_s': min(times),
                'max_update_time_s': max(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
            }
            client_stats.append(stats)

        return {
            'simulation_type': 'asynchronous',
            'num_rounds': num_rounds,
            'client_stats': client_stats,
            'fastest_client_avg': min(s['avg_update_time_s'] for s in client_stats),
            'slowest_client_avg': max(s['avg_update_time_s'] for s in client_stats),
            'sync_overhead': max(s['avg_update_time_s'] for s in client_stats) -
            min(s['avg_update_time_s'] for s in client_stats),
        }

    def save_results(self, results: Dict, filepath: str = 'results/network_simulation_report.json'):
        """Save network simulation results"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"\n✅ Network simulation report saved: {filepath}")


def test_network_simulation():
    """Test network simulation"""

    logger.info("\n" + "="*70)
    logger.info("NETWORK SIMULATION TESTING")
    logger.info("="*70 + "\n")

    simulator = NetworkSimulator(num_clients=4)

    # Test 1: Multiple synchronous rounds
    sync_results = simulator.simulate_multiple_rounds(
        num_rounds=10, model_size_mb=2.08)

    # Test 2: Bandwidth bottleneck analysis
    bottleneck_results = simulator.analyze_bandwidth_bottleneck(
        model_size_mb=2.08)

    # Test 3: Asynchronous updates
    async_results = simulator.simulate_asynchronous_updates(num_rounds=10)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("NETWORK SIMULATION SUMMARY")
    logger.info("="*70)
    logger.info(
        f"Total training time (10 rounds): {sync_results['total_training_time_s']:.2f}s")
    logger.info(f"Average round time: {sync_results['avg_round_time_s']:.3f}s")
    logger.info(
        f"Critical bottlenecks: {bottleneck_results['critical_bottlenecks']}")
    logger.info(
        f"Synchronization overhead: {async_results['sync_overhead']:.3f}s")
    logger.info("="*70 + "\n")

    # Save all results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'synchronous_simulation': sync_results,
        'bottleneck_analysis': bottleneck_results,
        'asynchronous_simulation': async_results,
    }

    simulator.save_results(all_results)

    logger.info("✅ NETWORK SIMULATION TESTING COMPLETE\n")

    return True


if __name__ == "__main__":
    import sys
    success = test_network_simulation()
    sys.exit(0 if success else 1)
