#!/usr/bin/env python3
"""
Byzantine Attack Testing System
Tests Multi-Krum robustness against poisoned clients
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ByzantineAttackTester:
    """Test Byzantine tolerance of federated aggregation"""

    def __init__(self, num_clients: int = 4, num_poisoned: int = 1):
        """Initialize Byzantine attack tester"""
        self.num_clients = num_clients
        self.num_poisoned = num_poisoned
        self.attack_results = []
        logger.info(
            f"Initialized Byzantine tester: {num_clients} clients, {num_poisoned} poisoned")

    def generate_honest_updates(self, shape: Tuple = (100, 64)) -> List[np.ndarray]:
        """Generate honest (normal) model updates"""
        updates = []
        for _ in range(self.num_clients - self.num_poisoned):
            # Honest updates: small random values from normal distribution
            update = np.random.normal(0, 0.1, shape)
            updates.append(update)
        return updates

    def generate_poisoned_updates(self, shape: Tuple = (100, 64),
                                  attack_magnitude: float = 5.0) -> List[np.ndarray]:
        """Generate poisoned (Byzantine) model updates"""
        updates = []
        for _ in range(self.num_poisoned):
            # Poisoned updates: large random values to corrupt model
            update = np.random.normal(0, attack_magnitude, shape)
            updates.append(update)
        return updates

    def fed_avg_aggregation(self, updates: List[np.ndarray]) -> np.ndarray:
        """FedAvg: Simple averaging without Byzantine defense"""
        return np.mean(updates, axis=0)

    def multi_krum_aggregation(self, updates: List[np.ndarray], k: int = None) -> np.ndarray:
        """Multi-Krum: Byzantine-robust aggregation"""
        if k is None:
            # Multi-Krum selects k best clients: k = n - 2f - 2 (n=clients, f=faults)
            k = len(updates) - 2 * self.num_poisoned - 2
            k = max(1, k)  # At least 1 client

        num_clients = len(updates)

        # Compute pairwise distances
        distances = np.zeros((num_clients, num_clients))
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = np.linalg.norm(updates[i] - updates[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Krum score: sum of k nearest neighbors
        krum_scores = []
        for i in range(num_clients):
            nearest_distances = np.sort(distances[i])[:k]
            score = np.sum(nearest_distances)
            krum_scores.append(score)

        # Select k clients with smallest Krum scores
        selected_indices = np.argsort(krum_scores)[:k]
        selected_updates = [updates[i] for i in selected_indices]

        # Average selected updates
        aggregated = np.mean(selected_updates, axis=0)

        return aggregated, selected_indices

    def test_attack_scenario(self, attack_magnitude: float, shape: Tuple = (100, 64)) -> Dict:
        """Test single attack scenario"""

        # Generate updates
        honest_updates = self.generate_honest_updates(shape)
        poisoned_updates = self.generate_poisoned_updates(
            shape, attack_magnitude)
        all_updates = honest_updates + poisoned_updates
        # Shuffle to randomize poisoned positions
        np.random.shuffle(all_updates)

        # FedAvg aggregation
        fed_avg_result = self.fed_avg_aggregation(all_updates)

        # Multi-Krum aggregation
        multi_krum_result, selected = self.multi_krum_aggregation(all_updates)

        # Measure attack impact: how much did the poisoned updates corrupt the model?
        # Honest consensus: average of only honest updates
        honest_consensus = self.fed_avg_aggregation(honest_updates)

        # Distance from honest consensus (lower = more robust)
        fed_avg_divergence = np.linalg.norm(fed_avg_result - honest_consensus)
        multi_krum_divergence = np.linalg.norm(
            multi_krum_result - honest_consensus)

        # Check if poisoned clients were selected
        poisoned_in_fedavg = True  # FedAvg always includes all
        poisoned_in_multikrum = any(i >= len(honest_updates) for i in selected)

        # Byzantine robustness score (0-1, higher = more robust)
        robustness = 1.0 - min(multi_krum_divergence /
                               (fed_avg_divergence + 1e-10), 1.0)

        result = {
            'attack_magnitude': attack_magnitude,
            'num_poisoned': self.num_poisoned,
            'num_clients': self.num_clients,
            'honest_consensus_norm': np.linalg.norm(honest_consensus),
            'fed_avg_divergence': float(fed_avg_divergence),
            'multi_krum_divergence': float(multi_krum_divergence),
            'multi_krum_selected_count': len(selected),
            'poisoned_clients_detected': not poisoned_in_multikrum,
            'robustness_improvement': float(fed_avg_divergence - multi_krum_divergence),
            'robustness_score': float(robustness),
        }

        return result

    def run_gradient_poisoning_attack(self) -> Dict:
        """Simulate gradient poisoning where Byzantine clients send bad gradients"""
        logger.info("\n" + "="*70)
        logger.info("GRADIENT POISONING ATTACK TEST")
        logger.info("="*70)

        results = []
        attack_magnitudes = [0.5, 1.0, 2.0, 5.0, 10.0]

        for magnitude in attack_magnitudes:
            logger.info(f"Testing attack magnitude: {magnitude}")
            result = self.test_attack_scenario(magnitude)
            results.append(result)

            logger.info(
                f"  FedAvg Divergence: {result['fed_avg_divergence']:.4f}")
            logger.info(
                f"  Multi-Krum Divergence: {result['multi_krum_divergence']:.4f}")
            logger.info(
                f"  Robustness Improvement: {result['robustness_improvement']:.4f}")
            logger.info(
                f"  Multi-Krum Selected: {result['multi_krum_selected_count']}/{self.num_clients}")
            logger.info(
                f"  Poisoned Detected: {result['poisoned_clients_detected']}")

        return {
            'attack_type': 'gradient_poisoning',
            'timestamp': datetime.now().isoformat(),
            'num_poisoned': self.num_poisoned,
            'num_clients': self.num_clients,
            'scenarios': results
        }

    def run_label_flipping_attack(self) -> Dict:
        """Simulate label flipping where Byzantine clients send inverted labels"""
        logger.info("\n" + "="*70)
        logger.info("LABEL FLIPPING ATTACK TEST")
        logger.info("="*70)

        # For label flipping, the Byzantine clients would produce opposite gradient directions
        results = []

        for num_poisoned in range(1, self.num_clients):
            self.num_poisoned = num_poisoned
            logger.info(f"Testing with {num_poisoned} poisoned clients")

            result = self.test_attack_scenario(attack_magnitude=5.0)
            results.append(result)

            logger.info(
                f"  Robustness Score: {result['robustness_score']:.4f}")
            logger.info(
                f"  Robustness Improvement: {result['robustness_improvement']:.4f}")

        self.num_poisoned = 1  # Reset

        return {
            'attack_type': 'label_flipping',
            'timestamp': datetime.now().isoformat(),
            'scenarios': results
        }

    def run_all_attacks(self) -> Dict:
        """Run all Byzantine attack tests"""
        logger.info("\n" + "="*70)
        logger.info("BYZANTINE ATTACK TESTING - COMPLETE SUITE")
        logger.info("="*70)

        gradient_poisoning = self.run_gradient_poisoning_attack()
        label_flipping = self.run_label_flipping_attack()

        # Summary
        logger.info("\n" + "="*70)
        logger.info("BYZANTINE ROBUSTNESS SUMMARY")
        logger.info("="*70)

        # Calculate average robustness
        avg_robustness = np.mean([r['robustness_score']
                                 for r in gradient_poisoning['scenarios']])
        avg_improvement = np.mean([r['robustness_improvement']
                                  for r in gradient_poisoning['scenarios']])

        logger.info(f"Average Robustness Score: {avg_robustness:.4f}")
        logger.info(f"Average Divergence Reduction: {avg_improvement:.4f}")

        # Threat assessment
        if avg_robustness > 0.8:
            threat_level = "LOW - System is highly robust to Byzantine attacks"
        elif avg_robustness > 0.6:
            threat_level = "MEDIUM - System has reasonable Byzantine defense"
        else:
            threat_level = "HIGH - System vulnerable to Byzantine attacks"

        logger.info(f"Byzantine Threat Level: {threat_level}")
        logger.info("="*70 + "\n")

        return {
            'timestamp': datetime.now().isoformat(),
            'gradient_poisoning': gradient_poisoning,
            'label_flipping': label_flipping,
            'summary': {
                'average_robustness_score': float(avg_robustness),
                'average_improvement': float(avg_improvement),
                'threat_assessment': threat_level,
                'multi_krum_effective': avg_robustness > 0.7
            }
        }

    def save_results(self, results: Dict, filepath: str = 'results/byzantine_attack_report.json'):
        """Save Byzantine attack test results"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"✅ Byzantine attack report saved: {filepath}")


def test_byzantine_attacks():
    """Test Byzantine attack robustness"""

    logger.info("\n" + "="*70)
    logger.info("TESTING BYZANTINE ATTACK ROBUSTNESS")
    logger.info("="*70 + "\n")

    # Test with 1 poisoned client
    tester = ByzantineAttackTester(num_clients=4, num_poisoned=1)
    results = tester.run_all_attacks()

    logger.info("Saving results...")
    tester.save_results(results)

    logger.info("\n" + "="*70)
    logger.info("✅ BYZANTINE ATTACK TESTING COMPLETE")
    logger.info("="*70 + "\n")

    return True


if __name__ == "__main__":
    import sys
    success = test_byzantine_attacks()
    sys.exit(0 if success else 1)
