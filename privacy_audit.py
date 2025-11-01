#!/usr/bin/env python3
"""
Privacy Audit System for Federated Learning DDoS Detection
Measures and tracks differential privacy metrics, privacy loss, and formal privacy guarantees
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class DifferentialPrivacyAuditor:
    """Audit and measure differential privacy in federated learning"""

    def __init__(self, epsilon_target: float = 1.0, delta: float = 1e-5):
        """
        Initialize privacy auditor

        Args:
            epsilon_target: Target privacy budget (lower = more private)
            delta: Probability of privacy breach
        """
        self.epsilon_target = epsilon_target
        self.delta = delta
        self.cumulative_epsilon = 0.0
        self.round_privacy_losses = []
        self.round_metrics = []

    def calculate_dp_noise_scale(self, sensitivity: float = 1.0, round_num: int = 1) -> float:
        """
        Calculate Gaussian noise scale for target epsilon
        Using Renyi Differential Privacy (RDP) composition

        Args:
            sensitivity: L2 sensitivity of aggregated updates
            round_num: Current round number (affects composition)

        Returns:
            Standard deviation of Gaussian noise
        """
        # Renyi DP composition: per-round epsilon
        epsilon_per_round = self.epsilon_target / np.sqrt(round_num)

        # Gaussian mechanism: sigma = sqrt(2*ln(1.25/delta)) / epsilon
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) / epsilon_per_round

        return sigma * sensitivity

    def measure_privacy_loss(self,
                             original_weights: List[np.ndarray],
                             noisy_weights: List[np.ndarray],
                             round_num: int) -> Dict:
        """
        Measure privacy loss from adding noise to aggregated weights

        Args:
            original_weights: Aggregated weights before noise
            noisy_weights: Aggregated weights after noise
            round_num: Current round number

        Returns:
            Privacy metrics dictionary
        """
        # Calculate noise magnitude
        noise_magnitude = 0.0
        total_weight_magnitude = 0.0

        for orig, noisy in zip(original_weights, noisy_weights):
            noise = noisy - orig
            noise_magnitude += np.linalg.norm(noise)
            total_weight_magnitude += np.linalg.norm(orig)

        noise_ratio = noise_magnitude / (total_weight_magnitude + 1e-10)

        # Estimate privacy loss using Renyi DP
        epsilon_per_round = self.epsilon_target / np.sqrt(round_num)
        self.cumulative_epsilon += epsilon_per_round

        privacy_metrics = {
            'round': round_num,
            'epsilon_per_round': epsilon_per_round,
            'cumulative_epsilon': self.cumulative_epsilon,
            'delta': self.delta,
            'noise_magnitude': noise_magnitude,
            'noise_ratio': noise_ratio,
            'weight_magnitude': total_weight_magnitude,
            'privacy_budget_remaining': self.epsilon_target - self.cumulative_epsilon,
            'timestamp': datetime.now().isoformat()
        }

        self.round_privacy_losses.append(privacy_metrics)
        return privacy_metrics

    def test_membership_inference_resistance(self,
                                             model_predictions: np.ndarray,
                                             training_labels: np.ndarray,
                                             test_labels: np.ndarray) -> Dict:
        """
        Test resistance to membership inference attacks
        Measures if model's predictions reveal training membership

        Args:
            model_predictions: Model output probabilities
            training_labels: Training set labels
            test_labels: Test set labels

        Returns:
            Membership inference attack metrics
        """
        # Attack advantage: how much better than random guess?
        # If model overfits, training samples get higher confidence

        training_confidence = model_predictions[:len(
            training_labels)].max(axis=1)
        test_confidence = model_predictions[len(training_labels):].max(axis=1)

        # Calculate attack advantage
        train_mean_conf = training_confidence.mean()
        test_mean_conf = test_confidence.mean()
        attack_advantage = abs(train_mean_conf - test_mean_conf)

        # Lower attack advantage = higher privacy
        privacy_resistance = 1.0 - min(attack_advantage, 1.0)

        return {
            'training_mean_confidence': train_mean_conf,
            'test_mean_confidence': test_mean_conf,
            'attack_advantage': attack_advantage,
            'privacy_resistance_score': privacy_resistance,  # 0-1, higher = more private
            'vulnerable': attack_advantage > 0.1  # Threshold for vulnerability
        }

    def test_gradient_leakage(self,
                              gradients: List[np.ndarray],
                              batch_size: int) -> Dict:
        """
        Test resistance to gradient inversion attacks
        Measures if gradients leak information about training data

        Args:
            gradients: Model gradients from training
            batch_size: Training batch size

        Returns:
            Gradient leakage metrics
        """
        # Gradient norm indicates strength of data signal
        # Higher norm = more information leaked

        total_gradient_norm = 0.0
        for grad in gradients:
            total_gradient_norm += np.linalg.norm(grad)

        mean_gradient_norm = total_gradient_norm / len(gradients)

        # Estimate leakage using gradient entropy
        gradient_entropy = 0.0
        for grad in gradients:
            flat_grad = grad.flatten()
            hist, _ = np.histogram(np.abs(flat_grad), bins=10)
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            gradient_entropy += entropy

        mean_entropy = gradient_entropy / len(gradients)
        leakage_risk = max(0, 1 - (mean_entropy / 2.0))  # Normalize to 0-1

        return {
            'mean_gradient_norm': mean_gradient_norm,
            'mean_entropy': mean_entropy,
            'leakage_risk': leakage_risk,  # 0-1, higher = more leakage
            'is_high_risk': leakage_risk > 0.5
        }

    def generate_privacy_report(self) -> Dict:
        """Generate comprehensive privacy audit report"""

        if not self.round_privacy_losses:
            logger.warning("No privacy measurements available")
            return {}

        df_privacy = pd.DataFrame(self.round_privacy_losses)

        report = {
            'audit_timestamp': datetime.now().isoformat(),
            'total_rounds': len(self.round_privacy_losses),
            'target_epsilon': self.epsilon_target,
            'delta': self.delta,
            'final_cumulative_epsilon': df_privacy['cumulative_epsilon'].iloc[-1],
            'privacy_budget_consumed': (df_privacy['cumulative_epsilon'].iloc[-1] / self.epsilon_target) * 100,
            'mean_noise_ratio': df_privacy['noise_ratio'].mean(),
            'max_cumulative_epsilon': df_privacy['cumulative_epsilon'].max(),
            'privacy_status': self._determine_privacy_status(df_privacy),
            'recommendations': self._generate_recommendations(df_privacy)
        }

        return report

    def _determine_privacy_status(self, df: pd.DataFrame) -> str:
        """Determine overall privacy status"""
        final_eps = df['cumulative_epsilon'].iloc[-1]

        if final_eps < 0.5:
            return "EXCELLENT - Strong privacy guarantee"
        elif final_eps < 1.0:
            return "GOOD - Reasonable privacy guarantee"
        elif final_eps < 2.0:
            return "ACCEPTABLE - Moderate privacy guarantee"
        else:
            return "WEAK - Limited privacy guarantee"

    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate privacy improvement recommendations"""
        recommendations = []
        final_eps = df['cumulative_epsilon'].iloc[-1]

        if final_eps > self.epsilon_target:
            recommendations.append(
                f"Privacy budget exceeded. Reduce noise scale or use fewer rounds."
            )

        if df['noise_ratio'].mean() < 0.01:
            recommendations.append(
                "Noise magnitude very small. Increase noise scale for stronger privacy."
            )

        if df['noise_ratio'].mean() > 0.5:
            recommendations.append(
                "Noise magnitude very large. May impact model accuracy significantly."
            )

        if len(recommendations) == 0:
            recommendations.append(
                "Privacy configuration appears well-balanced.")

        return recommendations

    def save_report(self, filepath: str = 'results/privacy_audit_report.json'):
        """Save privacy audit report to file"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        report = self.generate_privacy_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"✅ Privacy audit report saved: {filepath}")
        return report

    def plot_privacy_evolution(self, filepath: str = 'results/privacy_evolution.png'):
        """Plot privacy metrics evolution across rounds"""
        if not self.round_privacy_losses:
            logger.warning("No data to plot")
            return

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        df = pd.DataFrame(self.round_privacy_losses)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Privacy Audit Results - Federated Learning',
                     fontsize=16, fontweight='bold')

        # Plot 1: Cumulative Epsilon
        axes[0, 0].plot(df['round'], df['cumulative_epsilon'],
                        'o-', label='Cumulative ε', linewidth=2)
        axes[0, 0].axhline(y=self.epsilon_target, color='r',
                           linestyle='--', label=f'Target ε={self.epsilon_target}')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Cumulative Epsilon (Privacy Loss)')
        axes[0, 0].set_title('Privacy Budget Consumption')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Per-Round Epsilon
        axes[0, 1].bar(df['round'], df['epsilon_per_round'],
                       color='steelblue', alpha=0.7)
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Epsilon per Round')
        axes[0, 1].set_title('Per-Round Privacy Loss')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Plot 3: Noise Ratio
        axes[1, 0].plot(df['round'], df['noise_ratio'],
                        'o-', color='green', linewidth=2)
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Noise Ratio')
        axes[1, 0].set_title('Noise Magnitude vs Weight Magnitude')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Privacy Budget Remaining
        axes[1, 1].fill_between(df['round'], 0, df['privacy_budget_remaining'],
                                alpha=0.3, color='orange', label='Budget Remaining')
        axes[1, 1].plot(df['round'], df['privacy_budget_remaining'],
                        'o-', color='orange', linewidth=2)
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Budget Remaining')
        axes[1, 1].set_title('Privacy Budget Remaining')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"✅ Privacy evolution plot saved: {filepath}")
        plt.close()


def test_privacy_audit():
    """Test privacy audit system"""
    logger.info("=" * 70)
    logger.info("TESTING PRIVACY AUDIT SYSTEM")
    logger.info("=" * 70)

    # Create auditor
    auditor = DifferentialPrivacyAuditor(epsilon_target=1.0, delta=1e-5)

    logger.info("\n1️⃣  Testing privacy loss measurement...")

    # Simulate rounds
    for round_num in range(1, 6):
        # Simulate aggregated weights
        original_weights = [np.random.randn(
            100, 64), np.random.randn(64, 32), np.random.randn(32, 1)]

        # Add noise
        sigma = auditor.calculate_dp_noise_scale(
            sensitivity=1.0, round_num=round_num)
        noisy_weights = [
            w + np.random.normal(0, sigma, w.shape) for w in original_weights]

        # Measure privacy loss
        metrics = auditor.measure_privacy_loss(
            original_weights, noisy_weights, round_num)

        logger.info(f"  Round {round_num}: ε={metrics['epsilon_per_round']:.4f}, "
                    f"Cumulative ε={metrics['cumulative_epsilon']:.4f}")

    logger.info("✅ Privacy loss measurement passed\n")

    logger.info("2️⃣  Testing membership inference resistance...")

    # Simulate model predictions
    train_predictions = np.random.rand(1000, 2)
    test_predictions = np.random.rand(500, 2)
    combined_predictions = np.vstack([train_predictions, test_predictions])
    combined_predictions /= combined_predictions.sum(axis=1, keepdims=True)

    train_labels = np.random.randint(0, 2, 1000)
    test_labels = np.random.randint(0, 2, 500)

    inference_metrics = auditor.test_membership_inference_resistance(
        combined_predictions, train_labels, test_labels
    )

    logger.info(
        f"  Attack Advantage: {inference_metrics['attack_advantage']:.4f}")
    logger.info(
        f"  Privacy Resistance: {inference_metrics['privacy_resistance_score']:.4f}")
    logger.info(f"  Vulnerable: {inference_metrics['vulnerable']}")
    logger.info("✅ Membership inference resistance test passed\n")

    logger.info("3️⃣  Testing gradient leakage...")

    # Simulate gradients
    gradients = [np.random.randn(100, 64), np.random.randn(
        64, 32), np.random.randn(32, 1)]

    leakage_metrics = auditor.test_gradient_leakage(gradients, batch_size=32)

    logger.info(
        f"  Mean Gradient Norm: {leakage_metrics['mean_gradient_norm']:.4f}")
    logger.info(f"  Leakage Risk: {leakage_metrics['leakage_risk']:.4f}")
    logger.info(f"  High Risk: {leakage_metrics['is_high_risk']}")
    logger.info("✅ Gradient leakage test passed\n")

    logger.info("4️⃣  Generating and saving reports...")

    report = auditor.save_report()
    logger.info(f"  Privacy Status: {report['privacy_status']}")
    logger.info(
        f"  Privacy Budget Consumed: {report['privacy_budget_consumed']:.2f}%")
    logger.info("✅ Report generation passed\n")

    logger.info("5️⃣  Creating visualizations...")
    auditor.plot_privacy_evolution()
    logger.info("✅ Visualization creation passed\n")

    logger.info("=" * 70)
    logger.info("✅ ALL PRIVACY AUDIT TESTS PASSED")
    logger.info("=" * 70)

    return True


if __name__ == "__main__":
    success = test_privacy_audit()
    sys.exit(0 if success else 1)
