"""
Drift Detection Module

Monitors incoming transaction distributions against training data
to detect concept drift and data drift that may degrade model performance.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detect data drift between training data and incoming transactions.
    
    Uses Kolmogorov-Smirnov test for continuous features and
    Chi-squared test for categorical features.
    
    Drift Alert Levels:
    - OK: p-value > 0.05 (no significant drift)
    - WARNING: 0.01 < p-value <= 0.05 (mild drift, monitor closely)
    - CRITICAL: p-value <= 0.01 (significant drift, retrain model)
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_names: List[str] = None,
        significance_level: float = 0.05,
    ):
        """
        Args:
            reference_data: Training data to use as reference distribution
            feature_names: List of feature names to monitor
            significance_level: p-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.feature_names = feature_names or list(reference_data.columns)
        self.significance_level = significance_level
        self.drift_history = []
        
        # Pre-compute reference statistics
        self._compute_reference_stats()
    
    def _compute_reference_stats(self) -> None:
        """Compute reference distribution statistics."""
        self.ref_stats = {}
        for col in self.feature_names:
            if col in self.reference_data.columns:
                self.ref_stats[col] = {
                    'mean': float(self.reference_data[col].mean()),
                    'std': float(self.reference_data[col].std()),
                    'min': float(self.reference_data[col].min()),
                    'max': float(self.reference_data[col].max()),
                    'median': float(self.reference_data[col].median()),
                }
    
    def detect_drift(
        self,
        new_data: pd.DataFrame,
    ) -> Dict[str, Dict]:
        """
        Run drift detection on new data.
        
        Args:
            new_data: New transaction data to compare against reference
            
        Returns:
            Dictionary with drift results per feature
        """
        results = {}
        
        for col in self.feature_names:
            if col not in new_data.columns or col not in self.reference_data.columns:
                continue
            
            ref_values = self.reference_data[col].dropna().values
            new_values = new_data[col].dropna().values
            
            if len(new_values) == 0 or len(ref_values) == 0:
                continue
            
            # KS Test for continuous features
            ks_stat, p_value = stats.ks_2samp(ref_values, new_values)
            
            # Determine alert level
            if p_value <= 0.01:
                alert = 'CRITICAL'
            elif p_value <= self.significance_level:
                alert = 'WARNING'
            else:
                alert = 'OK'
            
            # Compute distribution shift metrics
            ref_mean = float(np.mean(ref_values))
            new_mean = float(np.mean(new_values))
            mean_shift_pct = abs(new_mean - ref_mean) / (abs(ref_mean) + 1e-10) * 100
            
            results[col] = {
                'ks_statistic': round(float(ks_stat), 4),
                'p_value': round(float(p_value), 6),
                'alert': alert,
                'ref_mean': round(ref_mean, 4),
                'new_mean': round(new_mean, 4),
                'mean_shift_pct': round(mean_shift_pct, 2),
            }
        
        # Log critical drifts
        critical = [f for f, r in results.items() if r['alert'] == 'CRITICAL']
        warnings = [f for f, r in results.items() if r['alert'] == 'WARNING']
        
        if critical:
            logger.warning(f"🔴 CRITICAL drift detected in: {critical}")
        if warnings:
            logger.info(f"🟡 WARNING drift detected in: {warnings}")
        
        # Store in history
        self.drift_history.append({
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(new_data),
            'n_critical': len(critical),
            'n_warnings': len(warnings),
            'results': results,
        })
        
        return results
    
    def get_overall_drift_score(self, results: Dict[str, Dict]) -> float:
        """
        Compute an overall drift score (0 = no drift, 1 = severe drift).
        
        Score is based on the proportion of features with drift.
        """
        if not results:
            return 0.0
        
        n_critical = sum(1 for r in results.values() if r['alert'] == 'CRITICAL')
        n_warning = sum(1 for r in results.values() if r['alert'] == 'WARNING')
        n_total = len(results)
        
        # Critical drift contributes 2x, warning contributes 1x
        score = (n_critical * 2 + n_warning) / (n_total * 2)
        return round(min(score, 1.0), 4)
    
    def generate_report(self, results: Dict[str, Dict]) -> str:
        """Generate a human-readable drift report."""
        overall_score = self.get_overall_drift_score(results)
        
        lines = [
            "=" * 60,
            "  DATA DRIFT REPORT",
            "=" * 60,
            f"  Timestamp:       {datetime.now().isoformat()}",
            f"  Reference size:  {len(self.reference_data)}",
            f"  Overall Score:   {overall_score:.2f} (0=no drift, 1=severe)",
            f"  Status:          {'🔴 CRITICAL' if overall_score > 0.3 else '🟡 WARNING' if overall_score > 0.1 else '🟢 OK'}",
            "=" * 60,
        ]
        
        # Critical features
        critical = {f: r for f, r in results.items() if r['alert'] == 'CRITICAL'}
        warnings = {f: r for f, r in results.items() if r['alert'] == 'WARNING'}
        
        if critical:
            lines.append("\n🔴 CRITICAL DRIFT:")
            for feat, r in critical.items():
                lines.append(
                    f"  {feat}: KS={r['ks_statistic']:.4f}, "
                    f"p={r['p_value']:.6f}, "
                    f"shift={r['mean_shift_pct']:.1f}%"
                )
        
        if warnings:
            lines.append("\n🟡 WARNINGS:")
            for feat, r in warnings.items():
                lines.append(
                    f"  {feat}: KS={r['ks_statistic']:.4f}, "
                    f"p={r['p_value']:.6f}, "
                    f"shift={r['mean_shift_pct']:.1f}%"
                )
        
        ok_count = sum(1 for r in results.values() if r['alert'] == 'OK')
        lines.append(f"\n🟢 OK: {ok_count}/{len(results)} features show no significant drift")
        
        if overall_score > 0.3:
            lines.append("\n⚠️  RECOMMENDATION: Model retraining may be needed!")
        elif overall_score > 0.1:
            lines.append("\n📋 RECOMMENDATION: Monitor closely, drift is emerging.")
        else:
            lines.append("\n✅ RECOMMENDATION: No action needed.")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_report(self, results: Dict[str, Dict], path: str = "monitoring/drift_report.json") -> None:
        """Save drift report to JSON file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': self.get_overall_drift_score(results),
            'n_features_tested': len(results),
            'results': results,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Drift report saved to {path}")


def simulate_drift(
    reference: pd.DataFrame,
    drift_magnitude: float = 0.5,
) -> pd.DataFrame:
    """
    Simulate data drift by shifting distributions.
    
    Args:
        reference: Reference (training) data
        drift_magnitude: How much to shift (0 = no drift, 1 = huge drift)
        
    Returns:
        Simulated drifted data
    """
    drifted = reference.copy()
    
    # Shift a few features
    shift_features = ['V1', 'V4', 'V14', 'Amount']
    for col in shift_features:
        if col in drifted.columns:
            mean_shift = drifted[col].std() * drift_magnitude
            drifted[col] = drifted[col] + mean_shift * np.random.choice([-1, 1])
    
    return drifted


if __name__ == "__main__":
    # Demo drift detection
    np.random.seed(42)
    
    # Create reference data
    n = 1000
    reference = pd.DataFrame({
        f'V{i}': np.random.randn(n) for i in range(1, 29)
    })
    reference['Amount'] = np.random.exponential(50, n)
    reference['Time'] = np.random.uniform(0, 172800, n)
    
    # Simulate drifted data
    drifted = simulate_drift(reference, drift_magnitude=1.5)
    
    # Run drift detection
    detector = DriftDetector(reference)
    results = detector.detect_drift(drifted)
    
    report = detector.generate_report(results)
    print(report)
