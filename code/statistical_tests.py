

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

@dataclass
class StatisticalResult:
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    confidence_level: float = 0.95
    is_significant: bool = False
    alpha: float = 0.05
    sample_size: Optional[int] = None
    power: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "effect_size": float(self.effect_size) if self.effect_size is not None else None,
            "effect_size_interpretation": self.effect_size_interpretation,
            "confidence_interval": self.confidence_interval,
            "confidence_level": self.confidence_level,
            "is_significant": self.is_significant,
            "alpha": self.alpha,
            "sample_size": self.sample_size,
            "power": float(self.power) if self.power is not None else None
        }

class StatisticalAnalyzer:

    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95):
        self.alpha = alpha
        self.confidence_level = confidence_level

    def cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0, "undefined (zero variance)"

        d = (np.mean(group1) - np.mean(group2)) / pooled_std

        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return d, interpretation

    def cohens_d_one_sample(self, data: np.ndarray, population_mean: float = 0) -> Tuple[float, str]:
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0, "undefined (zero variance)"

        d = (np.mean(data) - population_mean) / std

        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return d, interpretation

    def eta_squared(self, f_statistic: float, df_between: int, df_within: int) -> Tuple[float, str]:
        eta_sq = (f_statistic * df_between) / (f_statistic * df_between + df_within)

        if eta_sq < 0.01:
            interpretation = "negligible"
        elif eta_sq < 0.06:
            interpretation = "small"
        elif eta_sq < 0.14:
            interpretation = "medium"
        else:
            interpretation = "large"

        return eta_sq, interpretation

    def confidence_interval_mean(self, data: np.ndarray) -> Tuple[float, float]:
        n = len(data)
        mean = np.mean(data)
        se = stats.sem(data)

        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, n - 1)
        margin = t_critical * se

        return (mean - margin, mean + margin)

    def confidence_interval_difference(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float]:
        n1, n2 = len(group1), len(group2)
        mean_diff = np.mean(group1) - np.mean(group2)

        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        se_diff = np.sqrt(var1 / n1 + var2 / n2)

        df = (var1 / n1 + var2 / n2) ** 2 / (
            (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        )

        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df)
        margin = t_critical * se_diff

        return (mean_diff - margin, mean_diff + margin)

    def power_analysis_t_test(self, effect_size: float, n: int, alpha: float = None) -> float:
        if alpha is None:
            alpha = self.alpha

        ncp = effect_size * np.sqrt(n / 2)

        t_critical = stats.t.ppf(1 - alpha / 2, 2 * n - 2)

        power = 1 - stats.nct.cdf(t_critical, 2 * n - 2, ncp) + stats.nct.cdf(-t_critical, 2 * n - 2, ncp)

        return power

    def required_sample_size(self, effect_size: float, power: float = 0.8, alpha: float = None) -> int:
        if alpha is None:
            alpha = self.alpha

        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))

    def independent_t_test(self, group1: np.ndarray, group2: np.ndarray,
                           equal_var: bool = False) -> StatisticalResult:
        group1 = np.asarray(group1)
        group2 = np.asarray(group2)

        t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)

        d, d_interp = self.cohens_d(group1, group2)

        ci = self.confidence_interval_difference(group1, group2)

        n_avg = (len(group1) + len(group2)) / 2
        power = self.power_analysis_t_test(abs(d), int(n_avg))

        return StatisticalResult(
            test_name="Independent t-test (Welch's)" if not equal_var else "Independent t-test (Student's)",
            statistic=t_stat,
            p_value=p_value,
            effect_size=d,
            effect_size_interpretation=d_interp,
            confidence_interval=ci,
            confidence_level=self.confidence_level,
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            sample_size=len(group1) + len(group2),
            power=power
        )

    def paired_t_test(self, before: np.ndarray, after: np.ndarray) -> StatisticalResult:
        before = np.asarray(before)
        after = np.asarray(after)

        if len(before) != len(after):
            raise ValueError("Paired samples must have equal length")

        t_stat, p_value = stats.ttest_rel(before, after)

        diff = after - before
        d, d_interp = self.cohens_d_one_sample(diff)

        ci = self.confidence_interval_mean(diff)

        return StatisticalResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            effect_size=d,
            effect_size_interpretation=d_interp,
            confidence_interval=ci,
            confidence_level=self.confidence_level,
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            sample_size=len(before)
        )

    def one_sample_t_test(self, data: np.ndarray, population_mean: float = 0) -> StatisticalResult:
        data = np.asarray(data)

        t_stat, p_value = stats.ttest_1samp(data, population_mean)

        d, d_interp = self.cohens_d_one_sample(data, population_mean)
        ci = self.confidence_interval_mean(data)

        return StatisticalResult(
            test_name="One-sample t-test",
            statistic=t_stat,
            p_value=p_value,
            effect_size=d,
            effect_size_interpretation=d_interp,
            confidence_interval=ci,
            confidence_level=self.confidence_level,
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            sample_size=len(data)
        )

    def one_way_anova(self, *groups) -> StatisticalResult:
        groups = [np.asarray(g) for g in groups]

        f_stat, p_value = stats.f_oneway(*groups)

        k = len(groups)
        n_total = sum(len(g) for g in groups)
        df_between = k - 1
        df_within = n_total - k

        eta_sq, eta_interp = self.eta_squared(f_stat, df_between, df_within)

        return StatisticalResult(
            test_name="One-way ANOVA",
            statistic=f_stat,
            p_value=p_value,
            effect_size=eta_sq,
            effect_size_interpretation=eta_interp,
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            sample_size=n_total
        )

    def bonferroni_correction(self, p_values: List[float]) -> Tuple[List[float], List[bool]]:
        n = len(p_values)
        adjusted_p = [min(p * n, 1.0) for p in p_values]
        is_significant = [p < self.alpha for p in adjusted_p]

        return adjusted_p, is_significant

    def fdr_correction(self, p_values: List[float]) -> Tuple[List[float], List[bool]]:
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        adjusted_p = np.zeros(n)
        for i, (idx, p) in enumerate(zip(sorted_indices, sorted_p)):
            adjusted_p[idx] = p * n / (i + 1)

        adjusted_p = np.minimum.accumulate(adjusted_p[::-1])[::-1]
        adjusted_p = np.minimum(adjusted_p, 1.0)

        is_significant = [p < self.alpha for p in adjusted_p]

        return adjusted_p.tolist(), is_significant

    def generate_report(self, result: StatisticalResult) -> str:
        lines = [
            f"=== {result.test_name} ===",
            f"Statistic: {result.statistic:.4f}",
            f"p-value: {result.p_value:.6f}",
            f"Significance level (alpha): {result.alpha}",
            f"Conclusion: {'significant' if result.is_significant else 'not significant'} (p {'<' if result.is_significant else '>'} α)",
        ]

        if result.effect_size is not None:
            lines.append(f"Effect size: {result.effect_size:.4f} ({result.effect_size_interpretation})")

        if result.confidence_interval is not None:
            lines.append(f"{result.confidence_level*100:.0f}% CI: [{result.confidence_interval[0]:.6f}, {result.confidence_interval[1]:.6f}]")

        if result.sample_size is not None:
            lines.append(f"Sample size: {result.sample_size}")

        if result.power is not None:
            lines.append(f"Statistical power: {result.power:.4f}")
            if result.power < 0.8:
                lines.append(f"  Warning: power < 0.8 (Type II error risk)")

        return "\n".join(lines)

class ExperimentStatistics:

    def __init__(self, alpha: float = 0.05):
        self.analyzer = StatisticalAnalyzer(alpha=alpha)

    def compare_baseline_vs_suppressed(self,
                                       baseline_confidences: List[float],
                                       suppressed_confidences: List[float]) -> Dict[str, Any]:
        baseline = np.array(baseline_confidences)
        suppressed = np.array(suppressed_confidences)

        descriptive = {
            "baseline": {
                "mean": float(np.mean(baseline)),
                "std": float(np.std(baseline, ddof=1)),
                "median": float(np.median(baseline)),
                "min": float(np.min(baseline)),
                "max": float(np.max(baseline)),
                "n": len(baseline)
            },
            "suppressed": {
                "mean": float(np.mean(suppressed)),
                "std": float(np.std(suppressed, ddof=1)),
                "median": float(np.median(suppressed)),
                "min": float(np.min(suppressed)),
                "max": float(np.max(suppressed)),
                "n": len(suppressed)
            }
        }

        mean_change = descriptive["suppressed"]["mean"] - descriptive["baseline"]["mean"]
        percent_change = (mean_change / descriptive["baseline"]["mean"]) * 100 if descriptive["baseline"]["mean"] != 0 else 0

        if len(baseline) == len(suppressed):

            t_test_result = self.analyzer.paired_t_test(baseline, suppressed)
        else:

            t_test_result = self.analyzer.independent_t_test(baseline, suppressed)

        baseline_ci = self.analyzer.confidence_interval_mean(baseline)
        suppressed_ci = self.analyzer.confidence_interval_mean(suppressed)

        if t_test_result.effect_size is not None:
            required_n = self.analyzer.required_sample_size(abs(t_test_result.effect_size), power=0.8)
        else:
            required_n = None

        return {
            "descriptive": descriptive,
            "change": {
                "absolute": mean_change,
                "percentage": percent_change
            },
            "hypothesis_test": t_test_result.to_dict(),
            "confidence_intervals": {
                "baseline": baseline_ci,
                "suppressed": suppressed_ci
            },
            "power_analysis": {
                "current_power": t_test_result.power,
                "required_n_for_80_power": required_n,
                "is_underpowered": t_test_result.power < 0.8 if t_test_result.power else True
            },
            "conclusion": {
                "is_significant": t_test_result.is_significant,
                "effect_size": t_test_result.effect_size,
                "effect_interpretation": t_test_result.effect_size_interpretation,
                "p_value": t_test_result.p_value
            }
        }

    def compare_multiple_conditions(self, conditions: Dict[str, List[float]]) -> Dict[str, Any]:
        groups = list(conditions.values())
        group_names = list(conditions.keys())

        anova_result = self.analyzer.one_way_anova(*groups)

        pairwise_results = {}
        p_values = []

        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                name1, name2 = group_names[i], group_names[j]
                comparison_name = f"{name1} vs {name2}"

                result = self.analyzer.independent_t_test(
                    np.array(groups[i]),
                    np.array(groups[j])
                )
                pairwise_results[comparison_name] = result.to_dict()
                p_values.append(result.p_value)

        bonferroni_p, bonferroni_sig = self.analyzer.bonferroni_correction(p_values)
        fdr_p, fdr_sig = self.analyzer.fdr_correction(p_values)

        comparison_names = list(pairwise_results.keys())
        for i, name in enumerate(comparison_names):
            pairwise_results[name]["bonferroni_p"] = bonferroni_p[i]
            pairwise_results[name]["bonferroni_significant"] = bonferroni_sig[i]
            pairwise_results[name]["fdr_p"] = fdr_p[i]
            pairwise_results[name]["fdr_significant"] = fdr_sig[i]

        return {
            "anova": anova_result.to_dict(),
            "pairwise_comparisons": pairwise_results,
            "multiple_comparison_correction": {
                "method": ["Bonferroni", "FDR (Benjamini-Hochberg)"],
                "n_comparisons": len(p_values)
            }
        }

    def analyze_effect_detection(self,
                                baseline_confidences: List[float],
                                suppressed_confidences: List[float],
                                threshold_percentage: float = 5.0) -> Dict[str, Any]:
        comparison = self.compare_baseline_vs_suppressed(
            baseline_confidences,
            suppressed_confidences
        )

        is_statistically_significant = comparison["conclusion"]["is_significant"]

        effect_size = comparison["conclusion"]["effect_size"]
        has_meaningful_effect = (
            effect_size is not None and
            comparison["conclusion"]["effect_interpretation"] in ["medium", "large"]
        )

        percent_change = comparison["change"]["percentage"]
        old_method_would_detect = abs(percent_change) > threshold_percentage

        return {
            "statistical_analysis": comparison,
            "effect_detection": {
                "statistically_significant": is_statistically_significant,
                "practically_significant": has_meaningful_effect,
                "percent_change": percent_change,
                "old_method_threshold": threshold_percentage,
                "old_method_result": old_method_would_detect,
                "methods_agree": is_statistically_significant == old_method_would_detect
            },
            "recommendation": (
                "SIGNIFICANT: Both statistical significance and meaningful effect size detected"
                if is_statistically_significant and has_meaningful_effect
                else "MARGINAL: Statistically significant but small effect size"
                if is_statistically_significant
                else "NOT SIGNIFICANT: Cannot conclude effect exists with current data"
            )
        }

def demo():
    print("=== Statistical Module Demo ===\n")

    analyzer = StatisticalAnalyzer()
    exp_stats = ExperimentStatistics()

    np.random.seed(42)
    baseline = np.random.normal(0.002, 0.0005, 50)
    suppressed = np.random.normal(0.0015, 0.0006, 50)

    print("1. Baseline vs suppressed comparison:")
    result = exp_stats.compare_baseline_vs_suppressed(
        baseline.tolist(),
        suppressed.tolist()
    )

    print(f"   Baseline mean: {result['descriptive']['baseline']['mean']:.6f}")
    print(f"   Suppressed mean: {result['descriptive']['suppressed']['mean']:.6f}")
    print(f"   Change: {result['change']['percentage']:.2f}%")
    print(f"   p-value: {result['conclusion']['p_value']:.6f}")
    print(f"   Effect size (Cohen's d): {result['conclusion']['effect_size']:.4f} ({result['conclusion']['effect_interpretation']})")
    print(f"   Statistically significant: {result['conclusion']['is_significant']}")
    if result['power_analysis']['current_power'] is not None:
        print(f"   Current power: {result['power_analysis']['current_power']:.4f}")
    if result['power_analysis']['required_n_for_80_power'] is not None:
        print(f"   Required N for 80% power: {result['power_analysis']['required_n_for_80_power']}")

    print("\n2. Multi-condition comparison (suppression strengths):")
    conditions = {
        "baseline": baseline.tolist(),
        "strength_0.3": np.random.normal(0.0018, 0.0005, 50).tolist(),
        "strength_0.5": np.random.normal(0.0015, 0.0006, 50).tolist(),
        "strength_0.8": np.random.normal(0.0012, 0.0007, 50).tolist()
    }

    multi_result = exp_stats.compare_multiple_conditions(conditions)
    print(f"   ANOVA F: {multi_result['anova']['statistic']:.4f}")
    print(f"   ANOVA p-value: {multi_result['anova']['p_value']:.6f}")
    print(f"   Effect size (eta^2): {multi_result['anova']['effect_size']:.4f}")

    print("\n3. Effect detection analysis:")
    detection = exp_stats.analyze_effect_detection(
        baseline.tolist(),
        suppressed.tolist()
    )
    print(f"   Statistically significant: {detection['effect_detection']['statistically_significant']}")
    print(f"   Practically significant: {detection['effect_detection']['practically_significant']}")
    print(f"   Recommendation: {detection['recommendation']}")

if __name__ == "__main__":
    demo()
