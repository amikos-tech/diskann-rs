//! Recall calculation utilities for ANN benchmarks.

use std::collections::HashSet;

/// Statistics about recall across multiple queries.
#[derive(Debug, Clone)]
pub struct RecallStats {
    /// Mean recall (0.0 to 1.0)
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum recall
    pub min: f64,
    /// Maximum recall
    pub max: f64,
    /// Median recall
    pub median: f64,
    /// 5th percentile (95% have recall above this)
    pub p5: f64,
    /// Per-query recall values
    pub per_query: Vec<f64>,
    /// Number of queries
    pub num_queries: usize,
    /// K value used
    pub k: usize,
}

impl Default for RecallStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            p5: 0.0,
            per_query: Vec::new(),
            num_queries: 0,
            k: 0,
        }
    }
}

impl std::fmt::Display for RecallStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Recall@{}: {:.4} (±{:.4}) [min={:.4}, max={:.4}, median={:.4}] n={}",
            self.k, self.mean, self.std_dev, self.min, self.max, self.median, self.num_queries
        )
    }
}

/// Calculates recall@k for approximate search results against ground truth.
///
/// Recall@k = |approximate_results ∩ ground_truth_top_k| / k
///
/// # Arguments
/// * `results` - Approximate search results (k IDs per query)
/// * `ground_truth` - Ground truth nearest neighbors (at least k IDs per query)
/// * `k` - Number of neighbors to consider
pub fn calculate_recall(results: &[Vec<u32>], ground_truth: &[Vec<u32>], k: usize) -> RecallStats {
    if results.len() != ground_truth.len() {
        panic!("results and ground_truth must have the same length");
    }

    let num_queries = results.len();
    if num_queries == 0 {
        return RecallStats { k, ..Default::default() };
    }

    let per_query: Vec<f64> = results
        .iter()
        .zip(ground_truth.iter())
        .map(|(res, gt)| calculate_single_recall(res, gt, k))
        .collect();

    compute_stats(&per_query, k)
}

fn calculate_single_recall(result: &[u32], gt: &[u32], k: usize) -> f64 {
    // Build set of ground truth top-k
    let gt_k = k.min(gt.len());
    let gt_set: HashSet<u32> = gt.iter().take(gt_k).copied().collect();

    // Count matches
    let result_k = k.min(result.len());
    let matches = result.iter().take(result_k).filter(|id| gt_set.contains(id)).count();

    if k == 0 {
        1.0
    } else {
        matches as f64 / k as f64
    }
}

fn compute_stats(per_query: &[f64], k: usize) -> RecallStats {
    let n = per_query.len();
    if n == 0 {
        return RecallStats { k, ..Default::default() };
    }

    let mean = per_query.iter().sum::<f64>() / n as f64;

    let variance = per_query.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    let mut sorted = per_query.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    RecallStats {
        mean,
        std_dev,
        min: sorted[0],
        max: sorted[n - 1],
        median: percentile(&sorted, 0.5),
        p5: percentile(&sorted, 0.05),
        per_query: per_query.to_vec(),
        num_queries: n,
        k,
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let idx = p * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = (lower + 1).min(sorted.len() - 1);
    let frac = idx - lower as f64;

    sorted[lower] * (1.0 - frac) + sorted[upper] * frac
}

/// Calculates recall at multiple k values.
pub fn calculate_recall_at_k(
    results: &[Vec<u32>],
    ground_truth: &[Vec<u32>],
    ks: &[usize],
) -> Vec<RecallStats> {
    ks.iter()
        .map(|&k| calculate_recall(results, ground_truth, k))
        .collect()
}

/// Formats a recall report for multiple k values.
pub fn format_recall_report(stats: &[RecallStats]) -> String {
    let mut report = String::from("Recall Report\n=============\n\n");
    for s in stats {
        report.push_str(&format!("{}\n", s));
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_recall() {
        let gt = vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]];
        let results = vec![vec![0, 1, 2, 3, 4], vec![5, 6, 7, 8, 9]];

        let stats = calculate_recall(&results, &gt, 5);

        assert!((stats.mean - 1.0).abs() < 0.001);
        assert!((stats.min - 1.0).abs() < 0.001);
        assert_eq!(stats.num_queries, 2);
    }

    #[test]
    fn test_partial_recall() {
        let gt = vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]];
        let results = vec![vec![0, 1, 2, 3, 4, 100, 101, 102, 103, 104]];

        let stats = calculate_recall(&results, &gt, 10);

        assert!((stats.mean - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_zero_recall() {
        let gt = vec![vec![0, 1, 2, 3, 4]];
        let results = vec![vec![100, 101, 102, 103, 104]];

        let stats = calculate_recall(&results, &gt, 5);

        assert!(stats.mean.abs() < 0.001);
    }

    #[test]
    fn test_order_independent() {
        let gt = vec![vec![0, 1, 2, 3, 4]];
        let results = vec![vec![4, 3, 2, 1, 0]]; // Same elements, different order

        let stats = calculate_recall(&results, &gt, 5);

        assert!((stats.mean - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_recall_at_k() {
        let gt = vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]];
        let results = vec![vec![0, 1, 2, 100, 101, 102, 103, 104, 105, 106]];

        let stats = calculate_recall_at_k(&results, &gt, &[1, 3, 5, 10]);

        assert!((stats[0].mean - 1.0).abs() < 0.001);  // recall@1 = 1/1
        assert!((stats[1].mean - 1.0).abs() < 0.001);  // recall@3 = 3/3
        assert!((stats[2].mean - 0.6).abs() < 0.001); // recall@5 = 3/5
        assert!((stats[3].mean - 0.3).abs() < 0.001); // recall@10 = 3/10
    }
}
