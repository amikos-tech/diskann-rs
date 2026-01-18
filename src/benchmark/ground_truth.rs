//! Ground truth computation for ANN benchmarks.

use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Distance function type.
pub type DistanceFunc = fn(&[f32], &[f32]) -> f32;

/// Computes squared L2 (Euclidean) distance.
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

/// Computes L2 distance with square root.
pub fn l2_distance_sqrt(a: &[f32], b: &[f32]) -> f32 {
    l2_distance(a, b).sqrt()
}

/// Computes cosine distance (1 - cosine_similarity).
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Computes negative inner product (for maximum inner product search).
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>()
}

/// ID-distance pair for heap operations.
#[derive(Clone, Copy)]
struct IdDist {
    id: u32,
    dist: f32,
}

impl PartialEq for IdDist {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for IdDist {}

impl PartialOrd for IdDist {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IdDist {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: larger distance = higher priority
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}

/// Computes k nearest neighbors for each query using brute force.
///
/// This is the reference implementation for evaluating approximate nearest neighbor algorithms.
pub fn compute_ground_truth(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    dist_func: DistanceFunc,
) -> Vec<Vec<u32>> {
    queries
        .iter()
        .map(|query| compute_knn(base, query, k, dist_func))
        .collect()
}

/// Computes k nearest neighbors in parallel using rayon.
pub fn compute_ground_truth_parallel(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    dist_func: DistanceFunc,
) -> Vec<Vec<u32>> {
    queries
        .par_iter()
        .map(|query| compute_knn(base, query, k, dist_func))
        .collect()
}

/// Finds k nearest neighbors for a single query.
fn compute_knn(base: &[Vec<f32>], query: &[f32], k: usize, dist_func: DistanceFunc) -> Vec<u32> {
    let mut heap = BinaryHeap::with_capacity(k + 1);

    for (i, vec) in base.iter().enumerate() {
        let dist = dist_func(query, vec);

        if heap.len() < k {
            heap.push(IdDist { id: i as u32, dist });
        } else if let Some(top) = heap.peek() {
            if dist < top.dist {
                heap.pop();
                heap.push(IdDist { id: i as u32, dist });
            }
        }
    }

    // Extract results in order of increasing distance
    let mut result: Vec<_> = heap.into_iter().collect();
    result.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
    result.into_iter().map(|x| x.id).collect()
}

/// Computes k nearest neighbors with their distances.
pub fn compute_ground_truth_with_distances(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    dist_func: DistanceFunc,
) -> (Vec<Vec<u32>>, Vec<Vec<f32>>) {
    let results: Vec<_> = queries
        .par_iter()
        .map(|query| compute_knn_with_distances(base, query, k, dist_func))
        .collect();

    let ids = results.iter().map(|(ids, _)| ids.clone()).collect();
    let dists = results.iter().map(|(_, dists)| dists.clone()).collect();
    (ids, dists)
}

fn compute_knn_with_distances(
    base: &[Vec<f32>],
    query: &[f32],
    k: usize,
    dist_func: DistanceFunc,
) -> (Vec<u32>, Vec<f32>) {
    let mut heap = BinaryHeap::with_capacity(k + 1);

    for (i, vec) in base.iter().enumerate() {
        let dist = dist_func(query, vec);

        if heap.len() < k {
            heap.push(IdDist { id: i as u32, dist });
        } else if let Some(top) = heap.peek() {
            if dist < top.dist {
                heap.pop();
                heap.push(IdDist { id: i as u32, dist });
            }
        }
    }

    let mut result: Vec<_> = heap.into_iter().collect();
    result.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));

    let ids = result.iter().map(|x| x.id).collect();
    let dists = result.iter().map(|x| x.dist).collect();
    (ids, dists)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let dist = l2_distance(&a, &b);
        assert!((dist - 25.0).abs() < 0.001); // 3^2 + 4^2 = 25
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![2.0, 0.0, 0.0];
        let dist = cosine_distance(&a, &b);
        assert!(dist.abs() < 0.001); // Same direction = 0

        let c = vec![0.0, 1.0, 0.0];
        let dist2 = cosine_distance(&a, &c);
        assert!((dist2 - 1.0).abs() < 0.001); // Orthogonal = 1
    }

    #[test]
    fn test_compute_ground_truth() {
        let base = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![3.0, 0.0],
            vec![10.0, 10.0],
        ];
        let queries = vec![vec![0.0, 0.0]];

        let gt = compute_ground_truth(&base, &queries, 3, l2_distance);

        assert_eq!(gt.len(), 1);
        assert_eq!(gt[0].len(), 3);
        assert_eq!(gt[0][0], 0); // Nearest is itself
    }

    #[test]
    fn test_ground_truth_ordering() {
        let base = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        ];
        let queries = vec![vec![0.0, 0.0]];

        let gt = compute_ground_truth(&base, &queries, 3, l2_distance);

        // Should be ordered by increasing distance
        assert_eq!(gt[0], vec![0, 1, 2]);
    }
}
