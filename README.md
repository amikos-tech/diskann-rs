# ami-diskann

A Rust vector search library built on the [Vamana graph algorithm](https://proceedings.neurips.cc/paper_files/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html) (DiskANN). Memory-mapped storage, scalar quantization with per-dimension scaling, zero-copy index loading, and filtered search.

Forked from [lukaesch/diskann-rs](https://github.com/lukaesch/diskann-rs) and developed independently by [Amikos Tech](https://github.com/amikos-tech).

## Features

| Feature | Description |
|---------|-------------|
| **Scalar Quantization** | F16 (2x compression) and Int8 with per-dimension scaling (4x compression) |
| **VectorQuantizer Trait** | Composable quantization interface for F16, Int8, and Product Quantization |
| **Zero-Copy Loading** | `DiskANNRef` loads indexes from byte slices without copying data |
| **Filtered Search** | Query with metadata predicates (equality, range, set membership, AND/OR) |
| **Incremental Updates** | Add/delete vectors without full rebuild; compact when needed |
| **Product Quantization** | Up to 64x compression with trained codebooks |
| **SIMD Acceleration** | Optimized distance computation (AVX2, SSE4.1, NEON) |
| **Memory-Mapped I/O** | Single-file storage, OS pages in only what's accessed |

## Quick Start

```rust
use anndists::dist::DistL2;
use ami_diskann::{DiskANN, DiskAnnParams, QuantizationType};

// Build an F32 index
let vectors: Vec<Vec<f32>> = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
let index = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, "index.db")?;

// Search
let query = vec![0.1, 0.2, 0.4];
let neighbors = index.search(&query, 10, 256);

// Build with Int8 quantization (4x smaller on disk)
let index = DiskANN::<DistL2>::build_index_with_params(
    &vectors, DistL2 {}, "index_int8.db",
    DiskAnnParams {
        quantization: QuantizationType::Int8,
        ..Default::default()
    },
)?;
```

## Scalar Quantization

ami-diskann supports three storage formats:

| Format | Bytes/element | Compression | Use case |
|--------|--------------|-------------|----------|
| **F32** | 4 | 1x | Maximum precision |
| **F16** | 2 | 2x | Good precision, half the storage |
| **Int8** | 1 | 4x | Best compression, per-dimension scaling preserves precision |

### Per-Dimension Int8 Scaling

Unlike per-vector scaling (one min/max for all dimensions), ami-diskann uses **per-dimension scaling** -- each dimension gets its own [min, max] range mapped to [0, 255]. This matters when dimensions have different distributions, which is common in real embeddings.

```
Example: vector with dim 0 in [0, 1] and dim 1 in [0, 1000]

Per-vector scaling:  dim 0 error ~= 1000/255 ~= 4.0   (unusable)
Per-dimension scaling: dim 0 error ~= 1/255 ~= 0.004   (precise)
```

The scales are stored once per dimension (not per vector), so overhead is minimal: 8 bytes * dim (e.g. 1 KB for 128-dim).

## Zero-Copy Index Loading

`DiskANNRef` borrows from an external byte slice, enabling zero-copy loading from embedded data, network payloads, or pre-loaded files:

```rust
use ami_diskann::{DiskANN, DiskANNRef};
use anndists::dist::DistL2;

// Serialize an index to bytes
let index = DiskANN::<DistL2>::build_index_default(&vectors, DistL2 {}, "index.db")?;
let bytes = index.to_bytes()?;

// Load without copying (bytes must outlive the index)
let loaded = DiskANNRef::<DistL2>::from_bytes(&bytes, DistL2 {})?;
let results = loaded.search(&query, 10, 64);

// Bulk extract all vectors (dequantized to f32)
let all_vectors = loaded.expand_vectors();
```

## VectorQuantizer Trait

A composable interface for quantization strategies, usable independently of the index:

```rust
use ami_diskann::{VectorQuantizer, F16Quantizer, Int8Quantizer};

// F16: stateless, just converts f32 <-> f16
let f16q = F16Quantizer;
let codes = f16q.encode(&[1.0, 2.0, 3.0]);
let decoded = f16q.decode(&codes);

// Int8: trained on data, per-dimension scales
let training_data = vec![vec![0.0, 0.0], vec![1.0, 1000.0]];
let int8q = Int8Quantizer::train(&training_data, 2);
let codes = int8q.encode(&[0.5, 500.0]);
let dist = int8q.asymmetric_distance(&query, &codes);

// Also implemented for ProductQuantizer
use ami_diskann::pq::{ProductQuantizer, PQConfig};
let pq = ProductQuantizer::train(&vectors, PQConfig::default())?;
let codes = VectorQuantizer::encode(&pq, &vectors[0]);
```

## Filtered Search

Query with metadata predicates. The graph is still explored fully, but only matching candidates enter the result set:

```rust
use ami_diskann::{FilteredDiskANN, Filter};
use anndists::dist::DistL2;

let vectors: Vec<Vec<f32>> = /* ... */;
let labels: Vec<Vec<u64>> = (0..1000).map(|i| vec![i % 10, i]).collect();

let index = FilteredDiskANN::<DistL2>::build(&vectors, &labels, "filtered.db")?;

// Simple equality
let results = index.search_filtered(&query, 10, 128, &Filter::label_eq(0, 5));

// Compound filters
let filter = Filter::and(vec![
    Filter::label_eq(0, 5),
    Filter::label_range(1, 10, 100),
]);
let results = index.search_filtered(&query, 10, 128, &filter);
```

## Incremental Updates

Add and delete vectors without rebuilding the full index:

```rust
use ami_diskann::IncrementalDiskANN;
use anndists::dist::DistL2;

let index = IncrementalDiskANN::<DistL2>::build_default(&vectors, "index.db")?;

// Add vectors (appended to a delta layer with its own mini-graph)
index.add_vectors(&[vec![1.0; 128]])?;

// Delete vectors (instant tombstoning)
index.delete_vectors(&[0, 1, 2])?;

// Compact when the delta layer gets large
if index.should_compact() {
    index.compact("index_v2.db")?;
}
```

## Product Quantization

For extreme compression (up to 64x), train a codebook on your data:

```rust
use ami_diskann::pq::{ProductQuantizer, PQConfig};

let pq = ProductQuantizer::train(&vectors, PQConfig {
    num_subspaces: 8,
    num_centroids: 256,
    ..Default::default()
})?;

// 128-dim f32 (512 bytes) -> 8 bytes
let codes = pq.encode(&vectors[0]);
let table = pq.create_distance_table(&query);
let dist = pq.distance_with_table(&table, &codes);
```

## SIMD Acceleration

Drop-in SIMD-optimized distance metrics:

```rust
use ami_diskann::{DiskANN, SimdL2, SimdCosine, simd_info};

println!("{}", simd_info()); // "SIMD: NEON" or "SIMD: AVX2, SSE4.1"

let index = DiskANN::<SimdL2>::build_index_default(&vectors, SimdL2, "index.db")?;
```

## Parameters

### Build Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_degree` | 64 | Max neighbors per node (32-64 typical) |
| `build_beam_width` | 128 | Construction search width (128-256) |
| `alpha` | 1.2 | Pruning diversity factor (1.2-2.0) |
| `quantization` | F32 | Storage format: `F32`, `F16`, or `Int8` |

### Search Parameters

| Parameter | Typical | Trade-off |
|-----------|---------|-----------|
| `beam_width` | 128-512 | Higher = better recall, slower |
| `k` | 10-100 | Number of neighbors to return |

## Architecture

### File Layout

```
[ metadata_len:u64 ][ metadata (bincode) ][ padding to 1 MiB ]
[ scales (dim * 8 bytes, Int8 only) ]
[ vectors (n * bytes_per_vector) ]
[ adjacency (n * max_degree * 4 bytes) ]
```

Metadata is versioned (v2 format). Legacy v1 (F32-only) indexes are auto-upgraded on load. Old per-vector Int8 indexes return a clear error asking the user to rebuild.

### Module Structure

```
ami_diskann
  lib.rs         -- DiskANN<D>, DiskANNRef<'a, D>, build/search/quantization
  sq.rs          -- VectorQuantizer trait, F16Quantizer, Int8Quantizer
  pq.rs          -- ProductQuantizer (also implements VectorQuantizer)
  filtered.rs    -- FilteredDiskANN with metadata predicates
  incremental.rs -- IncrementalDiskANN with delta layer
  simd.rs        -- SIMD-optimized distance functions
  benchmark/     -- Dataset loaders, recall metrics, synthetic data
```

## Building and Testing

```bash
cargo build --release
cargo test
cargo clippy
cargo bench --bench benchmark
```

## Lineage

This project is a fork of [diskann-rs](https://github.com/lukaesch/diskann-rs) by Lukas Schmyrczyk and Jianshu Zhao. Key additions since the fork:

- Zero-copy index loading (`DiskANNRef`, `to_bytes()`)
- F16 scalar quantization
- Int8 scalar quantization with per-dimension scaling
- `VectorQuantizer` trait for composable quantization
- `expand_vectors()` for bulk vector extraction
- Fixed filtered search for quantized indexes (was hardcoded to F32)

## License

MIT License -- see [LICENSE](LICENSE) for details.

## References

- [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node (NeurIPS 2019)](https://proceedings.neurips.cc/paper_files/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
- [Microsoft DiskANN](https://github.com/Microsoft/DiskANN)
- [Original diskann-rs](https://github.com/lukaesch/diskann-rs)
