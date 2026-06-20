# HPC MultiRound Reclustering - Complete Guide

Distributed SLURM-based BitBirch clustering for massive molecular libraries.

## Quick Start

```bash
# 1. Generate initial round jobs
iChem initial-round *.smi \
  --out-dir ./clustering \
  --files-per-job 10 \
  --verbose

# 2. Submit initial jobs
./clustering/submit_initial_jobs.sh

# 3. Wait for completion, then generate midsection jobs
iChem midsection-round \
  --output-dir ./clustering \
  --round-idx 2 \
  --bin-size 5 \
  --verbose

# 4. Submit midsection jobs
./clustering/submit_midsection_round_2_jobs.sh

# 5. Generate final job
iChem final-round \
  --output-dir ./clustering \
  --prev-round-idx 2 \
  --verbose

# 6. Submit final job
./clustering/submit_final_round_job.sh

# Results: clusters.pkl, cluster-centroids-packed.pkl
```

---

## Architecture Overview

The refactored HPC pipeline splits clustering into independent phases, each with its own CLI command and worker modules:

### Phase 1: Initial Round (Fingerprint + Clustering)
- **CLI Command**: `iChem initial-round`
- **Generator Module**: `_hpc_initial_submit.py`
- **Worker Module**: `_hpc_initial.py`
- **Output**: `submit_initial_jobs.sh`
- **Each Job**:
  - Loads SMILES from file chunk
  - Generates fingerprints (ECFP4, configurable)
  - Performs BitBirch clustering
  - Saves buffers and indices: `round-1-bufs*.npy`, `round-1-idxs*.pkl`
- **Memory**: 16GB, 1 CPU per job
- **Runs**: N jobs in parallel (one per file batch)

### Phase 2: Midsection Rounds (Merge & Cluster)
- **CLI Command**: `iChem midsection-round`
- **Generator Module**: `_hpc_midsection_submit.py`
- **Worker Module**: `_hpc_midsection.py`
- **Output**: `submit_midsection_round_{N}_jobs.sh`
- **Each Job**:
  - Loads buffers from previous round file pairs
  - Merges BitFeatures from multiple batches
  - Performs reclustering (optional)
  - Saves new buffers: `round-{N}-bufs*.npy`, `round-{N}-idxs*.pkl`
- **Memory**: 48GB, 3 CPUs per job
- **Runs**: M jobs in parallel (one per batch, batch size = `--bin-size`)
- **Repeatable**: Chain multiple midsection rounds

### Phase 3: Final Round (Consolidation)
- **CLI Command**: `iChem final-round`
- **Generator Module**: `_hpc_final_submit.py`
- **Worker Module**: `_hpc_final.py`
- **Output**: `submit_final_round_job.sh`
- **Single Job**:
  - Loads ALL buffers from previous round
  - Performs final tree consolidation
  - Saves output:
    - `clusters.pkl` - Molecule → Cluster ID mapping
    - `cluster-centroids-packed.pkl` - Cluster centroids (optional)
    - `bitbirch.pkl` - Full tree object (optional)
- **Memory**: 96GB, 6 CPUs
- **Runs**: 1 job

---

## Configuration

All defaults centralized in `iChem/bitbirch/_config.py`. Override via CLI flags.

### Global Defaults
```python
THRESHOLD = 0.3                              # BitBirch threshold
BRANCHING_FACTOR = 10_000                    # BitBirch branching factor
MERGE_CRITERION = "diameter"                 # "diameter" or "radius"
FINGERPRINT_TYPE = "ECFP4"                   # Fingerprint type RDKIT, MACCS, AP
N_BITS = 2048                                # Fingerprint bits
```

### Reclustering
```python
RECLUSTERING_ITERATIONS_INITIAL = 3
RECLUSTERING_ITERATIONS_MIDSECTION = 0
RECLUSTERING_ITERATIONS_FINAL = 0
RECLUSTERING_EXTRA_THRESHOLD = 0.025
```

### SLURM Resources
```python
SLURM_MEM_INITIAL = "16G"                    # Initial round memory
SLURM_CPUS_INITIAL = 1
SLURM_MEM_MIDSECTION = "48G"                 # Midsection memory
SLURM_CPUS_MIDSECTION = 3
SLURM_MEM_FINAL = "96G"                      # Final round memory
SLURM_CPUS_FINAL = 6
SLURM_TIME = "24:00:00"                      # All jobs
SLURM_PARTITION = ""                         # Empty = default partition (CPU)
```

**Note**: All jobs are CPU-only. Leave `--slurm-partition` empty (default) or specify your CPU partition.

### HPC Parameters
```python
FILES_PER_JOB = 10                           # Files per initial job
BIN_SIZE = 5                                 # Buffer pairs per midsection job
```

---

## CLI Commands

### `iChem initial-round`

Generate and view initial round job submission script. Generates fingerprints and performs initial clustering on SMILES files in parallel.

```bash
iChem initial-round \
  FILE1.smi FILE2.smi FILE3.smi \
  --out-dir ./clustering \
  --files-per-job 10 \
  --threshold 0.3 \
  --branching-factor 10000 \
  --merge-criterion diameter \
  --fp-type ECFP4 \
  --n-bits 2048 \
  --reclustering-iterations 3 \
  --reclustering-extra-threshold 0.025 \
  --slurm-mem 16G \
  --slurm-cpus 1 \
  --slurm-time 24:00:00 \
  --result-base-dir ./results \
  --verbose
```

**Key Inputs**:
- **FILES** (positional): One or more `.smi` or `.smi.gz` files to cluster. Order does not matter; internal numbering is automatic.
- **`--out-dir`** (required): Output directory where job scripts and logs are stored. Results go here after submission.
- **`--files-per-job`**: How many input files to process per SLURM job. Each job gets N files and processes all molecules in those files.
- **`--fp-type`**: Fingerprint type (`ECFP4`, `RDKIT`, `MACCS`, `AP`). ECFP4 recommended for diversity.
- **`--n-bits`**: Fingerprint length (default 2048). Larger = more detail but slower. 1024-4096 typical.
- **`--threshold`**: BitBirch clustering threshold (0.0-1.0). Lower = more clusters (finer). Start with 0.3.
- **`--branching-factor`**: BitBirch tree branching factor (default 10000). Rarely needs tuning.
- **`--merge-criterion`**: How to merge clusters (`diameter` or `radius`). `diameter` is standard.
- **`--reclustering-iterations`**: How many reclustering passes (default 3 for initial). Increases quality but time.
- **`--slurm-mem`/`--cpus`/`--time`**: SLURM resource allocation per job.

**Output**: `./clustering/submit_initial_jobs.sh` (or multiple scripts if > max jobs per script)
- Contains multiple `sbatch` commands (one per batch)
- User runs: `bash ./clustering/submit_initial_jobs.sh`

**Nuances**:
- Files are processed in **global molecule index order**: first file molecules get indices 0-N, second file N-(N+M), etc. This is preserved across all rounds.
- Each job produces two files per fingerprint type:
  - `round-1-bufs.label-{LABEL}-{DTYPE}.npy`: Fingerprint buffers (NumPy array)
  - `round-1-idxs.label-{LABEL}-{DTYPE}.pkl`: Molecule indices (pickle file)
- Both files are needed for midsection. If one is missing, the round cannot proceed.
- Fingerprints are saved streaming to disk (not materialized in memory), so even 1B+ molecules can fit if `--files-per-job` is reasonable.

### `iChem midsection-round`

Generate midsection round job submission script. Merges and reclusters batches from previous round.

```bash
iChem midsection-round \
  --output-dir ./clustering \
  --round-idx 2 \
  --bin-size 5 \
  --threshold 0.3 \
  --branching-factor 10000 \
  --merge-criterion diameter \
  --reclustering-iterations 0 \
  --reclustering-extra-threshold 0.025 \
  --slurm-mem 48G \
  --slurm-cpus 3 \
  --slurm-time 24:00:00 \
  --verbose
```

**Key Inputs**:
- **`--output-dir`** (required): Same directory as initial round (where `round-1-*` files exist).
- **`--round-idx`** (required): Current round number. Reads from `round-{round-idx-1}-*`, writes to `round-{round-idx}-*`.
  - First midsection: `--round-idx 2` (reads `round-1-*`, writes `round-2-*`)
  - Second midsection: `--round-idx 3` (reads `round-2-*`, writes `round-3-*`)
  - And so on...
- **`--bin-size`**: Number of buffer/index **pairs** per midsection job. If 10 initial jobs → 10 file pairs; `--bin-size 5` → 2 midsection jobs (5 pairs each).
- **`--threshold`/`--branching-factor`**: Can differ from initial round. Often same, but you can recluster with different thresholds.
- **`--reclustering-iterations`**: Typically 0 for midsection (reclustering done in initial). Set > 0 only if refining clusters.

**Output**: `./clustering/submit_midsection_round_2_jobs.sh` (multiple scripts if needed)

**Automatically Discovers**:
- Previous round files: `round-{round_idx-1}-bufs*.npy`, `round-{round_idx-1}-idxs*.pkl`
- Auto-detects all file pairs and chunks them by `--bin-size`
- Creates one job per batch

**Nuances**:
- **File pair matching**: The command scans for all `round-{prev}-bufs*.npy` and `round-{prev}-idxs*.pkl` files. Counts must match exactly. If one type is missing, it fails.
- **Batch sorting**: Within each batch, file pairs are sorted by fingerprint dtype (uint8, uint16, etc.), with largest first. This improves cluster quality.
- **Memory per job**: Roughly proportional to `bin-size × average-molecules-per-pair`. If OOM, reduce `--bin-size`.
- **Round files cleaned**: After midsection job completes, it **deletes the files it processed** from the previous round. This saves disk space but makes restart tricky (see recovery).
- You can chain multiple midsection rounds if you want to further pyramid the tree (e.g., `--round-idx 3`, then `--round-idx 4`, etc.).

### `iChem final-round`

Generate final round job submission script. Merges all results into final clusters and optionally saves outputs.

```bash
iChem final-round \
  --output-dir ./clustering \
  --prev-round-idx 2 \
  --threshold 0.3 \
  --branching-factor 10000 \
  --merge-criterion diameter \
  --reclustering-iterations 0 \
  --reclustering-extra-threshold 0.025 \
  --save-centroids \
  --save-tree \
  --save-npy \
  --slurm-mem 96G \
  --slurm-cpus 6 \
  --slurm-time 24:00:00 \
  --verbose
```

**Key Inputs**:
- **`--output-dir`** (required): Same as previous rounds.
- **`--prev-round-idx`** (required): Last round index. Reads from `round-{prev_round_idx}-*` files.
  - If last midsection used `--round-idx 2`, use `--prev-round-idx 2`
  - If last midsection used `--round-idx 3`, use `--prev-round-idx 3`
- **`--save-centroids`** (default True): Save centroid fingerprints and cluster → molecule mapping.
- **`--save-tree`**: Save the full BitBirch tree object. Useful for re-analysis but large (~GB for 100M molecules).
- **`--save-npy`**: Save clusters as separate `.npy` files (one per cluster). Disk-intensive; use only if needed for downstream analysis.

**Output**: `./clustering/submit_final_round_job.sh`

**Automatically Discovers**:
- All files: `round-{prev_round_idx}-bufs*.npy`, `round-{prev_round_idx}-idxs*.pkl`
- Merges them into one final tree

**Nuances**:
- **Final outputs**:
  - `clusters.pkl`: List of lists; `clusters[i]` = molecule IDs in cluster i
  - `cluster-centroids-packed.pkl`: Packed centroids (if `--save-centroids`)
  - `bitbirch.pkl`: Full tree object (if `--save-tree`)
  - `clusters/*.npy`: Individual cluster files (if `--save-npy`)
- **Intermediate cleanup**: Final round **automatically deletes all `round-N-*` files** after merging. These can be very large; deletion saves disk.
- **Memory**: This is the largest job. 96GB should handle 100M+ molecules, but for billions, may need more.
- **Single job**: Only one final job runs (no parallelism). Merging is inherently sequential.

---

## Advanced Usage Nuances

### File Naming and Pairing

All intermediate files follow a strict naming convention:

```
round-{N}-bufs.label-{LABEL}-{DTYPE}.npy     # Fingerprint buffers
round-{N}-idxs.label-{LABEL}-{DTYPE}.pkl     # Molecule indices
```

- **{N}**: Round number (1 for initial, 2+ for midsection, etc.)
- **{LABEL}**: Batch label from initial round (e.g., "00", "01", ..., "09")
- **{DTYPE}**: Fingerprint dtype (e.g., "uint8", "uint16", "uint32"). Auto-determined by fingerprint type and bits.

**Critical**: Each job expects exactly one `.npy` file paired with one `.pkl` file for each label-dtype combination. If this pairing is broken (e.g., one file deleted, one corrupted), the round cannot proceed.

### Global Molecule Indexing

Every molecule in your dataset receives a **global index** from 0 to (total_molecules - 1). This index is preserved across all rounds:

1. **Initial round**: Molecules in file 0 get indices 0-N₀, file 1 gets N₀-(N₀+N₁), etc.
2. **Midsection/final rounds**: Indices are read from pickle files and preserved.
3. **Output**: `clusters.pkl[i]` contains molecule IDs, which are these global indices.

This means you can always map a molecule ID back to its original input file:
```python
import pickle
with open("clusters.pkl", "rb") as f:
    cluster_mol_ids = pickle.load(f)

# cluster_mol_ids[0] is a list of molecule IDs (global indices) in cluster 0
```

### Streaming Fingerprint Generation

Initial round uses **streaming save** to avoid materializing all fingerprints in memory:

1. For each input file, fingerprints are generated in chunks.
2. Chunks are written directly to `.npy` in NumPy format (not standard Pickle).
3. This allows processing 1B+ molecules on a single node without OOM.

**Implication**: Fingerprint `.npy` files are NOT standard NumPy files; they're NumPy-formatted with a custom header. Load with `np.load()`, not raw file I/O.

### Batch Sorting in Midsection

When midsection jobs process file pairs, they're automatically sorted within each batch:

```python
# Pairs are sorted by fingerprint dtype (largest uint first)
# E.g., if a batch has uint32, uint16, uint8 pairs
# They're processed in order: uint32 → uint16 → uint8
```

This improves cluster quality by processing finer fingerprints first.

### Automatic File Cleanup

- **Midsection jobs**: After completion, delete the file pairs they processed (from previous round).
- **Final job**: After completion, delete ALL `round-{N}-*` files.

This saves disk but makes **restart tricky** (see Restart After Failure).

### Configuration Inheritance

Each round can have different parameters:

```bash
# Initial: high quality
iChem initial-round ... --threshold 0.3 --reclustering-iterations 3

# Midsection: faster (different threshold)
iChem midsection-round ... --threshold 0.4 --reclustering-iterations 0

# Final: same as midsection
iChem final-round ... --threshold 0.4
```

Parameters are **not inherited**; you must re-specify them each round if desired.

### Multiple Midsection Rounds

You can chain midsection rounds to pyramid the clustering tree:

```bash
# After initial round completes:
iChem midsection-round --output-dir ./clustering --round-idx 2 ...
./clustering/submit_midsection_round_2_jobs.sh
# wait for completion...

# Add another midsection layer:
iChem midsection-round --output-dir ./clustering --round-idx 3 ...
./clustering/submit_midsection_round_3_jobs.sh
# wait for completion...

# Final round reads from round-3-*:
iChem final-round --output-dir ./clustering --prev-round-idx 3 ...
```

This progressively merges tree layers, useful for very large datasets.

### SLURM Partition Selection

- **`--slurm-partition ""`** (empty, default): Uses HPC default partition (usually CPU queue).
- **`--slurm-partition "gpu"`**: Would request GPU partition (but jobs are CPU-only, so unused).
- Leave empty unless you have specific queue requirements.

---

## Complete Workflow Example

### Setup

```bash
# Assume you have 100 SMILES files, each with ~1M molecules
ls data/*.smi | wc -l          # 100 files
wc -l data/*.smi | tail -1    # ~100M total lines
```

### Step 1: Generate Initial Jobs

```bash
iChem initial-round \
  data/*.smi \
  --out-dir ./clustering \
  --files-per-job 10 \
  --threshold 0.3 \
  --branching-factor 10000 \
  --reclustering-iterations 3 \
  --verbose
```

Output:
```
[Initial Round Setup]
  Input files: 100
  Files per job: 10
  Output directory: ./clustering
  Created 10 jobs
    Job 0: 10 files, molecules 0-9999999
    Job 1: 10 files, molecules 10000000-19999999
    ...
    Job 9: 10 files, molecules 90000000-99999999

✓ Generated submission script: ./clustering/submit_initial_jobs.sh
  Run with: ./submit_initial_jobs.sh
```

### Step 2: Submit Initial Jobs

```bash
cd clustering
./submit_initial_jobs.sh
```

Output:
```
Submitted batch job 12345
Submitted batch job 12346
Submitted batch job 12347
Submitted batch job 12348
Submitted batch job 12349
Submitted batch job 12350
Submitted batch job 12351
Submitted batch job 12352
Submitted batch job 12353
Submitted batch job 12354
```

### Step 3: Monitor

```bash
watch -n 5 squeue -u $(whoami)
# or
tail -f logs/initial_*.out
```

### Step 4: Generate Midsection Jobs

After all initial jobs complete:

```bash
iChem midsection-round \
  --output-dir ./clustering \
  --round-idx 2 \
  --bin-size 5 \
  --reclustering-iterations 0 \
  --verbose
```

Output:
```
[Midsection Round 2 Setup]
  Output directory: ./clustering
  Bin size: 5
  Reading from: round-1-* files
  Found 10 buffer/index file pairs
  Created 2 batches
    Batch 0: 5 pairs
    Batch 1: 5 pairs

✓ Generated submission script: ./clustering/submit_midsection_round_2_jobs.sh
  Run with: ./submit_midsection_round_2_jobs.sh
```

### Step 5: Submit Midsection Jobs

```bash
./submit_midsection_round_2_jobs.sh
```

### Step 6: Generate Final Job

```bash
iChem final-round \
  --output-dir ./clustering \
  --prev-round-idx 2 \
  --save-centroids \
  --verbose
```

Output:
```
[Final Round Setup]
  Output directory: ./clustering
  Reading from: round-2-* files
  Found 2 buffer/index file pairs to merge

✓ Generated submission script: ./clustering/submit_final_round_job.sh
  Run with: ./submit_final_round_job.sh
```

### Step 7: Submit Final Job

```bash
./submit_final_round_job.sh
```

### Step 8: Results

```bash
ls -lh clustering/
# clusters.pkl                      # Final output
# cluster-centroids-packed.pkl      # Centroids
# bitbirch.pkl                      # Tree (if requested)
# round-*.npy/pkl                   # Intermediate (optional cleanup)
# logs/                             # All job logs
# submit_initial_jobs.sh            # Generated scripts
# submit_midsection_round_2_jobs.sh
# submit_final_round_job.sh
```

---

## Output Directory Structure

After all phases complete:

```
clustering/
├── round-1-bufs.label-*.npy           # Initial round buffers
├── round-1-idxs.label-*.pkl           # Initial round indices
├── round-2-bufs.label-*.npy           # Midsection round buffers (if kept)
├── round-2-idxs.label-*.pkl           # Midsection round indices (if kept)
├── clusters.pkl                       # FINAL: Molecule ID → Cluster ID
├── cluster-centroids-packed.pkl       # FINAL: Cluster centroids
├── bitbirch.pkl                       # FINAL: Tree object (optional)
├── logs/
│   ├── initial_00.out/err
│   ├── initial_01.out/err
│   ├── ...
│   ├── initial_09.out/err
│   ├── midsection_2_00.out/err
│   ├── midsection_2_01.out/err
│   ├── final_round.out/err
├── submit_initial_jobs.sh             # Generated submission scripts
├── submit_midsection_round_2_jobs.sh
└── submit_final_round_job.sh
```

---

## Key Parameters Explained

### `--files-per-job` (Initial Round Only)

**Definition**: Number of .smi/.smi.gz files to process per initial job.

**Example**:
- 100 .smi files with 1M molecules each
- `--files-per-job 10` → 10 jobs
- Each job processes 10 files = 10M molecules

**Recommendation**:
- Aim for **10-100M molecules per job**
- If files are 1M each → use 10-100
- If files are 10M each → use 1-10
- If files are 100M each → use 1

**Tuning by dataset**:
- **Small dataset** (< 1M total): `--files-per-job 1` (avoid excessive parallelism overhead)
- **Medium** (1M-100M): `--files-per-job 5-20`
- **Large** (100M-1B): `--files-per-job 1-5`
- **Massive** (1B+): `--files-per-job 1`, use multiple midsection rounds

### `--bin-size` (Midsection Round Only)

**Definition**: Number of buffer/index file pairs per midsection batch.

**Example**:
- 10 initial jobs → 10 buffer/index pairs
- `--bin-size 5` → 2 midsection jobs
- Job 0 processes pairs 0-4
- Job 1 processes pairs 5-9

**Recommendation**:
- Default 5 is good for most cases
- Larger (e.g., 10) = fewer jobs but longer per job
- Smaller (e.g., 2) = more jobs but faster per job
- **Rule of thumb**: Aim for 3-10 jobs per midsection round for parallelism

**Tuning by scale**:
- **10-20 initial jobs**: `--bin-size 5-10`
- **20-100 initial jobs**: `--bin-size 5-20`
- **100+ initial jobs**: `--bin-size 10-50`

### `--round-idx` (Midsection Round)

**Definition**: Current round index (reads from `round-{round_idx-1}-*` files).

**Example**:
- After initial round completes, files are `round-1-*`
- First midsection: `--round-idx 2` (reads round-1-*, writes round-2-*)
- Second midsection: `--round-idx 3` (reads round-2-*, writes round-3-*)

**Critical**: Must increment by 1 each time.

### `--prev-round-idx` (Final Round)

**Definition**: Previous round index (reads from `round-{prev_round_idx}-*` files).

**Example**:
- If last midsection wrote `round-2-*`, use `--prev-round-idx 2`

---

## Parameter Tuning Guide

### Choosing `--threshold`

**Range**: 0.0 (all unique clusters) to 1.0 (all same cluster)

**Effect**:
- **Lower threshold** (e.g., 0.1): Many clusters, finer granularity, slower
- **Higher threshold** (e.g., 0.7): Fewer clusters, coarser grouping, faster

**Typical values**:
- Chemically similar molecules: 0.2-0.4
- Diverse library: 0.3-0.5
- Very broad groups: 0.6-0.8

**How to choose**:
1. Test on a small sample (e.g., 1000 molecules)
2. Generate fingerprints with chosen threshold
3. Inspect cluster sizes and composition
4. Adjust threshold up (larger clusters) or down (smaller clusters) accordingly

### Choosing Fingerprint Type and Bits

| Type | Bits | Speed | Diversity | Use Case |
|------|------|-------|-----------|----------|
| ECFP4 | 2048 | Fast | High | General purpose (default) |
| RDKIT | 2048 | Medium | Medium | Alternative fingerprint |
| MACCS | 167 | Very Fast | Low | When speed critical |
| AP | 2048 | Medium | Medium | Atom pair-based |

**Recommendation**:
- Start with `--fp-type ECFP4 --n-bits 2048`
- If too slow, try MACCS (167 bits) for quick iteration
- If need more detail, increase to 4096 bits

### Choosing Reclustering Iterations

| Round | Iterations | Reason |
|-------|-----------|--------|
| Initial | 1-3 | Refine initial clusters, quality matters |
| Midsection | 0 | Merging already-clustered data, usually skipped |
| Final | 0 | Single final pass, reclustering adds little value |

**Exception**: If you notice poor final cluster quality, add reclustering:
```bash
iChem final-round ... --reclustering-iterations 1
```

### Memory vs. Speed Tradeoff

Larger jobs = more memory, potentially slower:

```bash
# Fast (many jobs, low memory each)
iChem initial-round ... --files-per-job 1 --slurm-mem 8G

# Balanced (medium jobs)
iChem initial-round ... --files-per-job 10 --slurm-mem 16G

# Slow (few jobs, high memory each)
iChem initial-round ... --files-per-job 100 --slurm-mem 32G
```

**Trade-off analysis**:
- More jobs = higher SLURM overhead but parallelizes faster
- Fewer jobs = lower overhead but longer per job
- **Optimal**: 10-100 initial jobs for most clusters

---



---

## Troubleshooting

### Initial Round Jobs Fail

**Check logs**:
```bash
cat clustering/logs/initial_00.err
cat clustering/logs/initial_00.out
```

**Common issues**:
- **SMILES parsing errors**: Some molecules invalid
  - Solution: Review error message, reduce `--n-bits` or adjust fingerprinting
- **Memory exceeded**: 16GB not enough for batch size
  - Solution: Decrease `--files-per-job`
- **File not found**: Paths in script are wrong
  - Solution: Regenerate with absolute paths

### Midsection Jobs Fail

**Check logs**:
```bash
cat clustering/logs/midsection_2_00.err
```

**Common issues**:
- **No files found**: Previous round didn't complete
  - Solution: Verify `round-1-*.npy` and `round-1-*.pkl` exist
- **Mismatched buffer/index counts**: Some jobs failed
  - Solution: Check if all initial jobs completed successfully

### Final Job Fails

**Check logs**:
```bash
cat clustering/logs/final_round.err
```

**Common issues**:
- **Insufficient memory**: Tree consolidation exceeded 96GB
  - Solution: Increase `--slurm-mem`
- **Timeout**: Job took >24 hours
  - Solution: Increase `--slurm-time`

### Restart After Failure

**Important**: Midsection/final jobs **delete the files they process**. If a job fails partway through, some intermediate files may be missing.

**Diagnosis**:
```bash
# Check what exists:
ls clustering/round-2-bufs*.npy | wc -l    # How many round-2 files?
ls clustering/round-1-bufs*.npy | wc -l    # How many round-1 files remain?

# See job errors:
tail -100 clustering/logs/midsection_2_00.err
```

**Scenario 1: Midsection job failed, some pairs deleted, some round-1 files remain**
- Some `round-1-*` files may be missing (deleted by partially-successful jobs)
- **Solution A** (if source data available): Regenerate from initial:
  ```bash
  iChem initial-round ... --out-dir ./clustering
  ./clustering/submit_initial_jobs.sh
  # Then retry midsection after initial completes
  ```
- **Solution B** (if only some batches failed): Manually retry failed batches by regenerating and resubmitting midsection script

**Scenario 2: Final job failed, round files partially cleaned**
- If final job crashed, you may have missing round-N files
- **Solution**: Restore from backup (create before running final):
  ```bash
  cp -r clustering clustering.backup  # Create backup before final
  # ... run final ...
  # If final fails:
  cp clustering.backup/round-2-*.* clustering/
  iChem final-round --output-dir ./clustering --prev-round-idx 2
  ```

**General restart process**:
1. Identify which round failed
2. Fix the issue (memory, parameters, paths, etc.)
3. Regenerate the submission script for that round
4. Resubmit

```bash
# Example: Midsection job OOM'd
# Solution: Reduce bin-size and retry
iChem midsection-round --output-dir ./clustering --round-idx 2 --bin-size 3

# Or if initial job failed:
# Fix memory issue and regenerate
iChem initial-round ... --slurm-mem 32G --out-dir ./clustering
./clustering/submit_initial_jobs.sh
```

---

## Monitoring and Understanding Output

### Job Status

Monitor jobs with SLURM commands:

```bash
# Watch running jobs
watch -n 5 squeue -u $(whoami)

# Check specific job
squeue -j 12345

# View job efficiency after completion
seff 12345

# List all jobs (including completed)
squeue -u $(whoami) -a
```

### Log Files

Each job produces logs in `logs/` directory:

```
logs/
├── initial_00_12345.log          # Combined stdout/stderr
├── initial_01_12346.log
├── midsection_2_00_12347.log     # Format: midsection_{round}_{batch}_{jobid}.log
├── final_round_12348.log
```

**Typical initial job log output**:
```
[00] Processing 10 SMILES files (global indices 0-9999999)
[00] Results directory: ./clustering
[00] Starting BitBirch clustering
[00] Loading ./data/file_0.smi
[00] Loaded 1000000 SMILES from ./data/file_0.smi
[00] Assigning global indices 0 to 999999
[00] Generating ECFP4 fingerprints (2048 bits)
[00] Saved temporary fingerprints to ./clustering/temp_fps_00.npy
[00] Loading ./data/file_1.smi
...
[00] Reclustering (3 iterations)
[00] Saving results
[00] Cleaned up temporary file
[00] ✓ Complete (142.35s, 3.45 GB peak)
```

**Typical midsection job log output**:
```
[Round 2, Batch 00] Starting midsection clustering
[Round 2, Batch 00] Processing 5 file pairs
[Round 2, Batch 00] Loading round-1-bufs.label-00-uint8.npy
[Round 2, Batch 00] Loading round-1-bufs.label-01-uint8.npy
...
[Round 2, Batch 00] Reclustering (0 iterations)
[Round 2, Batch 00] Saving results
[Round 2, Batch 00] ✓ Complete (87.23s, 5.12 GB peak)
```

### Monitoring Peak Memory

Each job prints peak memory usage in the log. Check if jobs are exceeding allocation:

```bash
# View peak memory for all initial jobs
grep "peak" logs/initial_*.log

# Example output:
# [00] ✓ Complete (142.35s, 3.45 GB peak)
# [01] ✓ Complete (156.12s, 3.98 GB peak)
```

If peak exceeds allocation (e.g., 4 GB peak but `--slurm-mem 4G`), job will be killed. Increase memory and retry.

### Understanding Fingerprint Output

Each job outputs fingerprint files:

```bash
# Initial round output (multiple dtypes possible)
ls clustering/round-1-bufs.label-*.npy

# Example:
# round-1-bufs.label-00-uint8.npy      # 10M molecules with 2048-bit ECFP4
# round-1-idxs.label-00-uint8.pkl      # Molecule indices

# File sizes:
# uint8:  ~2.5 GB per 100M molecules  (2048 bits / 8)
# uint16: ~5 GB per 100M molecules    (2048 bits / 16, but stored as uint16)
# uint32: ~10 GB per 100M molecules
```

### Job Interdependencies

Understanding job flow:

```
Initial Round (10 jobs)
  ↓ [all must complete]
Midsection Round 2 (2 jobs)
  ↓ [all must complete]
Midsection Round 3 (optional, 1 job)
  ↓ [if used]
Final Round (1 job)
```

- Initial jobs run in parallel, independent of each other
- Midsection jobs run in parallel, but all depend on initial completion
- Final job depends on last midsection/initial completion

---

## Common Issues and Solutions

### Initial Round Jobs Fail

**Check logs**:
```bash
cat clustering/logs/initial_00.err
tail -50 clustering/logs/initial_00.log
```

**Common issues**:

1. **SMILES parsing errors** ("Invalid SMILES"):
   - Some molecules invalid or malformed
   - Solution: Verify input files, reduce fingerprint bits, or skip invalid molecules

2. **Memory exceeded** (job killed without output):
   - 16GB not enough for batch size
   - Solution: Decrease `--files-per-job` (e.g., 5 instead of 10)

3. **File not found**:
   - Paths in script are wrong or files moved
   - Solution: Regenerate with absolute paths or verify input files exist

### Midsection Jobs Fail

**Check logs**:
```bash
cat clustering/logs/midsection_2_00.err
```

**Common issues**:

1. **"No buffer/index files found"**:
   - Previous round didn't complete or files deleted early
   - Solution: Check if initial jobs all completed, restore backup if available

2. **"Mismatch: found 5 bufs and 3 idxs"**:
   - File pairing broken (different counts)
   - Solution: Verify all round-1-* files intact, regenerate initial if needed

3. **Memory exceeded**:
   - 48GB not enough for bin-size
   - Solution: Decrease `--bin-size` (e.g., 3 instead of 5)

### Final Job Fails

**Check logs**:
```bash
cat clustering/logs/final_round.err
```

**Common issues**:

1. **Memory exceeded**:
   - Tree consolidation exceeded 96GB
   - Solution: Increase `--slurm-mem 128G` or use more midsection rounds (pyramid)

2. **Timeout** (job killed at 24:00:00):
   - Final consolidation took too long
   - Solution: Increase `--slurm-time 48:00:00` or use more midsection rounds

---



### 1. Optimal Batch Sizing

Aim for ~20-50M molecules per initial job:
```bash
# If each file = 1M molecules
--files-per-job 20-50

# If each file = 10M molecules
--files-per-job 2-5

# If each file = 50M molecules
--files-per-job 1
```

### 2. Parallel Execution

More initial jobs = faster initial round (but more SLURM jobs):
- 100 small files + `--files-per-job 1` = 100 parallel jobs (fast)
- 100 small files + `--files-per-job 10` = 10 parallel jobs (slower, but simpler)

### 3. Threshold Tuning

Higher threshold = fewer clusters (faster):
- `--threshold 0.3` → many clusters (slower, more detailed)
- `--threshold 0.5` → fewer clusters (faster, coarser)

Test on small sample first to find good threshold.

### 4. Disable Unnecessary Reclustering

Midsection/final rounds rarely need reclustering:
```bash
# Initial: do reclustering (default 3 iterations)
iChem initial-round ... --reclustering-iterations 3

# Midsection: skip (default 0)
iChem midsection-round ... --reclustering-iterations 0

# Final: skip (default 0)
iChem final-round ... --reclustering-iterations 0
```

### 5. Use Local Storage

Store intermediate files on fast local disk, not network storage:
```bash
iChem initial-round /network/data/*.smi --out-dir /local/scratch/clustering
```

---

## Memory Efficiency

System is designed to minimize memory usage:

1. **Streaming FP save**: NumPy arrays written to disk without full materialization
2. **Progressive merging**: Buffers consolidated in phases, not all at once
3. **Intermediate cleanup**: Round files auto-deleted (optional)
4. **Tree pruning**: Internal nodes removed before serialization

**Typical memory profile** for 100M molecules:
- Initial: 16GB × ~10 jobs = distributed across HPC
- Midsection: 48GB × ~2 jobs
- Final: 96GB × 1 job
- **Peak**: 96GB (single final job, not cumulative)

---

## Advanced Scenarios

### Clustering Billion-Scale Datasets

For 1B+ molecules, single midsection round may OOM. Use multi-layer pyramiding:

```bash
# Initial round: many small jobs
iChem initial-round data/*.smi \
  --out-dir ./clustering \
  --files-per-job 1 \         # One file per job = 1000 jobs for 1000 files
  --slurm-mem 8G

# After initial completes:
# Midsection round 2: coarse merge
iChem midsection-round \
  --output-dir ./clustering \
  --round-idx 2 \
  --bin-size 50 \              # 1000 / 50 = 20 jobs
  --slurm-mem 48G

# Midsection round 3: further coarsen
iChem midsection-round \
  --output-dir ./clustering \
  --round-idx 3 \
  --bin-size 10 \              # 20 / 10 = 2 jobs
  --slurm-mem 96G

# Final
iChem final-round \
  --output-dir ./clustering \
  --prev-round-idx 3 \
  --save-centroids \
  --slurm-mem 256G             # Larger for final consolidation
```

### Local Scratch for Performance

If available, use fast local disk for intermediate files:

```bash
# Submit initial round with local scratch
SCRATCH=/local/scratch
mkdir -p $SCRATCH/clustering

iChem initial-round /network/data/*.smi \
  --out-dir $SCRATCH/clustering \
  --files-per-job 10

# After initial completes, copy results to network:
cp -r $SCRATCH/clustering/* /network/results/clustering/

# Continue with midsection from network location
iChem midsection-round \
  --output-dir /network/results/clustering \
  --round-idx 2 \
  ...
```

### Changing Parameters Between Rounds

Different thresholds per round for progressive refinement:

```bash
# Initial: coarse
iChem initial-round ... --threshold 0.5

# Midsection: medium
iChem midsection-round ... --threshold 0.35

# Final: fine
iChem final-round ... --threshold 0.2
```

This creates a hierarchical clustering where initial clusters are coarse, then refined.

### Reprocessing with Different Fingerprints

To cluster with a different fingerprint type, regenerate from initial (cannot reuse initial output with different FP type):

```bash
# Initial attempt with ECFP4
iChem initial-round ... --fp-type ECFP4 --n-bits 2048

# If want to try with MACCS instead:
rm clustering/round-1-*  # Clear old fingerprints
iChem initial-round ... --fp-type MACCS --n-bits 167

# Then midsection/final as normal
```

### Output Format Reference

**clusters.pkl**:
```python
import pickle
with open("clusters.pkl", "rb") as f:
    clusters = pickle.load(f)

# clusters is list of lists
# clusters[i] is a list of molecule IDs (global indices) in cluster i
# Example: clusters[0] = [0, 5, 12, 24]  (4 molecules in cluster 0)
```

**cluster-centroids-packed.pkl**:
```python
with open("cluster-centroids-packed.pkl", "rb") as f:
    centroids = pickle.load(f)

# centroids is dict with packed fingerprints for each cluster
# Use with BitBirch library for further analysis
```

**bitbirch.pkl** (if `--save-tree`):
```python
from bblean import BitBirch
tree = BitBirch.load("bitbirch.pkl")

# Full BitBirch tree object, can reuse for queries, etc.
```

---

## Performance Benchmarks

**Dataset**: 100M PubChem molecules, ECFP4 2048-bit

| Phase | Config | Jobs | Time | Memory Peak |
|-------|--------|------|------|-------------|
| Initial | 10 files/job | 100 | 4h | 12 GB/job |
| Midsection 2 | bin-size 5 | 20 | 1.5h | 35 GB/job |
| Final | single | 1 | 2h | 92 GB |
| **Total** | | | ~7.5h | 92 GB (peak) |

**Scaling**:
- 1B molecules: ~3-4 days with 1000 initial jobs + 2 midsection rounds + final
- 10B molecules: ~10-14 days with aggressive pyramiding (3+ midsection rounds)

**Bottlenecks**:
1. Initial round fingerprint generation (4 hours for 100M)
2. Final consolidation (2 hours, must complete before output available)

---



### File Modules

**Generator modules** (run on login node):
- `_hpc_initial_submit.py` → generates `submit_initial_jobs.sh`
- `_hpc_midsection_submit.py` → generates `submit_midsection_round_*.sh`
- `_hpc_final_submit.py` → generates `submit_final_round_job.sh`

**Worker modules** (run in SLURM jobs):
- `_hpc_initial.py` → initial clustering worker
- `_hpc_midsection.py` → midsection merge worker
- `_hpc_final.py` → final consolidation worker

**Configuration**:
- `_config.py` → centralized defaults (threshold, branching_factor, SLURM resources, etc.)

### CLI Integration

**New commands** in `cli.py`:
- `iChem initial-round` → calls `_run_initial_round()`
- `iChem midsection-round` → calls `_run_midsection_round()`
- `iChem final-round` → calls `_run_final_round()`

Each command:
1. Calls corresponding generator module
2. Generates submission script
3. Prints instructions

---

## Advantages Over Traditional Multiround

| Aspect | Traditional | HPC Version |
|--------|-------------|-------------|
| Input | Pre-computed .npy fingerprints | Raw .smi/.smi.gz files |
| FP Generation | Bottleneck on single machine | Parallelized across jobs |
| Submission | Python multiprocessing | SLURM jobs with dependencies |
| Scalability | Limited by node memory | Distributed across cluster |
| Job Control | Automatic, opaque | Explicit, transparent |
| Monitoring | Python progress bars | SLURM `squeue`, job logs |
| Restartability | No (restart from scratch) | Yes (restart from failed phase) |

---

## See Also

- BitBirch documentation: `bblean` package
- Standard multiround: `multiround_reclustering.py`
- Configuration defaults: `_config.py`
