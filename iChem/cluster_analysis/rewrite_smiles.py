from pathlib import Path
import pickle as pkl
import argparse
import gzip
import sys
from multiprocessing import Pool
from ..utils import load_smiles, load_smiles_gzipped

_clusters_shared = None
_smi_files_shared = None
_input_smiles_dir_shared = None
_output_dir_shared = None
_smiles_per_file_shared = None
_compressed_shared = None


def _init_worker(clusters, smi_files, input_smiles_dir, output_dir, smiles_per_file, compressed):
    global _clusters_shared, _smi_files_shared, _input_smiles_dir_shared, _output_dir_shared, _smiles_per_file_shared, _compressed_shared
    _clusters_shared = clusters
    _smi_files_shared = smi_files
    _input_smiles_dir_shared = input_smiles_dir
    _output_dir_shared = output_dir
    _smiles_per_file_shared = smiles_per_file
    _compressed_shared = compressed


def _process_cluster_worker(cluster_id):
    """Worker function to process a single cluster."""
    output_path = Path(_output_dir_shared)
    ext = '.smi.gz' if _compressed_shared else '.smi'
    output_file = output_path / f"cluster_{cluster_id}{ext}"

    # Skip if already processed
    if output_file.exists():
        print(f"Cluster {cluster_id}: already exists, skipping", flush=True)
        sys.stdout.flush()
        return cluster_id

    cluster_indices = _clusters_shared[cluster_id]
    if not cluster_indices:
        return cluster_id

    sorted_indices = sorted(cluster_indices)
    cluster_smiles = []

    # Single pass through sorted indices, jumping files when needed
    current_file_id = -1
    current_file_smiles = None
    file_start = 0

    for idx in sorted_indices:
        file_id = idx // _smiles_per_file_shared

        # Load new file when file_id changes
        if file_id != current_file_id:
            current_file_smiles = None

            if file_id >= len(_smi_files_shared):
                raise IndexError(f"File ID {file_id} out of range (only {len(_smi_files_shared)} files)")

            smi_file = _smi_files_shared[file_id]
            if smi_file.suffix == '.gz':
                current_file_smiles = load_smiles_gzipped(str(smi_file))
            else:
                current_file_smiles = load_smiles(str(smi_file))

            current_file_id = file_id
            file_start = file_id * _smiles_per_file_shared

        # Get position within current file
        pos_in_file = idx - file_start
        cluster_smiles.append(current_file_smiles[pos_in_file])

    if _compressed_shared:
        with gzip.open(output_file, 'wt') as f:
            for smi in cluster_smiles:
                f.write(smi + '\n')
    else:
        with open(output_file, 'w') as f:
            for smi in cluster_smiles:
                f.write(smi + '\n')

    print(f"Cluster {cluster_id}: wrote {len(cluster_smiles)} molecules to {output_file}", flush=True)
    sys.stdout.flush()

    return cluster_id


def find_first_missing_cluster(output_dir: str, num_clusters: int, compressed: bool = False) -> int:
    """Find the first cluster that hasn't been written yet.

    Parameters
    ----------
    output_dir : str
        Directory containing output SMILES files.
    num_clusters : int
        Total number of clusters.
    compressed : bool
        Whether to look for .smi.gz or .smi files.

    Returns
    -------
    int
        Index of first missing cluster, or num_clusters if all exist.
    """
    output_path = Path(output_dir)
    ext = '.smi.gz' if compressed else '.smi'

    for cluster_id in range(num_clusters):
        output_file = output_path / f"cluster_{cluster_id}{ext}"
        if not output_file.exists():
            return cluster_id

    return num_clusters


def rewrite_smiles_by_cluster(clusters: list[list[int]],
                               input_smiles_dir: str,
                               output_dir: str,
                               smiles_per_file: int = 1_000_000,
                               compressed: bool = False,
                               num_workers: int = 8,
                               start_at: int = 0):
    """Reorganize SMILES files by cluster, loading only necessary molecules.

    Parameters
    ----------
    clusters : list[list[int]]
        List of clusters where each cluster is a list of SMILES indices.
    input_smiles_dir : str
        Directory containing input SMILES files (*.smi or *.smi.gz).
    output_dir : str
        Directory to write cluster-organized SMILES files.
    smiles_per_file : int
        Number of SMILES per input file (default 1M).
    compressed : bool
        If True, write gzipped files (.smi.gz), else plain text (.smi).
    num_workers : int
        Number of parallel processes (default 8).
    start_at : int
        Cluster index to start processing from (default 0). Clusters before this index are freed from memory.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_smiles_dir)
    smi_files = sorted(input_dir.glob("*.smi")) + sorted(input_dir.glob("*.smi.gz"))

    if not smi_files:
        raise FileNotFoundError(f"No SMILES files found in {input_smiles_dir}")

    # Free memory for already-processed clusters
    for i in range(start_at):
        clusters[i] = None

    print(f"Starting processing from cluster {start_at}", flush=True)

    # Process clusters in parallel
    with Pool(processes=num_workers,
              initializer=_init_worker,
              initargs=(clusters, smi_files, input_smiles_dir, output_dir, smiles_per_file, compressed)) as pool:
        for cluster_id in pool.imap_unordered(_process_cluster_worker, range(start_at, len(clusters))):
            # Memory released after each cluster is processed
            clusters[cluster_id] = None


def main():
    parser = argparse.ArgumentParser(description='Rewrite SMILES by cluster.')
    parser.add_argument('--clusters-file', type=str, required=True, help='Path to pickled clusters file.')
    parser.add_argument('--smiles-dir', type=str, required=True, help='Directory containing original SMILES files.')
    parser.add_argument('--smiles-per-file', type=int, default=1_000_000, help='Number of SMILES per input file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save cluster SMILES files.')
    parser.add_argument('--compressed', action='store_true', help='Write gzipped files.')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of parallel processes (default 8).')
    parser.add_argument('--start-at', type=int, default=0, help='Cluster index to start processing from (default 0).')

    args = parser.parse_args()

    with open(args.clusters_file, 'rb') as f:
        clusters = pkl.load(f)

    rewrite_smiles_by_cluster(clusters, args.smiles_dir, args.output_dir,
                              args.smiles_per_file, args.compressed, args.num_workers, args.start_at)


if __name__ == '__main__':
    main()
