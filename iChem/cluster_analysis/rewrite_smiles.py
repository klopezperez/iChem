from pathlib import Path
import pickle as pkl
import argparse
import gzip
import sys
from ..utils import load_smiles, load_smiles_gzipped


def rewrite_smiles_by_cluster(clusters: list[list[int]],
                               input_smiles_dir: str,
                               output_dir: str,
                               smiles_per_file: int = 1_000_000,
                               compressed: bool = False):
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
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_dir = Path(input_smiles_dir)
    smi_files = sorted(input_dir.glob("*.smi")) + sorted(input_dir.glob("*.smi.gz"))

    if not smi_files:
        raise FileNotFoundError(f"No SMILES files found in {input_smiles_dir}")

    for cluster_id, cluster_indices in enumerate(clusters):
        if not cluster_indices:
            continue

        sorted_indices = sorted(cluster_indices)
        cluster_smiles = []

        # Single pass through sorted indices, jumping files when needed
        current_file_id = -1
        current_file_smiles = None
        file_start = 0

        for idx in sorted_indices:
            file_id = idx // smiles_per_file

            # Load new file when file_id changes
            if file_id != current_file_id:
                # Clear previous file from memory
                current_file_smiles = None

                if file_id >= len(smi_files):
                    raise IndexError(f"File ID {file_id} out of range (only {len(smi_files)} files)")

                smi_file = smi_files[file_id]
                if smi_file.suffix == '.gz':
                    current_file_smiles = load_smiles_gzipped(str(smi_file))
                else:
                    current_file_smiles = load_smiles(str(smi_file))

                current_file_id = file_id
                file_start = file_id * smiles_per_file

            # Get position within current file (no modulo, just subtraction)
            pos_in_file = idx - file_start
            cluster_smiles.append(current_file_smiles[pos_in_file])

        # Write cluster SMILES to output file
        ext = '.smi.gz' if compressed else '.smi'
        output_file = output_path / f"cluster_{cluster_id}{ext}"

        if compressed:
            with gzip.open(output_file, 'wt') as f:
                for smi in cluster_smiles:
                    f.write(smi + '\n')
        else:
            with open(output_file, 'w') as f:
                for smi in cluster_smiles:
                    f.write(smi + '\n')

        print(f"Cluster {cluster_id}: wrote {len(cluster_smiles)} molecules to {output_file}", flush=True)
        sys.stdout.flush()

        # Clear memory after processing cluster
        del cluster_smiles
        del sorted_indices
        del current_file_smiles
        clusters[cluster_id] = None  # Drop indices to free memory


def main():
    parser = argparse.ArgumentParser(description='Rewrite SMILES by cluster.')
    parser.add_argument('--clusters-file', type=str, required=True, help='Path to pickled clusters file.')
    parser.add_argument('--smiles-dir', type=str, required=True, help='Directory containing original SMILES files.')
    parser.add_argument('--smiles-per-file', type=int, default=1_000_000, help='Number of SMILES per input file.')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to save cluster SMILES files.')
    parser.add_argument('--compressed', action='store_true', help='Write gzipped files.')

    args = parser.parse_args()

    with open(args.clusters_file, 'rb') as f:
        clusters = pkl.load(f)

    rewrite_smiles_by_cluster(clusters, args.smiles_dir, args.output_dir,
                              args.smiles_per_file, args.compressed)


if __name__ == '__main__':
    main()
