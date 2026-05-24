"""Command line entry points for iChem."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from .bitbirch.cluster import cluster
from .bitbirch.multiround_reclustering import run_multiround_reclustering
from .bitbirch._hpc_initial_submit import prepare_initial_round_jobs
from .bitbirch._hpc_midsection_submit import prepare_midsection_round_jobs
from .bitbirch._hpc_final_submit import prepare_final_round_job
from .bitbirch import _config
from .utils.fingerprints import binary_fps, count_fps, real_fps
from ._cli import load_smiles, get_smi_files, get_output_path
import numpy as np


def _print_banner() -> None:
    """Print iChem banner with group attribution."""
    banner = r"""
      _     _     _     _     _
     / \   / \   / \   / \   / \
    |010|-|101|-|010|-|101|-|010|
     \_/   \_/   \_/   \_/   \_/
    
    iChem: Instant Cheminformatics
      _     _     _     _     _    
     / \   / \   / \   / \   / \
    |101|-|010|-|101|-|010|-|101|
     \_/   \_/   \_/   \_/   \_/

    Miranda-Quintana Group
    University of Florida
    Department of Chemistry
    """
    print(banner)


def _path_list(value: str) -> list[Path]:
    path = Path(value)
    if path.is_dir():
        return sorted(path.glob("*.npy"))
    return [path]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="iChem", description="iChem BitBIRCH tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    cluster_parser = subparsers.add_parser("cluster", help="Cluster a file or directory")
    cluster_parser.add_argument("input", type=Path, help="Path to a .smi, .smi.gz, .npy file, or a directory")
    cluster_parser.add_argument("--threshold", type=float, default=None)
    cluster_parser.add_argument("--fp-type", default=_config.FINGERPRINT_TYPE)
    cluster_parser.add_argument("--n-bits", type=int, default=_config.N_BITS)
    cluster_parser.add_argument("--branching-factor", type=int, default=_config.BRANCHING_FACTOR)
    cluster_parser.add_argument("--merge-criterion", default=_config.MERGE_CRITERION)
    cluster_parser.add_argument("--recluster-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_INITIAL)
    cluster_parser.add_argument("--recluster-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD)
    cluster_parser.add_argument("--force-sequential", action=argparse.BooleanOptionalAction, default=False)
    cluster_parser.add_argument("--save-tree", action=argparse.BooleanOptionalAction, default=_config.SAVE_TREE)
    cluster_parser.add_argument("--save-centroids", action=argparse.BooleanOptionalAction, default=_config.SAVE_CENTROIDS)
    cluster_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    multiround_parser = subparsers.add_parser("multiround", help="Run multi-round clustering")
    multiround_parser.add_argument("input", nargs="+", type=Path, help="Input .npy files or a directory containing them")
    multiround_parser.add_argument("--out-dir", type=Path, required=True)
    multiround_parser.add_argument("--n-features", type=int, default=None)
    multiround_parser.add_argument("--input-is-packed", action=argparse.BooleanOptionalAction, default=True)
    multiround_parser.add_argument("--num-initial-processes", type=int, default=10)
    multiround_parser.add_argument("--num-midsection-processes", type=int, default=None)
    multiround_parser.add_argument("--merge-criterion", default=_config.MERGE_CRITERION)
    multiround_parser.add_argument("--branching-factor", type=int, default=_config.BRANCHING_FACTOR)
    multiround_parser.add_argument("--threshold", type=float, default=_config.THRESHOLD)
    multiround_parser.add_argument("--midsection-threshold-change", type=float, default=_config.MIDSECTION_THRESHOLD_CHANGE)
    multiround_parser.add_argument("--num-midsection-rounds", type=int, default=_config.NUM_MIDSECTION_ROUNDS)
    multiround_parser.add_argument("--bin-size", type=int, default=_config.BIN_SIZE)
    multiround_parser.add_argument("--max-tasks-per-process", type=int, default=1)
    multiround_parser.add_argument("--save-tree", action=argparse.BooleanOptionalAction, default=_config.SAVE_TREE)
    multiround_parser.add_argument("--save-centroids", action=argparse.BooleanOptionalAction, default=_config.SAVE_CENTROIDS)
    multiround_parser.add_argument("--reclustering-iterations-initial", type=int, default=_config.RECLUSTERING_ITERATIONS_INITIAL)
    multiround_parser.add_argument("--reclustering-iterations-midsection", type=int, default=_config.RECLUSTERING_ITERATIONS_MIDSECTION)
    multiround_parser.add_argument("--reclustering-iterations-final", type=int, default=_config.RECLUSTERING_ITERATIONS_FINAL)
    multiround_parser.add_argument("--reclustering-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD)
    multiround_parser.add_argument("--max-fps", type=int, default=None)
    multiround_parser.add_argument("--cleanup", action=argparse.BooleanOptionalAction, default=True)
    multiround_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    initial_round_parser = subparsers.add_parser(
        "initial-round", help="Run initial round HPC clustering with automatic job submission"
    )
    initial_round_parser.add_argument("input", nargs="+", type=Path, help="Input .smi or .smi.gz files")
    initial_round_parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for job scripts and logs")
    initial_round_parser.add_argument("--files-per-job", type=int, default=_config.FILES_PER_JOB, help="Number of files per job")
    initial_round_parser.add_argument("--max-jobs-per-script", type=int, default=_config.MAX_JOBS_PER_SCRIPT, help="Maximum jobs per submission script")
    initial_round_parser.add_argument("--threshold", type=float, default=_config.THRESHOLD, help="BitBirch threshold")
    initial_round_parser.add_argument("--branching-factor", type=int, default=_config.BRANCHING_FACTOR, help="BitBirch branching factor")
    initial_round_parser.add_argument("--merge-criterion", default=_config.MERGE_CRITERION, help="Merge criterion")
    initial_round_parser.add_argument("--fp-type", default=_config.FINGERPRINT_TYPE, help="Fingerprint type")
    initial_round_parser.add_argument("--n-bits", type=int, default=_config.N_BITS, help="Number of fingerprint bits")
    initial_round_parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_INITIAL, help="Reclustering iterations")
    initial_round_parser.add_argument("--reclustering-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold for reclustering")
    initial_round_parser.add_argument("--slurm-mem", default=_config.SLURM_MEM_INITIAL, help="SLURM memory allocation")
    initial_round_parser.add_argument("--slurm-cpus", type=int, default=_config.SLURM_CPUS_INITIAL, help="SLURM CPU count")
    initial_round_parser.add_argument("--slurm-time", default=_config.SLURM_TIME, help="SLURM time limit")
    initial_round_parser.add_argument("--slurm-partition", default=_config.SLURM_PARTITION, help="SLURM partition (optional)")
    initial_round_parser.add_argument("--result-base-dir", type=Path, default=None, help="Base directory for results (optional)")
    initial_round_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    midsection_round_parser = subparsers.add_parser(
        "midsection-round", help="Run midsection round HPC clustering"
    )
    midsection_round_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory containing previous round results")
    midsection_round_parser.add_argument("--round-idx", type=int, required=True, help="Current round index")
    midsection_round_parser.add_argument("--bin-size", type=int, default=_config.BIN_SIZE, help="Number of buffer/index pairs per batch job")
    midsection_round_parser.add_argument("--max-jobs-per-script", type=int, default=_config.MAX_JOBS_PER_SCRIPT, help="Maximum jobs per submission script")
    midsection_round_parser.add_argument("--threshold", type=float, default=None, help="BitBirch threshold (default: same as config)")
    midsection_round_parser.add_argument("--branching-factor", type=int, default=None, help="BitBirch branching factor (default: same as config)")
    midsection_round_parser.add_argument("--merge-criterion", default=_config.MERGE_CRITERION, help="Merge criterion")
    midsection_round_parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_MIDSECTION, help="Reclustering iterations")
    midsection_round_parser.add_argument("--reclustering-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold for reclustering")
    midsection_round_parser.add_argument("--slurm-mem", default=_config.SLURM_MEM_MIDSECTION, help="SLURM memory allocation")
    midsection_round_parser.add_argument("--slurm-cpus", type=int, default=_config.SLURM_CPUS_MIDSECTION, help="SLURM CPU count")
    midsection_round_parser.add_argument("--slurm-time", default=_config.SLURM_TIME, help="SLURM time limit")
    midsection_round_parser.add_argument("--slurm-partition", default=_config.SLURM_PARTITION, help="SLURM partition (optional)")
    midsection_round_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    final_round_parser = subparsers.add_parser(
        "final-round", help="Run final round HPC clustering"
    )
    final_round_parser.add_argument("--output-dir", type=Path, required=True, help="Output directory containing previous round results")
    final_round_parser.add_argument("--prev-round-idx", type=int, required=True, help="Previous round index to read from")
    final_round_parser.add_argument("--threshold", type=float, default=None, help="BitBirch threshold (default: same as config)")
    final_round_parser.add_argument("--branching-factor", type=int, default=None, help="BitBirch branching factor (default: same as config)")
    final_round_parser.add_argument("--merge-criterion", default=_config.MERGE_CRITERION, help="Merge criterion")
    final_round_parser.add_argument("--reclustering-iterations", type=int, default=_config.RECLUSTERING_ITERATIONS_FINAL, help="Reclustering iterations")
    final_round_parser.add_argument("--reclustering-extra-threshold", type=float, default=_config.RECLUSTERING_EXTRA_THRESHOLD, help="Extra threshold for reclustering")
    final_round_parser.add_argument("--save-tree", action=argparse.BooleanOptionalAction, default=False, help="Save the BitBirch tree")
    final_round_parser.add_argument("--save-centroids", action=argparse.BooleanOptionalAction, default=False, help="Save centroids and cluster assignments")
    final_round_parser.add_argument("--slurm-mem", default=_config.SLURM_MEM_FINAL, help="SLURM memory allocation")
    final_round_parser.add_argument("--slurm-cpus", type=int, default=_config.SLURM_CPUS_FINAL, help="SLURM CPU count")
    final_round_parser.add_argument("--slurm-time", default=_config.SLURM_TIME, help="SLURM time limit")
    final_round_parser.add_argument("--slurm-partition", default=_config.SLURM_PARTITION, help="SLURM partition (optional)")
    final_round_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)

    # Fingerprint generation commands
    binary_fps_parser = subparsers.add_parser(
        "binary-fps", help="Generate binary fingerprints from SMILES"
    )
    binary_fps_parser.add_argument(
        "input", type=Path, help="Input .smi, .smi.gz file or directory containing them"
    )
    binary_fps_parser.add_argument(
        "--fp-type", default="ECFP4",
        help="Fingerprint type ['RDKIT', 'ECFP4', 'ECFP6', 'AP', 'TT', 'MACCS']"
    )
    binary_fps_parser.add_argument(
        "--n-bits", type=int, default=2048,
        help="Number of bits for fingerprint (ignored for MACCS)"
    )
    binary_fps_parser.add_argument(
        "--packed", action=argparse.BooleanOptionalAction, default=True,
        help="Return packed fingerprints (default: True)"
    )
    binary_fps_parser.add_argument(
        "--return-invalid", action=argparse.BooleanOptionalAction, default=True,
        help="Return indices of invalid SMILES"
    )
    binary_fps_parser.add_argument(
        "--standarize", action=argparse.BooleanOptionalAction, default=False,
        help="Standardize molecules before generating fingerprints"
    )
    binary_fps_parser.add_argument(
        "--n-processes", type=int, default=None,
        help="Number of processes for parallelization (default: min(available_cpus, 32))"
    )
    binary_fps_parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .npy file for fingerprints (default: binary_fps.npy)"
    )

    count_fps_parser = subparsers.add_parser(
        "count-fps", help="Generate count fingerprints from SMILES"
    )
    count_fps_parser.add_argument(
        "input", type=Path, help="Input .smi, .smi.gz file or directory containing them"
    )
    count_fps_parser.add_argument(
        "--fp-type", default="ECFP4",
        help="Fingerprint type ['RDKIT', 'ECFP4', 'ECFP6']"
    )
    count_fps_parser.add_argument(
        "--n-bits", type=int, default=2048,
        help="Number of bits for fingerprint"
    )
    count_fps_parser.add_argument(
        "--return-invalid", action=argparse.BooleanOptionalAction, default=True,
        help="Return indices of invalid SMILES"
    )
    count_fps_parser.add_argument(
        "--n-processes", type=int, default=None,
        help="Number of processes for parallelization (default: min(available_cpus, 32))"
    )
    count_fps_parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .npy file for fingerprints (default: count_fps.npy)"
    )

    real_fps_parser = subparsers.add_parser(
        "real-fps", help="Generate real-valued fingerprints from RDKit descriptors"
    )
    real_fps_parser.add_argument(
        "input", type=Path, help="Input .smi, .smi.gz file or directory containing them"
    )
    real_fps_parser.add_argument(
        "--return-invalid", action=argparse.BooleanOptionalAction, default=True,
        help="Return indices of invalid SMILES"
    )
    real_fps_parser.add_argument(
        "--out", type=Path, default=None,
        help="Output .npy file for fingerprints (default: real_fps.npy)"
    )

    return parser


def _run_cluster(args: argparse.Namespace) -> int:
    t0 = time.perf_counter()
    print(f"[cluster] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    result = cluster(
        args.input,
        threshold=args.threshold,
        fp_type=args.fp_type,
        n_bits=args.n_bits,
        branching_factor=args.branching_factor,
        merge_criterion=args.merge_criterion,
        recluster_iterations=args.recluster_iterations,
        recluster_extra_threshold=args.recluster_extra_threshold,
        verbose=args.verbose,
        force_sequential=args.force_sequential,
        save_tree=args.save_tree,
        save_centroids=args.save_centroids,
    )

    elapsed = time.perf_counter() - t0
    print(f"[cluster] Finished in {elapsed:.2f} s")

    # 0 means multiround path; non-zero sequential results are saved in cluster.py
    if result == 0:
        print("Multiround clustering completed. Results saved by multiround handler.")
    else:
        print("Sequential clustering completed. Results saved as clusters.pkl near the input fingerprints.")
    return 0


def _run_multiround(args: argparse.Namespace) -> int:
    input_files: list[Path] = []
    for value in args.input:
        input_files.extend(_path_list(str(value)))
    if not input_files:
        raise ValueError("No input files were found")

    t0 = time.perf_counter()
    print(f"[multiround] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    timer = run_multiround_reclustering(
        input_files=input_files,
        out_dir=args.out_dir,
        n_features=args.n_features,
        input_is_packed=args.input_is_packed,
        num_initial_processes=args.num_initial_processes,
        num_midsection_processes=args.num_midsection_processes,
        merge_criterion=args.merge_criterion,
        branching_factor=args.branching_factor,
        threshold=args.threshold,
        midsection_threshold_change=args.midsection_threshold_change,
        num_midsection_rounds=args.num_midsection_rounds,
        bin_size=args.bin_size,
        max_tasks_per_process=args.max_tasks_per_process,
        save_tree=args.save_tree,
        save_centroids=args.save_centroids,
        reclustering_iterations_initial=args.reclustering_iterations_initial,
        reclustering_iterations_midsection=args.reclustering_iterations_midsection,
        reclustering_iterations_final=args.reclustering_iterations_final,
        reclustering_extra_threshold=args.reclustering_extra_threshold,
        max_fps=args.max_fps,
        verbose=args.verbose,
        cleanup=args.cleanup,
    )
    elapsed = time.perf_counter() - t0
    print(f"[multiround] Finished in {elapsed:.2f} s")
    print(timer)
    return 0


def _run_initial_round(args: argparse.Namespace) -> int:
    input_files = [Path(f) for f in args.input]
    if not input_files:
        raise ValueError("No input files were found")

    script_paths = prepare_initial_round_jobs(
        input_files=input_files,
        output_dir=args.out_dir,
        files_per_job=args.files_per_job,
        threshold=args.threshold,
        branching_factor=args.branching_factor,
        merge_criterion=args.merge_criterion,
        fp_type=args.fp_type,
        n_bits=args.n_bits,
        reclustering_iterations=args.reclustering_iterations,
        extra_threshold=args.reclustering_extra_threshold,
        slurm_mem=args.slurm_mem,
        slurm_cpus=args.slurm_cpus,
        slurm_time=args.slurm_time,
        slurm_partition=args.slurm_partition,
        result_base_dir=args.result_base_dir,
        max_jobs_per_script=args.max_jobs_per_script,
        verbose=args.verbose,
    )

    # Handle single or multiple scripts
    if isinstance(script_paths, list):
        print(f"\n✓ Generated {len(script_paths)} submission scripts:")
        for i, path in enumerate(script_paths, 1):
            print(f"  {i}. bash {Path(path).resolve()}")
        print(f"\nRun each script to submit batches of jobs (max {args.max_jobs_per_script} per script)")
    else:
        print(f"\n✓ Generated submission script: {script_paths}")
        print(f"Run the following to submit all initial round jobs:")
        print(f"\n  bash {Path(script_paths).resolve()}\n")
    return 0


def _run_midsection_round(args: argparse.Namespace) -> int:
    script_paths = prepare_midsection_round_jobs(
        output_dir=args.output_dir,
        round_idx=args.round_idx,
        bin_size=args.bin_size,
        threshold=args.threshold,
        branching_factor=args.branching_factor,
        merge_criterion=args.merge_criterion,
        reclustering_iterations=args.reclustering_iterations,
        reclustering_extra_threshold=args.reclustering_extra_threshold,
        slurm_mem=args.slurm_mem,
        slurm_cpus=args.slurm_cpus,
        slurm_time=args.slurm_time,
        slurm_partition=args.slurm_partition,
        max_jobs_per_script=args.max_jobs_per_script,
        verbose=args.verbose,
    )

    # Handle single or multiple scripts
    if isinstance(script_paths, list):
        print(f"\n✓ Generated {len(script_paths)} submission scripts:")
        for i, path in enumerate(script_paths, 1):
            print(f"  {i}. bash {Path(path).resolve()}")
        print(f"\nRun each script to submit batches of jobs (max {args.max_jobs_per_script} per script)")
    else:
        print(f"\n✓ Generated submission script: {script_paths}")
        print(f"Run the following to submit all midsection round jobs:")
        print(f"\n  bash {Path(script_paths).resolve()}\n")
    return 0


def _run_final_round(args: argparse.Namespace) -> int:
    script_path = prepare_final_round_job(
        output_dir=args.output_dir,
        prev_round_idx=args.prev_round_idx,
        threshold=args.threshold,
        branching_factor=args.branching_factor,
        merge_criterion=args.merge_criterion,
        reclustering_iterations=args.reclustering_iterations,
        reclustering_extra_threshold=args.reclustering_extra_threshold,
        save_tree=args.save_tree,
        save_centroids=args.save_centroids,
        slurm_mem=args.slurm_mem,
        slurm_cpus=args.slurm_cpus,
        slurm_time=args.slurm_time,
        slurm_partition=args.slurm_partition,
        verbose=args.verbose,
    )

    print(f"\n✓ Generated submission script: {script_path}")
    print(f"Run the following to submit the final round job:")
    print(f"\n  bash {Path(script_path).resolve()}\n")
    return 0


def _run_binary_fps(args: argparse.Namespace) -> int:
    smi_files = get_smi_files(args.input)
    
    if not smi_files:
        raise ValueError(f"No .smi or .smi.gz files found in {args.input}")
    
    output_dir = args.input if args.input.is_dir() else args.input.parent
    
    for smi_file in smi_files:
        print(f"\nProcessing: {smi_file.name}")
        print("Loading SMILES...")
        smiles = load_smiles(smi_file)
        print(f"Loaded {len(smiles)} SMILES")

        print("Generating binary fingerprints...")
        t0 = time.perf_counter()
        if args.return_invalid:
            fps, invalid_indices = binary_fps(
                smiles,
                fp_type=args.fp_type,
                n_bits=args.n_bits,
                return_invalid=True,
                standarize=args.standarize,
                packed=args.packed,
                n_processes=args.n_processes,
            )
            if invalid_indices:
                print(f"Warning: {len(invalid_indices)} invalid SMILES found")
                invalid_path = output_dir / f"{smi_file.stem}_invalid_indices.json"
                with open(invalid_path, "w") as f:
                    json.dump(invalid_indices, f, indent=2)
                print(f"✓ Saved {len(invalid_indices)} invalid indices to {invalid_path}")
        else:
            fps = binary_fps(
                smiles,
                fp_type=args.fp_type,
                n_bits=args.n_bits,
                return_invalid=False,
                standarize=args.standarize,
                packed=args.packed,
                n_processes=args.n_processes,
            )
        elapsed = time.perf_counter() - t0
        print(f"✓ Generated in {elapsed:.2f}s")

        if args.out:
            output_path = args.out
        else:
            output_path = get_output_path(smi_file, args.fp_type, '.npy', output_dir)
        
        np.save(output_path, fps)
        print(f"✓ Saved {fps.shape[0]} fingerprints to {output_path}")

    return 0


def _run_count_fps(args: argparse.Namespace) -> int:
    smi_files = get_smi_files(args.input)
    
    if not smi_files:
        raise ValueError(f"No .smi or .smi.gz files found in {args.input}")
    
    output_dir = args.input if args.input.is_dir() else args.input.parent
    
    for smi_file in smi_files:
        print(f"\nProcessing: {smi_file.name}")
        print("Loading SMILES...")
        smiles = load_smiles(smi_file)
        print(f"Loaded {len(smiles)} SMILES")

        print("Generating count fingerprints...")
        t0 = time.perf_counter()
        if args.return_invalid:
            fps, invalid_indices = count_fps(
                smiles,
                fp_type=args.fp_type,
                n_bits=args.n_bits,
                return_invalid=True,
                n_processes=args.n_processes,
            )
            if invalid_indices:
                print(f"Warning: {len(invalid_indices)} invalid SMILES found")
                invalid_path = output_dir / f"{smi_file.stem}_invalid_indices.json"
                with open(invalid_path, "w") as f:
                    json.dump(invalid_indices, f, indent=2)
                print(f"✓ Saved {len(invalid_indices)} invalid indices to {invalid_path}")
        else:
            fps = count_fps(
                smiles,
                fp_type=args.fp_type,
                n_bits=args.n_bits,
                return_invalid=False,
                n_processes=args.n_processes,
            )
        elapsed = time.perf_counter() - t0
        print(f"✓ Generated in {elapsed:.2f}s")

        if args.out:
            output_path = args.out
        else:
            output_path = get_output_path(smi_file, args.fp_type, '.npy', output_dir)
        
        np.save(output_path, fps)
        print(f"✓ Saved {fps.shape[0]} fingerprints to {output_path}")

    return 0


def _run_real_fps(args: argparse.Namespace) -> int:
    smi_files = get_smi_files(args.input)
    
    if not smi_files:
        raise ValueError(f"No .smi or .smi.gz files found in {args.input}")
    
    output_dir = args.input if args.input.is_dir() else args.input.parent
    
    for smi_file in smi_files:
        print(f"\nProcessing: {smi_file.name}")
        print("Loading SMILES...")
        smiles = load_smiles(smi_file)
        print(f"Loaded {len(smiles)} SMILES")

        print("Generating real-valued fingerprints...")
        t0 = time.perf_counter()
        if args.return_invalid:
            fps, invalid_indices = real_fps(
                smiles, return_invalid=True
            )
            if invalid_indices:
                print(f"Warning: {len(invalid_indices)} invalid SMILES found")
                invalid_path = output_dir / f"{smi_file.stem}_invalid_indices.json"
                with open(invalid_path, "w") as f:
                    json.dump(invalid_indices, f, indent=2)
                print(f"✓ Saved {len(invalid_indices)} invalid indices to {invalid_path}")
        else:
            fps = real_fps(smiles, return_invalid=False)
        elapsed = time.perf_counter() - t0
        print(f"✓ Generated in {elapsed:.2f}s")

        if args.out:
            output_path = args.out
        else:
            output_path = get_output_path(smi_file, 'RDKitDescriptors', '.npy', output_dir)
        
        np.save(output_path, fps)
        print(f"✓ Saved {fps.shape[0]} fingerprints to {output_path}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _print_banner()

    if args.command == "cluster":
        return _run_cluster(args)
    if args.command == "multiround":
        return _run_multiround(args)
    if args.command == "initial-round":
        return _run_initial_round(args)
    if args.command == "midsection-round":
        return _run_midsection_round(args)
    if args.command == "final-round":
        return _run_final_round(args)
    if args.command == "binary-fps":
        return _run_binary_fps(args)
    if args.command == "count-fps":
        return _run_count_fps(args)
    if args.command == "real-fps":
        return _run_real_fps(args)
    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
