import gzip as gz
import pickle
from pathlib import Path

import numpy as np # type: ignore

from bblean.bitbirch import BitBirch # type: ignore
from bblean.fingerprints import _get_fps_file_num # type: ignore

from .optimal_threshold import optimal_threshold
from ..utils import binary_fps, load_smiles, load_smiles_gzipped
from .multiround_reclustering import run_multiround_reclustering

from ._config import (BRANCHING_FACTOR,
                      MERGE_CRITERION,
                      FINGERPRINT_TYPE,
                      N_BITS,
                      SAVE_TREE,
                      SAVE_CENTROIDS,
                      RECLUSTERING_ITERATIONS_INITIAL,
                      RECLUSTERING_EXTRA_THRESHOLD,
                      VERBOSE)


def _npy_path_for(path: Path) -> Path:
    """Return a Path in the same directory with all suffixes removed and
    replaced by a single .npy suffix. Preserves parent directory.
    Handles names like 'foo.smi.gz' -> 'foo.npy'.
    """
    base = path.name
    for s in path.suffixes:
        if base.endswith(s):
            base = base[: -len(s)]
    return path.with_name(base + '.npy')

def cluster(file_path: str,
            threshold: float = None,
            fp_type: str = FINGERPRINT_TYPE,
            n_bits: int = N_BITS,
            branching_factor: int = BRANCHING_FACTOR,
            merge_criterion: str = MERGE_CRITERION,
            recluster_iterations: int = RECLUSTERING_ITERATIONS_INITIAL,
            recluster_extra_threshold: float = RECLUSTERING_EXTRA_THRESHOLD,
            verbose: bool = VERBOSE,
            force_sequential: bool = False,
            save_tree: bool = SAVE_TREE,
            save_centroids: bool = SAVE_CENTROIDS):
    """Cluster the molecules using the best_practices recommendations on the
    paper.

    Parameters
    ----------
    file_path : Path
        Path to the file containing the .npy or .smi data.
    threshold : float, optional
        Similarity threshold for clustering. If None, it will be determined
        automatically.
    fp_type : str
        Type of fingerprints to use. 'ECFP4' or 'ECFP6', 'RDKIT', 'AP' etc.
    n_bits : int
        Number of bits for the fingerprints.
    branching_factor : int
        Branching factor for the BitBirch algorithm.
    merge_criterion : str, optional
        Criterion for merging nodes in the BitBirch algorithm.

    Returns
    -------
    cluster_ids : list
        A list of clusters, where each cluster is a list of molecule IDs. Save
        in a .pkl file.
    """
    file_path = Path(file_path)
    # Check if input is a directory or a file
    if file_path.is_file():
        if file_path.name.endswith(('.smi', '.smi.gz')):
            if verbose:
                print(f"Processing file: {file_path}")
            # Generate the .npy file and then recurse cluster with the .npy file
            npy_file_path = _prepare_fps_single(file_path, fp_type, n_bits, verbose)
            return cluster(
                npy_file_path,
                threshold=threshold,
                fp_type=fp_type,
                n_bits=n_bits,
                branching_factor=branching_factor,
                merge_criterion=merge_criterion,
                recluster_iterations=recluster_iterations,
                recluster_extra_threshold=recluster_extra_threshold,
                verbose=verbose,
                force_sequential=force_sequential,
                save_tree=save_tree,
                save_centroids=save_centroids,
            )
        elif file_path.suffix == '.npy':
            if verbose:
                print(f"Processing fingerprint file: {file_path}")
            n_mols = _get_fps_file_num(file_path)
            if n_mols > 10_000_000:
                print(f"Number of molecules in the file: {n_mols}")
                print(
                    "WARNING: We recommend using the multiround reclustering from CLI."
                )
                print(
                    "Prepare the fingerprints in separate files and then use multiround."
                )
                return 0
            if n_mols > 1_000_000 and force_sequential == False:
                print(f"Number of molecules in the file: {n_mols}")
                print(
                    "Using multiround reclustering for the file." \
                    "If you want to cluster sequentially, use the --force-sequential flag."
                )
                # Load the fps input file and split into number of initial processes
                fps = np.load(file_path, mmap_mode="r")
                num_initial_processes = 4
                npy_paths = []
                outdir = file_path.parent
                for i, chunk in enumerate(np.array_split(fps, num_initial_processes)):
                    npy_path = outdir / f"temporary_fps_{i}.npy"
                    np.save(npy_path, chunk)
                    npy_paths.append(npy_path)
                if threshold is None:
                    print("Estimating optimal threshold...")
                    threshold = optimal_threshold(fps, factor=3.5)
                    print(f"Optimal threshold estimated: {threshold:.4f}")
                run_multiround_reclustering(
                        input_files = npy_paths,
                        out_dir= outdir,
                        num_initial_processes = num_initial_processes,
                        num_midsection_processes = 2,
                        merge_criterion = merge_criterion,
                        branching_factor = branching_factor,
                        threshold = threshold,
                        midsection_threshold_change = 0,
                        # Advanced
                        num_midsection_rounds = 1,
                        bin_size = 4,
                        save_tree = save_tree,
                        save_centroids = save_centroids,
                        reclustering_iterations_initial = recluster_iterations,
                        reclustering_iterations_midsection = recluster_iterations,
                        reclustering_iterations_final = recluster_iterations,
                        reclustering_extra_threshold = recluster_extra_threshold,
                        # Debug
                        verbose = verbose,
                        cleanup = True,
                        )
                for npy_path in npy_paths:
                    npy_path.unlink() # Remove the temporary files
                return 0
            else:
                print(f"Number of molecules in the file: {n_mols}")
                print(
                    "Using sequential clustering for the file."
                )
                cluster_ids = _cluster_from_npy_file(
                    file_path,
                    threshold,
                    branching_factor,
                    merge_criterion,
                    recluster_iterations,
                    recluster_extra_threshold,
                    save_tree,
                    save_centroids,
                    verbose,
                )
                return cluster_ids
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    elif file_path.is_dir():
        # Read all .npy files in the directory and cluster them sequentially
        npy_files = sorted(file_path.glob('*.npy'))

        # Read all .smi files in the directory and cluster them sequentially
        smi_files = sorted(file_path.glob('*.smi')) + sorted(file_path.glob('*.smi.gz'))

        # If only one .npy file, treat it as a single file input
        if len(npy_files) == 1:
            return cluster(
                str(npy_files[0]),
                threshold=threshold,
                fp_type=fp_type,
                n_bits=n_bits,
                branching_factor=branching_factor,
                merge_criterion=merge_criterion,
                recluster_iterations=recluster_iterations,
                recluster_extra_threshold=recluster_extra_threshold,
                verbose=verbose,
                force_sequential=force_sequential,
                save_tree=save_tree,
                save_centroids=save_centroids,
            )

        if len(npy_files) > 0:
            if verbose:
                print(
                    f"Found {len(npy_files)} .npy files in directory. "
                )
            total_n_fingerprints = sum(_get_fps_file_num(npy_file) for npy_file in npy_files)
            if total_n_fingerprints > 10_000_000:
                print(f"Total number of fingerprints in the directory: {total_n_fingerprints}")
                print(
                    "WARNING: We recommend using the multiround reclustering from CLI."
                )
                print(
                    "Use multiple middle rounds for more efficient and more memory efficient clustering."
                )
                return 0
            # Estimate threshold once if not provided
            if threshold is None:
                if verbose:
                    print("Estimating optimal threshold...")
                threshold = optimal_threshold(np.load(npy_files[0], mmap_mode='r'), factor=3.5)
                if verbose:
                    print(f"Optimal threshold estimated on first file: {threshold:.4f}")
            if total_n_fingerprints > 1_000_000 and force_sequential == False:
                print(f"Total number of fingerprints in the directory: {total_n_fingerprints}")
                print(
                    "Using multiround reclustering for the directory." \
                    "This will be faster than sequential clustering." \
                    "If you want to cluster sequentially, use the --force-sequential flag."
                )
                num_initial_processes = 4
                output_dir = file_path
                run_multiround_reclustering(
                            input_files = npy_files,
                            out_dir= output_dir,
                            num_initial_processes = num_initial_processes,
                            num_midsection_processes = 2,
                            merge_criterion = merge_criterion,
                            branching_factor = branching_factor,
                            threshold = threshold,
                            midsection_threshold_change = 0,
                            # Advanced
                            num_midsection_rounds = 1,
                            bin_size = 4,
                            save_tree = save_tree,
                            save_centroids = save_centroids,
                            reclustering_iterations_initial = recluster_iterations,
                            reclustering_iterations_midsection = recluster_iterations,
                            reclustering_iterations_final= recluster_iterations,
                            reclustering_extra_threshold = recluster_extra_threshold,
                            # Debug
                            verbose = verbose,
                            cleanup = True,
                            )
            else:
                print(f"Total number of fingerprints in the directory: {total_n_fingerprints}")
                print(
                    "Using sequential clustering for the directory." 
                )
                cluster_ids = _cluster_multiple_npy_sequential(
                    npy_files,
                    threshold,
                    branching_factor,
                    merge_criterion,
                    recluster_iterations,
                    recluster_extra_threshold,
                    file_path,
                    save_tree,
                    save_centroids,
                    verbose,
                )
                return cluster_ids
        elif len(smi_files) > 0:
            if verbose:
                print(
                    f"Found {len(smi_files)} .smi files in directory. "
                    "Preaparing fingerprints for clustering."
                )
            npy_paths = _prepare_fps_directory(file_path, fp_type, n_bits, verbose)
            # Recurse on the directory so `cluster()` can detect the newly
            # created .npy files and decide whether to run multiround or
            # sequential clustering based on their count/size.
            return cluster(
                file_path,
                threshold=threshold,
                fp_type=fp_type,
                n_bits=n_bits,
                branching_factor=branching_factor,
                merge_criterion=merge_criterion,
                recluster_iterations=recluster_iterations,
                recluster_extra_threshold=recluster_extra_threshold,
                verbose=verbose,
                force_sequential=force_sequential,
                save_tree=save_tree,
                save_centroids=save_centroids,
            )
        else:
            raise ValueError(f"No .npy or .smi files found in directory: {file_path}")
    else:
        raise ValueError(f"Invalid input path: {file_path}")


def _save_clusters(output_dir: Path, cluster_ids, verbose: bool = VERBOSE) -> Path:
    """Persist sequential clustering output to clusters.pkl in output_dir."""
    output_path = Path(output_dir) / "clusters.pkl"
    with open(output_path, "wb") as handle:
        pickle.dump(cluster_ids, handle)
    if verbose:
        print(f"Saved clustering output to {output_path}")
    else:
        print(f"Saved clustering output to {output_path}")
    return output_path

def _cluster_from_npy_file(file_path: Path,
            threshold: float = None,
            branching_factor: int = BRANCHING_FACTOR,
            merge_criterion: str = MERGE_CRITERION,
            recluster_iterations: int = RECLUSTERING_ITERATIONS_INITIAL,
            recluster_extra_threshold: float = RECLUSTERING_EXTRA_THRESHOLD,
            save_tree: bool = SAVE_TREE,
            save_centroids: bool = SAVE_CENTROIDS,
            verbose: bool = VERBOSE):
    """Cluster the molecules from a single .npy file"""
    if threshold is None:
        if verbose:
            print("Determining optimal threshold...")
        threshold = optimal_threshold(np.load(file_path, mmap_mode='r'), factor=3.5)
        if verbose:
            print(f"Optimal threshold determined: {threshold:.4f}")

    # Create the BitBirch instance
    bb_object = BitBirch(
        merge_criterion=merge_criterion,
        threshold=threshold,
        branching_factor=branching_factor,
    )
    # Fit the fingerprints into the BitBirch model
    bb_object.fit(file_path)
    # Recluster to decrease the number of clusters
    bb_object.recluster_inplace(
        iterations=recluster_iterations,
        extra_threshold=recluster_extra_threshold,
        verbose=verbose,
    )

    if save_tree:
        bb_object.save(file_path.parent / "bitbirch.pkl")
        if verbose:
            print(f"Saved BitBirch tree to {file_path.parent / 'bitbirch.pkl'}")

    if save_centroids:
        output = bb_object.get_centroids_mol_ids()
        with open(file_path.parent / "clusters.pkl", mode="wb") as f:
            pickle.dump(output["mol_ids"], f)
        with open(file_path.parent / "cluster-centroids-packed.pkl", mode="wb") as f:
            pickle.dump(output["centroids"], f)
        if verbose:
            print(f"Saved clusters to {file_path.parent / 'clusters.pkl'}")
            print(
                f"Saved packed centroids to {file_path.parent / 'cluster-centroids-packed.pkl'}"
            )
        return output["mol_ids"]

    cluster_ids = bb_object.get_cluster_mol_ids()
    _save_clusters(file_path.parent, cluster_ids, verbose)
    return cluster_ids

def _prepare_fps_single(file_path: Path,
            fp_type: str = FINGERPRINT_TYPE,
            n_bits: int = N_BITS,
            verbose: bool = VERBOSE):
    """Prepare fingerprints from a single .smi or .smi.gz file"""
    # Check if the file is gzipped    
    if file_path.suffix == '.gz':
        smiles = load_smiles_gzipped(file_path)
    else:
        smiles = load_smiles(file_path)

    # Generate the fingerprints
    fps, invalid_ids = binary_fps(
        smiles,
        fp_type=fp_type,
        n_bits=n_bits,
        packed=True,
        return_invalid=True,
    )

    # Write the fingerprints to a .npy file
    npy_file_path = _npy_path_for(file_path)
    np.save(npy_file_path, fps)

    # Drop the invalid smiles and rewrite the .smi
    if len(invalid_ids) > 0:
        if verbose:
            print(f"Warning: {len(invalid_ids)} invalid SMILES were skipped.")
            print(f"Rewriting the .smi file with only valid SMILES.")
        valid_smiles = [smi for i, smi in enumerate(smiles) if i not in invalid_ids]
        if file_path.suffix == '.gz':
            corrected_file_path = file_path.with_name(
                file_path.name.replace('.smi.gz', '_valid.smi')
            )
        else:
            corrected_file_path = file_path.with_name(file_path.stem + '_valid.smi')

        # Rewrite the .smi file with only valid smiles
        with open(corrected_file_path, 'w') as f:
            for smi in valid_smiles:
                f.write(f"{smi}\n")

    return npy_file_path

def _cluster_multiple_npy_sequential(npy_files,
            threshold: float = None,
            branching_factor: int = BRANCHING_FACTOR,
            merge_criterion: str = MERGE_CRITERION,
            recluster_iterations: int = RECLUSTERING_ITERATIONS_INITIAL,
            recluster_extra_threshold: float = RECLUSTERING_EXTRA_THRESHOLD,
            output_dir: Path | None = None,
            save_tree: bool = SAVE_TREE,
            save_centroids: bool = SAVE_CENTROIDS,
            verbose: bool = VERBOSE):
    """Cluster the molecules from a list of .npy files sequentially.

    `npy_files` may be a directory `Path` (in which case all `*.npy` files
    will be read) or an iterable of `Path` objects.
    """
    # Normalize input to a sorted list of Paths
    if isinstance(npy_files, (list, tuple)):
        files = [Path(p) for p in npy_files]
    else:
        files = sorted(Path(npy_files).glob('*.npy'))

    if len(files) == 0:
        raise ValueError("No .npy files provided to _cluster_multiple_npy_sequential")

    print(f"Clustering {len(files)} .npy files sequentially.")

    # If no threshold provided, estimate it from the first file and use for all
    if threshold is None:
        if verbose:
            print(f"Estimating optimal threshold from first file: {files[0]}")
        threshold = optimal_threshold(np.load(files[0], mmap_mode='r'), factor=3.5)
        if verbose:
            print(f"Estimated threshold: {threshold:.4f} (applied to all files)")

    # Create the BitBirch instance with the fixed threshold
    bb_object = BitBirch(
        merge_criterion=merge_criterion,
        threshold=threshold,
        branching_factor=branching_factor,
    )

    for k, npy_file in enumerate(files):
        if verbose:
            print(f"Processing file {k+1}/{len(files)}: {npy_file}")
        bb_object.fit(npy_file)

    # Recluster to decrease the number of clusters
    bb_object.recluster_inplace(
        iterations=recluster_iterations,
        extra_threshold=recluster_extra_threshold,
        verbose=verbose,
    )

    resolved_output_dir = Path(output_dir) if output_dir is not None else files[0].parent

    if save_tree:
        bb_object.save(resolved_output_dir / "bitbirch.pkl")
        if verbose:
            print(f"Saved BitBirch tree to {resolved_output_dir / 'bitbirch.pkl'}")

    if save_centroids:
        output = bb_object.get_centroids_mol_ids()
        with open(resolved_output_dir / "clusters.pkl", mode="wb") as f:
            pickle.dump(output["mol_ids"], f)
        with open(resolved_output_dir / "cluster-centroids-packed.pkl", mode="wb") as f:
            pickle.dump(output["centroids"], f)
        if verbose:
            print(f"Saved clusters to {resolved_output_dir / 'clusters.pkl'}")
            print(
                f"Saved packed centroids to {resolved_output_dir / 'cluster-centroids-packed.pkl'}"
            )
        return output["mol_ids"]

    cluster_ids = bb_object.get_cluster_mol_ids()
    _save_clusters(resolved_output_dir, cluster_ids, verbose)
    return cluster_ids

def _prepare_fps_directory(dir_path: Path,
            fp_type: str = FINGERPRINT_TYPE,
            n_bits: int = N_BITS,
            verbose: bool = VERBOSE):
    """Prepare fingerprints for all .smi files in a directory"""
    print(f"Preparing fingerprints for all .smi files in directory: {dir_path}")
    print("WARNING: If you datasets are too big this might take a while.")

    # Find all .smi files in the directory
    smi_files = sorted(dir_path.glob('*.smi')) + sorted(dir_path.glob('*.smi.gz'))

    npy_paths = []
    for k, smi_file in enumerate(smi_files):
        if verbose:
            print(f"Processing file {k+1}/{len(smi_files)}: {smi_file}")
        npy_path = _prepare_fps_single(smi_file, fp_type, n_bits, verbose)
        npy_paths.append(npy_path)
    return npy_paths
