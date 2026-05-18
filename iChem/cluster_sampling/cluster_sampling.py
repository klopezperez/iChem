import numpy as np
import pickle as pkl
from pathlib import Path
from bblean.similarity import jt_isim_medoid, jt_sim_packed
from ..utils.utils import binary_fps, load_smiles, load_smiles_gzipped

def _singletons_sampling(clusters: list[list[int]],
                         fps: np.ndarray = None,
                         smiles: list = None):
    """Sample all the singletons in a clustering."""
    singleton_clusters = [cluster for cluster in clusters if len(cluster) == 1]
    singletons_list = [cluster[0] for cluster in singleton_clusters]
    if fps is not None and smiles is not None:
        return fps[singletons_list], [smiles[idx] for idx in singletons_list]
    elif fps is not None:
        return fps[singletons_list]
    elif smiles is not None:
        return [smiles[idx] for idx in singletons_list]
    else:
        return singletons_list
def _medoids_sampling(clusters: list[list[int]],
                      fps: np.ndarray = None,
                      smiles: list = None,
                      min_size: int = 0,
                      fp_type: str = 'ECFP4',
                      n_bits: int = 2048,
                      sample: bool = True,
                      sample_min_size: int = 1_000):
    """Sample the medoid of each cluster """
    if fps is not None:
        medoids_list = []
        for cluster in clusters:
            if len(cluster) >= min_size:
                if sample:
                    if len(cluster) > sample_min_size:
                        random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                    else:
                        random_sample = np.array(cluster)
                    cluster_fps = fps[random_sample]
                    medoid_idx, _ = jt_isim_medoid(cluster_fps)
                    medoids_list.append(random_sample[medoid_idx])
                else:
                    cluster_fps = fps[cluster]
                    medoid_idx, _ = jt_isim_medoid(fps[cluster])
                    medoids_list.append(cluster[medoid_idx])
    else:
        medoids_list = []
        for cluster in clusters:
            if len(cluster) >= min_size:
                if sample:
                    if len(cluster) > sample_min_size:
                        random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                    else:
                        random_sample = np.array(cluster)
                    cluster_smiles = [smiles[idx] for idx in random_sample] if smiles else None
                    cluster_fps = binary_fps(cluster_smiles, fp_type=fp_type, n_bits=n_bits)
                    medoid_idx, _ = jt_isim_medoid(cluster_fps)
                    medoids_list.append(random_sample[medoid_idx])
                else:
                    cluster_smiles = [smiles[idx] for idx in cluster] if smiles else None
                    cluster_fps = binary_fps(cluster_smiles, fp_type=fp_type, n_bits=n_bits)
                    medoid_idx, _ = jt_isim_medoid(cluster_fps)
                    medoids_list.append(cluster[medoid_idx])
    if smiles is not None:
        return [smiles[idx] for idx in medoids_list]
    else:
        return medoids_list


def _centroid_like_sampling(clusters: list[list[int]],
                           centroids: np.ndarray,
                           fps: np.ndarray = None,
                           smiles: list = None,
                           fp_type: str = 'ECFP4',
                           n_bits: int = 2048,
                           sample: bool = True,
                           sample_min_size: int = 1_000):
    """Sample the centroid-like of each cluster """
    if centroids is None:
        raise ValueError("Centroids are required for centroid-like sampling.")
    centroid_like_list = []
    for cluster, centroid in zip(clusters, centroids):
        if fps is not None:
            if sample:
                if len(cluster) > sample_min_size:
                    random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                else:
                    random_sample = np.array(cluster)
                cluster_fps = fps[random_sample]
                similarities = jt_sim_packed(centroid, cluster_fps)
                closest_idx = np.argmax(similarities)
                centroid_like_list.append(random_sample[closest_idx])
            else:
                cluster_fps = fps[cluster]
                similarities = jt_sim_packed(centroid, cluster_fps)
                closest_idx = np.argmax(similarities)
                centroid_like_list.append(cluster[closest_idx])
        else:
            if sample:
                if len(cluster) > sample_min_size:
                    random_sample = np.random.choice(cluster, size=sample_min_size, replace=False)
                else:
                    random_sample = np.array(cluster)
                cluster_smiles = [smiles[idx] for idx in random_sample] if smiles else None
                cluster_fps = binary_fps(cluster_smiles, fp_type=fp_type, n_bits=n_bits)
                similarities = jt_sim_packed(cluster_fps, centroid)
                closest_idx = np.argmax(similarities)
                centroid_like_list.append(random_sample[closest_idx])
            else:
                cluster_smiles = [smiles[idx] for idx in cluster] if smiles else None
                cluster_fps = binary_fps(cluster_smiles, fp_type=fp_type, n_bits=n_bits)
                similarities = jt_sim_packed(cluster_fps, centroid)
                closest_idx = np.argmax(similarities)
                centroid_like_list.append(cluster[closest_idx])
    if smiles:
        return [smiles[idx] for idx in centroid_like_list]
    else:
        return centroid_like_list


def sample_clusters(clusters,
                     sampling_method: str = 'medoids',
                     fps = None,
                     smiles = None,
                     centroids = None,
                     min_size: int = 0,
                     fp_type: str = 'ECFP4',
                     n_bits: int = 2048,
                     sample: bool = True,
                     sample_min_size: int = 1000,):
    """Sample molecules from precomputed clusters using various strategies.
    
    Parameters
    ----------
    clusters: str, Path, or list of lists
        Cluster assignments. Can be:
        - Path to pickled file containing clusters (list of lists)
        - Pre-loaded list of lists with cluster assignments
    sampling_method: str, default='medoids'
        Sampling strategy: 'singletons' (all single-element clusters), 
        'medoids' (most similar compound to cluster median), 
        or 'centroid-like' (most similar to precomputed centroid).
    fps: str, Path, or np.ndarray, optional
        Fingerprints data. Can be:
        - Path to .npy file or directory of .npy files
        - Pre-loaded numpy array (shape: n_molecules x n_bits)
    smiles: str, Path, or list, optional
        SMILES data. Can be:
        - Path to .smi or .smi.gz file or directory containing them
        - Pre-loaded list of SMILES strings
    centroids: str, Path, or np.ndarray, optional
        Centroid vectors. Can be:
        - Path to pickled file containing centroids
        - Pre-loaded numpy array (shape: n_clusters x n_bits)
        Required for 'centroid-like' sampling.
    min_size: int, default=0
        Minimum cluster size to sample from (for 'medoids' method).
    fp_type: str, default='ECFP4'
        Fingerprint type for computing fingerprints from SMILES.
    n_bits: int, default=2048
        Number of bits in fingerprint vectors.
    sample: bool, default=True
        If True, subsample large clusters before computing medoid/centroid similarity.
    sample_min_size: int, default=1000
        Subsample threshold - clusters larger than this are randomly subsampled.
    
    Returns
    -------
    list
        Sampled molecule indices or SMILES strings, depending on input data provided.
    """
    # Load clusters: either path or direct list
    if isinstance(clusters, list):
        pass  # Already a list
    else:
        clusters_path = Path(clusters)
        if clusters_path.is_file():
            clusters = pkl.load(open(clusters_path, 'rb'))
        else:
            raise FileNotFoundError(
                f"Clusters file not found: {clusters_path}"
            )

    # Handle fingerprints: either path or direct np.ndarray
    if isinstance(fps, np.ndarray):
        pass  # Already a numpy array
    elif fps is not None:
        fps_path = Path(fps)
        if fps_path.is_file():
            fps = np.load(fps_path, mmap_mode='r')
        elif fps_path.is_dir():
            fps_files = sorted(fps_path.glob("*.npy"))
            fps_list = [np.load(fp_file, mmap_mode='r') for fp_file in fps_files]
            fps = np.concatenate(fps_list, axis=0)
        else:
            fps = None
    else:
        fps = None

    # Handle smiles: either path or direct list
    if isinstance(smiles, list):
        pass  # Already a list
    elif smiles is not None:
        smiles_path = Path(smiles)
        if smiles_path.is_file():
            if smiles_path.suffix == '.smi':
                smiles = load_smiles(smiles_path)
            elif smiles_path.suffix == '.smi.gz':
                smiles = load_smiles_gzipped(smiles_path)
            else:
                smiles = None
        elif smiles_path.is_dir():
            smiles_files_smi = sorted(smiles_path.glob("*.smi"))
            smiles_files_gz = sorted(smiles_path.glob("*.smi.gz"))
            smiles = []
            for smi_file in smiles_files_smi:
                smiles.extend(load_smiles(smi_file))
            for smi_file in smiles_files_gz:
                smiles.extend(load_smiles_gzipped(smi_file))
        else:
            smiles = None
    else:
        smiles = None

    # Handle centroids: either path or direct np.ndarray
    if isinstance(centroids, np.ndarray):
        pass  # Already a numpy array
    elif centroids is not None:
        centroids_path = Path(centroids)
        if centroids_path.is_file():
            centroids = pkl.load(open(centroids_path, 'rb'))
        else:
            centroids = None
    else:
        centroids = None

    # Perform sampling
    if sampling_method == 'singletons':
        return _singletons_sampling(clusters,
                                    fps,
                                    smiles)
    elif sampling_method == 'medoids':
        return _medoids_sampling(clusters=clusters,
                                 fps=fps,
                                 smiles=smiles,
                                 min_size=min_size,
                                 fp_type=fp_type,
                                 n_bits=n_bits,
                                 sample=sample,
                                 sample_min_size=sample_min_size)
    elif sampling_method == 'centroid-like':
        if centroids is None:
            raise ValueError(
                "Centroids must be provided for centroid-like sampling."
            )
        return _centroid_like_sampling(clusters=clusters,
                                       centroids=centroids,
                                       fps=fps,
                                       smiles=smiles,
                                       fp_type=fp_type,
                                       n_bits=n_bits,
                                       sample=sample,
                                       sample_min_size=sample_min_size)