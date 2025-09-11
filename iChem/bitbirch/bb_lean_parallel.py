# type: ignore
# Model script to run parallel BitBIRCH calculations
#
# Please, cite the BitBIRCH paper: https://doi.org/10.1039/D5DD00030K
#
# BitBIRCH is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This code is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# License: GPL-3.0 https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Authors: Ramon Alain Miranda Quintana <ramirandaq@gmail.com>, <quintana@chem.ufl.edu>
#          Ignacio Pickering <ipickering@chem.ufl.edu>

import sys
import argparse
import shutil
from pathlib import Path
import typing as tp
from functools import partial
import pickle as pkl
import gc
import re
import time
import multiprocessing as mp

import numpy as np
from numpy.typing import NDArray

from bb_lean_modified import BitBirch, set_merge

try:
    import resource
except Exception:
    # non-unix systems
    pass


# requires psutil
def monitor_total_rss(file: Path | str, interval_s: float = 0.001) -> None:
    import psutil

    def total_rss() -> float:
        total_rss = 0.0
        for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
            info = proc.info
            cmdline = info["cmdline"]
            if cmdline is None:
                continue
            if Path(__file__).name in cmdline:
                total_rss += info["memory_info"].rss
        return total_rss

    while True:
        total_rss_gb = total_rss() / 1024**3
        with open(file, mode="a", encoding="utf-8") as f:
            f.write(f"{total_rss_gb}\n")
        time.sleep(interval_s)


def print_peak_mem(num_processes: int) -> None:
    if "resource" not in sys.modules:
        print("[Peak memory usage only tracked in Unix systems]")
    max_mem_bytes_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    max_mem_bytes_child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "linux":
        # In linux these are kiB, not bytes
        max_mem_bytes_self *= 1024
        max_mem_bytes_child *= 1024
    print(
        "Peak memory usage until now:\n"
        f"    Main proc.: {max_mem_bytes_self / 1024 ** 3:.4f} GiB"
    )
    if num_processes > 1:
        if mp.get_start_method() == "forkserver":
            print("    Max of child procs.: not tracked for 'forkserver'")
        else:
            print(
                f"    Max of child procs.: {max_mem_bytes_child / 1024 ** 3:.4f} GiB\n"
            )  # noqa:E501


def iterate_in_steps(lst: list[str], bin_size: int) -> tp.Iterator[list[str]]:
    for i in range(0, len(lst), bin_size):
        slice_ = slice(i, i + bin_size)
        yield lst[slice_]


def glob_and_sort_by_uint_bits(path: Path | str, glob: str) -> list[str]:
    path = Path(path)
    return sorted(
        map(str, path.glob(glob)),
        key=lambda name: int((re.search(r"uint(\d+)", name) or [0, 0])[1]),
        reverse=True,
    )


def load_fp_data_and_mol_idxs(
    out_dir: Path,
    fp_filename: str,
    round: int,
    mmap: bool = True,
) -> tuple[NDArray[tp.Any], tp.Any]:
    fp_data = np.load(fp_filename, mmap_mode="r" if mmap else None)
    # Infer the mol_idxs filename from the fp_np filename, and fetch the mol_idxs
    count, dtype = fp_filename.split(".")[0].split("_")[-2:]
    idxs_file = out_dir / f"mol_idxs_round{round}_{count}_{dtype}.pkl"
    with open(idxs_file, "rb") as f:
        mol_idxs = pkl.load(f)
    return fp_data, mol_idxs


def validate_output_dir(out_dir: Path, overwrite_outputs: bool = False) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise RuntimeError("Output dir should be a dir")
        if any(out_dir.iterdir()):
            if overwrite_outputs:
                shutil.rmtree(out_dir)
            else:
                raise RuntimeError(f"Output dir {out_dir} has files")
    out_dir.mkdir(exist_ok=True)


# Validate also that the naming convention for the input files is correct
def validate_input_dir(
    in_dir: Path | str, filename_idxs_are_slices: bool = False
) -> None:
    in_dir = Path(in_dir)
    if not in_dir.is_dir():
        raise RuntimeError(f"Input dir {in_dir} should be a dir")
    if not any(in_dir.glob("*.npy")):
        raise RuntimeError(f"Input dir {in_dir} should have *.npy fingerprint files")

    # TODO: There is currently no validation regarding fp sizes and stride sizes
    if filename_idxs_are_slices:
        return

    _file_idxs = []
    for f in in_dir.glob("*.npy"):
        matches = re.match(r".*_(\d+)_(\d+).npy", f.name)
        if matches is None:
            raise RuntimeError(f"Input file {str(f)} doesn't fit name convention")
        _file_idxs.append(int(matches[1]))

    # Sort arrays
    sort_idxs = np.argsort(_file_idxs)
    file_idxs = np.array(_file_idxs)[sort_idxs]

    if not (file_idxs == np.arange(len(file_idxs))).all():
        raise RuntimeError(f"Input file indices {file_idxs} must be a seq 0, 1, 2, ...")


def process_fp_files(
    fp_file: str,
    double_cluster_init: bool,
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    return_fp_lists: bool = False,
    filename_idxs_are_slices: bool = False,
    max_fps: int | None = None,
    mmap: bool = True,
) -> None:
    out_dir = Path(out_dir)
    fps = np.load(fp_file, mmap_mode="r" if mmap else None)[:max_fps]

    # Fit the fps. fit_reinsert is necessary to keep track of proper molecule indices
    # Use indices of molecules in the current batch, according to the total set
    idx0, idx1 = map(int, fp_file.split(".")[0].split("_")[-2:])
    set_merge("diameter")  # Initial batch uses diameter BitBIRCH
    brc_diameter = BitBirch(branching_factor=branching_factor, threshold=threshold)
    if filename_idxs_are_slices:
        # idxs are <start_mol_idx>_<end_mol_idx>
        range_ = range(idx0, idx1)
        start_mol_idx = idx0
    else:
        # idxs are <file_idx>_<start_mol_idx>
        range_ = range(idx1, idx1 + len(fps))
        start_mol_idx = idx1
    brc_diameter.fit_reinsert(fps, list(range_))

    # Extract the BitFeatures info of the leaves to refine the top cluster
    fps_bfs, mols_bfs = brc_diameter.bf_to_np_refine(
        fps, initial_mol=start_mol_idx, return_fp_lists=return_fp_lists
    )
    del fps
    del brc_diameter
    gc.collect()

    if double_cluster_init:
        # Passing the previous BitFeatures through the new tree, singleton clusters are
        # passed at the end
        set_merge("tolerance", tolerance)  # 'tolerance' used to refine the tree
        brc_tolerance = BitBirch(branching_factor=branching_factor, threshold=threshold)
        for fp_type, mol_idxs in zip(fps_bfs, mols_bfs):
            brc_tolerance.fit_np_reinsert(fp_type, mol_idxs)

        # Get the info from the fitted BFs in compact list format
        fps_bfs, mols_bfs = brc_tolerance.bf_to_np(return_fp_lists)
        del brc_tolerance
        gc.collect()

    if return_fp_lists:
        numpy_streaming_save(fps_bfs, out_dir / f"round1_{idx0}")
        for fp_type, mol_idxs in zip(fps_bfs, mols_bfs):
            suffix = f"round1_{idx0}_{str(fp_type[0].dtype)}"
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pkl.dump(mol_idxs, f)
    else:
        for fp_type, mol_idxs in zip(fps_bfs, mols_bfs):
            suffix = f"round1_{idx0}_{str(fp_type.dtype)}"
            np.save(out_dir / f"fp_{suffix}", fp_type)
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pkl.dump(mol_idxs, f)


# Save a list of numpy arrays into a single array in a streaming fashion, avoiding
# stacking them in memory
def numpy_streaming_save(
    fps_bfs: list[list[NDArray[tp.Any]]], path: Path | str
) -> None:
    path = Path(path)
    for fp_list in fps_bfs:
        first_arr = np.ascontiguousarray(fp_list[0])
        header = np.lib.format.header_data_from_array_1_0(first_arr)
        header["shape"] = (len(fp_list), len(first_arr))
        with open(path.with_name(f"{path.name}_{first_arr.dtype.name}.npy"), "wb") as f:
            np.lib.format.write_array_header_1_0(f, header)
            for arr in fp_list:
                np.ascontiguousarray(arr).tofile(f)


def second_round(
    chunk_info: tuple[int, list[str]],
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    return_fp_lists: bool = False,
    mmap: bool = True,
) -> None:
    out_dir = Path(out_dir)
    chunk_idx, chunk_filenames = chunk_info

    set_merge("tolerance", tolerance)
    brc_chunk = BitBirch(branching_factor=branching_factor, threshold=threshold)
    for fp_filename in chunk_filenames:
        fp_data, mol_idxs = load_fp_data_and_mol_idxs(out_dir, fp_filename, 1, mmap)
        brc_chunk.fit_np_reinsert(fp_data, mol_idxs)
        del mol_idxs
        del fp_data
        gc.collect()

    fps_bfs, mols_bfs = brc_chunk.bf_to_np(return_fp_lists)
    del brc_chunk
    gc.collect()

    if return_fp_lists:
        numpy_streaming_save(fps_bfs, out_dir / f"round2_{chunk_idx}")
        for fp_type, mol_idxs in zip(fps_bfs, mols_bfs):
            suffix = f"round2_{chunk_idx}_{str(fp_type[0].dtype)}"
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pkl.dump(mol_idxs, f)
    else:
        for fp_type, mol_idxs in zip(fps_bfs, mols_bfs):
            suffix = f"round2_{chunk_idx}_{str(fp_type.dtype)}"
            np.save(out_dir / f"fp_{suffix}", fp_type)
            with open(out_dir / f"mol_idxs_{suffix}.pkl", mode="wb") as f:
                pkl.dump(mol_idxs, f)


def final_clustering(
    branching_factor: int,
    threshold: float,
    tolerance: float,
    out_dir: Path | str,
    mmap: bool = True,
) -> None:
    out_dir = Path(out_dir)

    set_merge("tolerance", tolerance)
    brc_final = BitBirch(branching_factor=branching_factor, threshold=threshold)

    sorted_files2 = glob_and_sort_by_uint_bits(out_dir, "*round2*.npy")
    for fp_filename in sorted_files2:
        fp_data, mol_idxs = load_fp_data_and_mol_idxs(out_dir, fp_filename, 2, mmap)
        brc_final.fit_np_reinsert(fp_data, mol_idxs)
        del fp_data
        del mol_idxs
        gc.collect()

    mol_ids = brc_final.get_cluster_mol_ids().copy()
    del brc_final
    gc.collect()

    with open(out_dir / "clusters.pkl", mode="wb") as f:
        pkl.dump(mol_ids, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="bitbirch-parallel")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "bb_parallel_outputs",
        help="Dir for output files",
    )
    parser.add_argument(
        "--idxs-are-slices",
        dest="filename_idxs_are_slices",
        action="store_true",
        help="Use slices as filename idxs, e.g. *_1000_2000.npy, *_2000_3000.npy, ...",
    )
    parser.add_argument(
        "--no-overwrite-outputs",
        dest="overwrite_outputs",
        action="store_false",
        help="Disallow overwriting outputs",
    )
    parser.add_argument(
        "--only-first-round",
        action="store_true",
        help="Only do first round clustering and exit early",
    )
    parser.add_argument("--num-processes", type=int, default=10, help="Num. processes")
    parser.add_argument(
        "--bin-size", type=int, default=10, help="Bin size for chunking"
    )
    parser.add_argument(
        "--branching-factor", type=int, default=50, help="BitBIRCH branching factor"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.65, help="BitBIRCH threshold"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.05, help="BitBIRCH tolerance"
    )
    parser.add_argument("--no-return-fp-lists", action="store_false")
    parser.add_argument(
        "--max-fps",
        type=int,
        default=None,
        help="Max num. of fps to load from each input file",
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="Max num. files to read"
    )
    parser.add_argument(
        "--fork",
        action="store_true",
        help="Use 'fork' method instead of 'forkserver' (only has an effect on linux)",
    )
    parser.add_argument(
        "--no-mmap",
        dest="mmap",
        action="store_false",
        help="Don't mmap the fingerprint files",
    )
    parser.add_argument(
        "--monitor-rss",
        action="store_true",
        help="Monitor the total memory used by all processes (requires psutil)",
    )
    # The fps should be stored as .npy files
    # In this new version, the fps can be passed as uint8
    #
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=Path(__file__).parent / "bb_parallel_inputs",
        help=(
            "Dir with input *.npy files"
            " Files must have the format:"
            " *_0_0.npy, *_1_1000.npy, *_2_2000.npy, *_3_3000.npy ..."
            " Where the first idx is the 'file idx' and the second idx is the"
            " 'starting molecule idx'"
        ),
    )
    # 'double_cluster_init' indicates if the refinement of the batches is done before or
    # after combining all the data in the final tree
    # False: potentially slightly faster, but splits the biggest cluster of each batch
    #        and doesn't try to re-form it until all the data goes through the final
    #        tree.
    # True:  re-fits the splitted cluster in a new tree using tolerance merge
    #        this adds a bit of time and memory overhead, so depending on the volume of
    #        data in each batch it might need to be skipped, but this is a more
    #        solid/robust choice
    parser.add_argument(
        "--no-double-cluster-init",
        dest="double_cluster_init",
        action="store_false",
        help="Don't perform double-cluster-init",
    )
    parser.set_defaults(
        monitor_rss=False,
        mmap=True,
        overwrite_outputs=True,
        double_cluster_init=True,
        only_first_round=False,
        filename_idxs_are_slices=False,
        return_fp_lists=True,
    )
    args = parser.parse_args()
    # Set the multiprocessing start method
    if sys.platform == "linux":
        mp.set_start_method("fork" if args.fork else "forkserver")

    if args.monitor_rss:
        mp.Process(
            target=monitor_total_rss,
            kwargs=dict(file=Path.cwd() / "monitor-rss.csv", interval_s=0.001),
            daemon=True,
        ).start()

    # BitBIRCH parameters
    common_kwargs: dict[str, tp.Any] = dict(
        branching_factor=args.branching_factor,
        threshold=args.threshold,
        tolerance=args.tolerance,
        mmap=args.mmap,
    )
    # Input and output dirs
    common_kwargs["out_dir"] = args.out_dir
    validate_output_dir(common_kwargs["out_dir"], args.overwrite_outputs)
    validate_input_dir(args.in_dir, args.filename_idxs_are_slices)
    input_files = sorted(
        map(str, args.in_dir.glob("*.npy")),
        key=lambda x: int(x.split(".")[0].split("_")[-2]),
    )[: args.max_files]

    start_all = time.perf_counter()
    start_round1 = time.perf_counter()
    _start_msg = (
        f"parallel ({args.num_processes} processes)"
        if args.num_processes > 1
        else "serial (1 process)"
    )
    print(
        f"Starting {_start_msg} BitBIRCH\n\n"
        f"- Max files to load: {args.max_files}\n"
        f"- Actual files loaded: {len(input_files)}\n"
        f"- Max fingerprints loaded per file: {args.max_fps}\n"
        f"- Return fp lists memory optimization: {args.return_fp_lists}\n",
        end="",
    )
    if args.num_processes > 1:
        print(f"- Multiprocessing start method: {mp.get_start_method()}\n")
    else:
        print()

    print("Processing initial batch of packed fingerprints...")
    round_1_fn: tp.Callable[[str], None] = partial(
        process_fp_files,
        double_cluster_init=args.double_cluster_init,
        max_fps=args.max_fps,
        filename_idxs_are_slices=args.filename_idxs_are_slices,
        return_fp_lists=args.return_fp_lists,
        **common_kwargs,
    )
    if args.num_processes == 1:
        for file in input_files:
            round_1_fn(file)
    else:
        with mp.Pool(processes=args.num_processes, maxtasksperchild=1) as pool:
            pool.map(round_1_fn, input_files)
    sorted_files1 = glob_and_sort_by_uint_bits(common_kwargs["out_dir"], "*round1*.npy")
    chunk_info = list(
        enumerate(iterate_in_steps(sorted_files1, bin_size=args.bin_size))
    )
    print(
        f"Finished. Collected {len(sorted_files1)} files,"
        f" chunked into {len(chunk_info)} chunks"
    )
    print(f"Time for round 1: {time.perf_counter() - start_round1:.4f} s", flush=True)
    print_peak_mem(args.num_processes)

    if not args.only_first_round:
        start_round2 = time.perf_counter()
        print("Processing second round of clustering...")
        round_2_fn: tp.Callable[[tuple[int, list[str]]], None] = partial(
            second_round, return_fp_lists=args.return_fp_lists, **common_kwargs
        )
        if args.num_processes == 1:
            for chunk_info_part in chunk_info:
                round_2_fn(chunk_info_part)
        else:
            with mp.Pool(processes=args.num_processes, maxtasksperchild=1) as pool:
                pool.map(round_2_fn, chunk_info)
        print(
            f"Time for round 2: {time.perf_counter() - start_round2:.4f} s", flush=True
        )
        print_peak_mem(args.num_processes)

        start_final = time.perf_counter()
        print("Performing final clustering...")
        final_clustering(**common_kwargs)
        print(
            f"Time for final clustering: {time.perf_counter() - start_final:.4f} s",
            flush=True,
        )
        print_peak_mem(args.num_processes)

    print(f"Total time: {time.perf_counter() - start_all:.4f} s", flush=True)
