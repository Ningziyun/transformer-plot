#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 20:30:51 2025

@author: ningyan
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import os
import re

import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# Argument parsing
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a 2D histogram from ROOT TTrees, one plot per file, with optional normalization. "
                    "Automatically picks the first two array-like branches per file and interprets them "
                    "according to --first_kind/--second_kind."
    )
    # I/O
    parser.add_argument("--file_list", type=str, default="fileList.txt",
                        help="Text file listing ROOT files (one path per line).")
    parser.add_argument("--tree_name", type=str, default="auto",
                        help="TTree name; use 'auto' to detect per file (searches subdirectories).")

    # Semantic interpretation of the FIRST and SECOND selected branches
    #   - 'logDRinv' : already log(1/deltaR)
    #   - 'logkt'    : already log(kt)
    #   - 'deltaR'   : raw deltaR (will be converted to log(1/deltaR))
    #   - 'kt'       : raw kt (will be converted to log(kt))
    parser.add_argument("--first_kind", type=str, default="logDRinv",
                        choices=["logDRinv", "logkt", "deltaR", "kt"],
                        help="Semantic of the FIRST picked branch (default: logDRinv).")
    parser.add_argument("--second_kind", type=str, default="logkt",
                        choices=["logDRinv", "logkt", "deltaR", "kt"],
                        help="Semantic of the SECOND picked branch (default: logkt).")

    # Output
    parser.add_argument("--out_dir", type=str, default="plots",
                        help="Directory to save outputs.")
    parser.add_argument("--out_tag", type=str, default="lund2d",
                        help="Output filename prefix. Final filename is <out_tag>_<file-stem>.(png|npz).")

    # Binning & ranges
    parser.add_argument("--xbins", type=int, default=50, help="Number of X bins.")
    parser.add_argument("--ybins", type=int, default=50, help="Number of Y bins.")
    parser.add_argument("--xrange", type=float, nargs=2, default=None,
                        help="X range: xmin xmax (default auto).")
    parser.add_argument("--yrange", type=float, nargs=2, default=None,
                        help="Y range: ymin ymax (default auto).")

    # Normalization mode
    parser.add_argument("--norm", type=str, default="per_jet",
                        choices=["none", "per_jet", "per_emission"],
                        help="Normalization: 'none' (raw counts), "
                             "'per_jet' (counts / #jets), or "
                             "'per_emission' (counts / #emissions in-range). "
                             "Default: per_jet.")

    # Z-axis (colorbar) range
    parser.add_argument("--zmin", type=float, default=0.0,
                        help="Colorbar lower bound. Default 0.")
    parser.add_argument("--zmax", type=float, default=None,
                        help="Colorbar upper bound. Default auto (max).")

    # Plotting
    parser.add_argument("--title", type=str, default="QCD Lund Plane",
                        help="First line of the plot title.")
    parser.add_argument("--title_extra", type=str, default="",
                        help="Optional note to append on the 3rd title line (e.g., 'epoch=10').")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Matplotlib colormap.")

    # New: per-jet emission truncation
    parser.add_argument("--max_emissions_per_jet", "--maxN", type=int, default=None,
                        help="If set (e.g. --maxN 20), truncate each jet's emissions to the FIRST N elements "
                             "before histogramming. Default: use full natural length (no truncation).")

    # Misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true",
                        help="Parse and report counts; do not produce plots.")
    parser.add_argument("--debug", action="store_true",
                        help="Print extra debug info about branches.")

    return parser.parse_args()


# =========================
# Utilities
# =========================
def read_file_list(path_txt: str) -> list:
    """Read a plain text file of filepaths (one per line, '#' for comments)."""
    p = Path(path_txt)
    if not p.exists():
        raise FileNotFoundError(f"file_list not found: {p}")
    files = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        files.append(s)
    if not files:
        raise RuntimeError(f"No valid ROOT file paths found in {p}")
    return files


def _collect_ttrees(node) -> list:
    """
    Recursively collect TTree paths in a File or TDirectory.
    Returns list of 'path/to/TreeName' without ';cycle'.
    """
    trees = []
    for k, cls in node.classnames().items():
        base = k.split(";")[0]
        obj = node[k]
        if cls == "TTree":
            trees.append(base)
        elif cls in ("TDirectoryFile", "TDirectory"):
            subt = _collect_ttrees(obj)
            trees.extend([f"{base}/{t}" for t in subt])
    return trees


def _pick_tree_for_file(uf, explicit_tree):
    """
    Decide which TTree to use for this file.
    - If explicit_tree != 'auto', try to open it (supports subdir path).
    - Else, return the first TTree found (depth-first).
    """
    if explicit_tree and explicit_tree != "auto":
        try:
            return uf[explicit_tree], explicit_tree
        except Exception:
            base = explicit_tree.split(";")[0]
            return uf[base], explicit_tree
    trees = _collect_ttrees(uf)
    if not trees:
        raise KeyError("No TTree found in file.")
    tpath = trees[0]
    return uf[tpath], tpath


def _is_array_like_branch(tree, bname: str, sample_events: int = 64) -> bool:
    """
    Practical check for numeric branches (works for jagged and flat):
    1) Read a small sample as awkward array (entry_stop=sample).
    2) Flatten to 1D; empty sample counts as acceptable (may be empty early events).
    3) Convert to numpy and check dtype.kind in {float,int,uint}.
    """
    try:
        n = getattr(tree, "num_entries", 0) or 0
        entry_stop = min(n, sample_events) if n > 0 else None
        arr = tree[bname].array(library="ak", entry_stop=entry_stop)
        flat = ak.flatten(arr, axis=None)
        if flat is None:
            return False
        if len(flat) == 0:
            return True
        np_arr = ak.to_numpy(flat)
        return np_arr.dtype is not None and np_arr.dtype.kind in ("f", "i", "u")
    except Exception:
        return False


def _debug_branch_signature(tree, limit=50) -> str:
    """Return a readable list of branch names and (if possible) types, for debugging."""
    items = []
    names = list(tree.keys())
    for b in names[:limit]:
        try:
            tname = tree[b].typename
        except Exception:
            tname = "<?>"
        items.append(f"{b} : {tname}")
    more = "" if len(names) <= limit else f" ... (+{len(names)-limit} more)"
    return "\n".join(items) + more


def _name_priority_score(name: str) -> int:
    """Heuristic priority for kt/deltaR-like names when we must guess."""
    s = name.lower()
    score = 0
    if re.search(r"\bkt\b", s) or "logkt" in s:
        score += 5
    if "deltar" in s or "delta" in s or re.search(r"\bdr\b", s):
        score += 5
    if "log" in s:
        score += 2
    if "lund" in s:
        score += 1
    return score


def _pick_first_two_array_branches(tree, debug=False) -> list:
    """
    Pick two numeric (flat or jagged) branches.
    Strategy:
      (A) Sample-based filter.
      (B) If <2, rank by name heuristics.
      (C) If still <2, raise with a diagnostic list.
    """
    names = list(tree.keys())
    good = [b for b in names if _is_array_like_branch(tree, b)]

    if len(good) >= 2:
        picked = good[:2]
        if debug:
            print(f"[debug] sample-picked branches: {picked}")
        return picked

    ranked = sorted(names, key=lambda n: _name_priority_score(n), reverse=True)
    picked = []
    for b in ranked:
        if _is_array_like_branch(tree, b):
            picked.append(b)
            if len(picked) == 2:
                if debug:
                    print(f"[debug] heuristic-picked branches: {picked}")
                return picked

    sig = _debug_branch_signature(tree)
    raise RuntimeError(
        "Could not find two array-like (numeric) branches to read.\n"
        "Here are some branches and types (first 50):\n"
        f"{sig}\n"
        "Tip: If your numeric data are stored as objects/strings, export numeric arrays; "
        "or pass an explicit --tree_name pointing to the intended TTree."
    )


def _read_two_branches_as_arrays(tree, b1, b2):
    """Read two branches as awkward arrays (may be jagged) and return them."""
    a1 = tree[b1].array(library="ak")
    a2 = tree[b2].array(library="ak")
    return a1, a2


def _truncate_per_jet(a: ak.Array, maxN: int) -> ak.Array:
    """
    Truncate each jet's emissions to the FIRST maxN elements *if and only if*
    the array has an inner list axis (per-jet list of emissions).

    Implementation details:
    - We probe list-ness by calling `ak.num(a, axis=1)`. If this raises, there is no inner list.
    - If there IS an inner list axis, simply slice `a[:, :maxN]`.
    - If not list-like, return unchanged (no-op).
    """
    if maxN is None:
        return a
    if maxN <= 0:
        raise ValueError("--max_emissions_per_jet must be positive if provided.")
    try:
        # This will fail if `a` has no inner list axis.
        _ = ak.num(a, axis=1)
    except Exception:
        return a  # not a per-jet list -> no truncation possible
    # Safe: truncate along the inner (emission) axis
    return a[:, :maxN]


def _interpret_and_to_logs(first_flat, second_flat, first_kind, second_kind):
    """
    Map flattened arrays to (X, Y) where:
      X := log(1/deltaR)
      Y := log(kt)
    """
    X = None
    Y = None
    eps = 1e-12

    def to_logDRinv(x):
        x = np.clip(x, eps, None)
        return np.log(1.0 / x)

    def to_logkt(y):
        y = np.clip(y, eps, None)
        return np.log(y)

    # FIRST maps to either X or Y depending on declared kind
    if first_kind == "logDRinv":
        X = first_flat
    elif first_kind == "logkt":
        Y = first_flat
    elif first_kind == "deltaR":
        X = to_logDRinv(first_flat)
    elif first_kind == "kt":
        Y = to_logkt(first_flat)

    # SECOND fills the remaining one
    if second_kind == "logDRinv":
        X = second_flat if X is None else X
    elif second_kind == "logkt":
        Y = second_flat if Y is None else Y
    elif second_kind == "deltaR":
        X = to_logDRinv(second_flat) if X is None else X
    elif second_kind == "kt":
        Y = to_logkt(second_flat) if Y is None else Y

    if X is None or Y is None:
        raise ValueError("Ambiguous kinds: could not derive both X=log(1/ΔR) and Y=log(kt). "
                         f"Got first_kind={first_kind}, second_kind={second_kind}")
    return X, Y


# =========================
# Plotting
# =========================
def plot_2d_hist(
    x, y, n_jets, xbins, ybins, xrange_, yrange_,
    title, cmap, out_png, out_npz, norm_mode="per_jet",
    zmin=0.0, zmax=None
):
    """
    Build a 2D histogram H(x,y). Apply normalization according to norm_mode:
      - 'none'        : H stays as raw counts.
      - 'per_jet'     : H /= n_jets                  (counts per jet).
      - 'per_emission': H /= H.sum() (in-range only) (probability per emission).
    Save PNG and NPZ (edges + normalized H).
    Returns (sum_of_H_after_norm, raw_total_emissions_in_range).
    """
    if norm_mode == "per_jet" and n_jets <= 0:
        raise ValueError("n_jets must be positive to normalize per jet.")

    H, xedges, yedges = np.histogram2d(
        x, y,
        bins=[xbins, ybins],
        range=[xrange_, yrange_] if (xrange_ is not None and yrange_ is not None) else None
    )
    raw_emissions_in_range = H.sum()

    # Normalization
    if norm_mode == "per_jet":
        Hn = H / float(n_jets)
        zlabel = "counts per jet"
    elif norm_mode == "per_emission":
        denom = raw_emissions_in_range
        Hn = H if denom == 0 else H / float(denom)
        zlabel = "probability per emission"
    else:
        Hn = H
        zlabel = "counts"

    # Plot
    plt.figure(figsize=(7.8, 6.6))
    mesh = plt.pcolormesh(
        xedges, yedges, Hn.T,
        shading="auto", cmap=cmap, vmin=zmin, vmax=zmax
    )
    cbar = plt.colorbar(mesh)
    cbar.set_label(zlabel)

    plt.xlabel("log(1/ΔR)")
    plt.ylabel("log(kt)")
    plt.title(title)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # Save arrays
    np.savez(out_npz, H=Hn, xedges=xedges, yedges=yedges,
             norm_mode=norm_mode, raw_emissions_in_range=raw_emissions_in_range)

    return Hn.sum(), raw_emissions_in_range


# =========================
# Main (one plot per file)
# =========================
def main():
    args = parse_args()
    np.random.seed(args.seed)

    files = read_file_list(args.file_list)
    print(f"[files] {len(files)} files from {args.file_list}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over files and produce one plot per file
    for f in tqdm(files, desc="Processing files"):
        fpath = Path(f)
        stem = fpath.stem  # filename without suffix
        out_tag = f"{args.out_tag}_{stem}" if args.out_tag else f"lund2d_{stem}"
        out_png = str(out_dir / f"{out_tag}.png")
        out_npz = str(out_dir / f"{out_tag}.npz")

        with uproot.open(f) as uf:
            tree, tpath = _pick_tree_for_file(uf, args.tree_name)

            if args.debug:
                print(f"[debug] Branch summary for {fpath.name} (tree='{tpath}'):\n"
                      f"{_debug_branch_signature(tree)}")

            # Pick two numeric branches
            b1, b2 = _pick_first_two_array_branches(tree, debug=args.debug)

            # Read as awkward arrays (event-structured)
            a1, a2 = _read_two_branches_as_arrays(tree, b1, b2)

            # --- Per-jet emission truncation (FIRST N elements) ---
            if args.max_emissions_per_jet is not None:
                a1 = _truncate_per_jet(a1, args.max_emissions_per_jet)
                a2 = _truncate_per_jet(a2, args.max_emissions_per_jet)

            # Number of jets = number of events (outer dimension)
            n_jets = len(a1)

            # Flatten to emissions (across all jets, after optional truncation)
            first_flat = ak.to_numpy(ak.flatten(a1, axis=None))
            second_flat = ak.to_numpy(ak.flatten(a2, axis=None))

            # Joint finite mask (NaN/inf safe) + apply
            mask = np.isfinite(first_flat) & np.isfinite(second_flat)
            first_flat = first_flat[mask]
            second_flat = second_flat[mask]

            # Convert to (X=log(1/dR), Y=log(kt)) by declared kinds
            X, Y = _interpret_and_to_logs(first_flat, second_flat, args.first_kind, args.second_kind)

            # Count emissions AFTER masking (reflects the actual histogrammed stats)
            n_emissions_after = len(X)

            if args.dry_run:
                print(f"[dry-run] {fpath.name}: jets={n_jets}, emissions(after mask)={n_emissions_after}, "
                      f"tree='{tpath}', first='{b1}', second='{b2}', maxN={args.max_emissions_per_jet}")
                continue

            # --- Title (3 lines): base title / file stem / stats+extra ---
            line1 = args.title
            line2 = stem
            line3_bits = [f"norm={args.norm}", f"N_jets={n_jets}", f"N_emissions={n_emissions_after}"]
            if args.max_emissions_per_jet is not None:
                line3_bits.append(f"maxN={args.max_emissions_per_jet}")
            if args.title_extra:
                line3_bits.append(args.title_extra)
            title = f"{line1}\n{line2}\n{'; '.join(line3_bits)}"

            # --- Plot & save for this file ---
            sum_after_norm, raw_in_range = plot_2d_hist(
                x=X, y=Y, n_jets=n_jets,
                xbins=args.xbins, ybins=args.ybins,
                xrange_=tuple(args.xrange) if args.xrange else None,
                yrange_=tuple(args.yrange) if args.yrange else None,
                title=title, cmap=args.cmap,
                out_png=out_png, out_npz=out_npz,
                norm_mode=args.norm, zmin=args.zmin, zmax=args.zmax,
            )

            print(f"[done] {fpath.name} -> saved: {out_png}")
            print(f"[done] {fpath.name} -> saved: {out_npz}")
            print(f"[check] {fpath.name}: sum(H after norm) = {sum_after_norm:.6f} ; "
                  f"emissions in-range (raw) = {int(raw_in_range)} ; "
                  f"branches: first='{b1}', second='{b2}' ; tree='{tpath}' ; "
                  f"maxN={args.max_emissions_per_jet}")


if __name__ == "__main__":
    main()
