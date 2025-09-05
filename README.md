**Project: Per-file Lund Plane Plotter (`plot.py`)**

Generate one **2D histogram (Lund plane)** per ROOT file. The script auto-detects the TTree (unless provided), auto-picks two numeric branches (flat or jagged), interprets them semantically as **log(1/ΔR)** and **log(kt)** (or converts from raw ΔR/kt), and saves a heatmap **per file** with a 3-line title.

**requirements**
```
numpy
uproot
awkward
matplotlib
tqdm
```

**Input Files (`fileList.txt` example)**
```
# One ROOT file path per line; lines beginning with # are ignored
/path/to/sample_A.root
/path/to/subdir/sample_B.root
```

**Quick Start**
```
# Per-jet normalization (default), auto-detect tree/branches, one PNG+NPZ per input file
python plot.py --file_list fileList.txt
```

**Output Files (per input file `X.root`)**
```
<out_dir>/<out_tag>_X.png   # heatmap (2D histogram)
<out_dir>/<out_tag>_X.npz   # arrays: H (possibly normalized), xedges, yedges, norm_mode, raw_emissions_in_range
```

**Title Format (3 lines)**
```
1) --title (default: QCD Lund Plane)
2) <file-stem>            # input filename without .root
3) norm=...; N_jets=...; N_emissions=...; <title_extra if provided>
```

**Normalization Meaning**
```
- none         : raw bin counts
- per_jet     : bin counts divided by number of jets (events) in that file
- per_emission: bin counts divided by total emissions falling into the histogram range (probability per emission)
```

**Recommended Ranges (typical Lund-plane)**
```
--xrange 0 6 --yrange -6 2
```

**Headless Servers (no display)**
```
# Add to the very top of plot.py, before importing pyplot:
import matplotlib
matplotlib.use("Agg")
```

**Option Reference (each option explained with usage)**

**--file_list**  
Text file listing ROOT files, one path per line. Lines starting with `#` are ignored.  
Default: `fileList.txt`  
Use it when you want to control which files to process.
```
python plot.py --file_list fileList.txt
```

**--tree_name**  
TTree path (supports subdirectories like `Dir/SubDir/TreeName`) or `auto` to auto-detect the **first** TTree found in the file.  
Default: `auto`  
Use explicit name if your file has multiple trees and you want a specific one.
```
python plot.py --tree_name MyDir/Subdir/lundTree
```

**--first_kind**  
Semantic meaning of the **first** auto-picked branch. Accepted values:
```
logDRinv  # already log(1/ΔR) → used directly as X
logkt     # already log(kt)   → used directly as Y
deltaR    # raw ΔR → converted to log(1/ΔR) for X
kt        # raw kt → converted to log(kt) for Y
```
Default: `logDRinv`  
Choose this to tell the script what the first branch represents (not axis swapping).
```
python plot.py --first_kind deltaR
```

**--second_kind**  
Semantic meaning of the **second** auto-picked branch. Same accepted values as `--first_kind`.  
Default: `logkt`  
Usually pair with `--first_kind` to fully define (X=log(1/ΔR), Y=log(kt)).
```
python plot.py --second_kind kt
```

**--out_dir**  
Directory to store outputs (created if missing).  
Default: `plots`  
Change to keep results separate.
```
python plot.py --out_dir results/plots
```

**--out_tag**  
Output filename **prefix**. Final filenames are `<out_tag>_<file-stem>.png` and `<out_tag>_<file-stem>.npz`.  
Default: `lund2d`  
Use this to version or label outputs.
```
python plot.py --out_tag lund2d_v2
```

**--xbins**  
Number of bins along X (log(1/ΔR)).  
Default: `50`  
Increase for finer resolution (more memory/time).
```
python plot.py --xbins 80
```

**--ybins**  
Number of bins along Y (log(kt)).  
Default: `50`
```
python plot.py --ybins 80
```

**--xrange XMIN XMAX**  
Force X range. If omitted, range is inferred from data.  
Use to make files comparable or match literature plots.
```
python plot.py --xrange 0 6
```

**--yrange YMIN YMAX**  
Force Y range. If omitted, range is inferred from data.
```
python plot.py --yrange -6 2
```

**--norm**  
Histogram normalization mode:
```
none          # raw counts
per_jet       # counts / N_jets (events); default
per_emission  # counts / (sum of counts within [xrange, yrange])
```
Default: `per_jet`  
Pick `per_emission` to view a probability map; pick `none` for raw yields.
```
python plot.py --norm none
python plot.py --norm per_emission
```

**--zmin**  
Lower bound for the color scale (vmin).  
Default: `0.0`  
Raise it to suppress very low-count bins visually.
```
python plot.py --zmin 0.01
```

**--zmax**  
Upper bound for the color scale (vmax). If omitted, uses the max of the (normalized) histogram.  
Cap it to emphasize structure.
```
python plot.py --zmax 300
```

**--title**  
First line of the plot title (line 1).  
Default: `QCD Lund Plane`
```
python plot.py --title "QCD Lund Plane (13 TeV)"
```

**--title_extra**  
Optional note appended on the **third** title line (after stats).  
Default: empty  
Useful for tagging epoch, selection, or sample info.
```
python plot.py --title_extra "epoch=10; anti-kT R=0.8"
```

**--cmap**  
Matplotlib colormap name.  
Default: `viridis`  
Try `magma`, `inferno`, `plasma`, `cividis`, etc.
```
python plot.py --cmap plasma
```

**--seed**  
Random seed for NumPy (included for reproducibility; not critical here).  
Default: `0`
```
python plot.py --seed 123
```

**--dry_run**  
If set, the script **does not** produce plots. It only parses files, prints selected tree/branches, and reports jet/emission counts.  
Use to validate inputs quickly.
```
python plot.py --dry_run --debug
```

**--debug**  
If set, prints extra diagnostics: the chosen TTree path, the first 50 branch names/types, and the two auto-picked branches.  
Use this when branch auto-pick surprises you.
```
python plot.py --debug
```

**Common Usage Patterns**

**1) Raw ΔR and kt branches (auto-convert to logs), fixed ranges and capped colorbar, per-jet normalization**
```
python plot.py --file_list fileList.txt \
  --first_kind deltaR --second_kind kt \
  --xrange 0 6 --yrange -6 2 --zmax 300 \
  --norm per_jet --out_dir plots --out_tag lund2d
```

**2) Already-logged branches (`log_1_over_deltaR`, `log_kt`)**
```
python plot.py --file_list fileList.txt \
  --first_kind logDRinv --second_kind logkt \
  --norm per_emission
```

**3) Explicit TTree path (skip auto) and custom bins**
```
python plot.py --tree_name Analysis/Lund/lundTree --xbins 80 --ybins 80
```

**4) Sanity check before heavy runs**
```
python plot.py --file_list fileList.txt --dry_run --debug
```

**Troubleshooting**
```
- "Could not find two array-like branches": use --debug to inspect branches; your data may be nested structs/objects.
  Export numeric arrays or point to another TTree with --tree_name.

- ΔR / kt appear swapped: these are semantics, not axes. Set --first_kind/--second_kind to declare meanings correctly.

- Different files have different trees/branches: OK. Auto-discovery is done per file; each output is independent.

- Empty/odd plots: confirm xrange/yrange include your data; try removing ranges to let the script auto-scale.

- Server without display: ensure a non-interactive backend (see "Headless Servers" note).
```

```
