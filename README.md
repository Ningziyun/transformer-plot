**Project: Jet Plotting Utilities**

This directory contains a collection of Python scripts for visualizing jet-related quantities, histograms, and training metrics in high-energy physics analyses. 

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

**plot.py**
```
# Main Lund-plane plotter. Reads one or multiple ROOT files, extracts kt and ΔR (or their logarithms), and produces 2D histograms (log(1/ΔR) vs log(kt)).
python plot.py --file_list fileList.txt --zmin 0 --zmax 0.025 --maxN 10
```

**plot_1dhist.py**
```
# Draws 1D histograms of scalar observables (e.g. jet pT, η, or custom quantities). Supports overlaid comparisons across multiple ROOT or NumPy inputs.
python plot_1dhist.py --file_list fileList.txt --mode kdr --maxN 10
```

**plot_bhist.py**
```
# Specialized “binned histogram” plotter for comparing distributions from multiple datasets or epochs. Includes ratio plots and normalization options.
python plot_bhist.py --file_list binList.txt
```

**plot_corr.py**
```
# Computes and visualizes correlation matrices between variables (1st and 2nd emissions). Produces a heatmap of Pearson correlation coefficients.
python plot_corr.py --file_list fileList.txt
```

**plot_eem.py**
```
# Emission-by-Emission Overlay Plotter
python plot_eem.py --file_list fileList.txt
```

**plot_loss.py**
```
# This script is a Loss vs Epoch Plotter — it reads ROOT files containing training history, extracts the loss (and optionally epoch) arrays, and plots loss vs epoch curves across multiple files.
python plot_loss.py --file_list fileList.txt
# loss function file could be found in transformer-lund/models/test
```

