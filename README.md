# Moore et al. 2013 Shoreline Data Reconnaissance

Inventory and processing of the historical shoreline shapefiles in the Murray Lab folder used by Moore et al. 2013 (GRL).

Prepared for a CEAMD meeting on May 6, 2026.

## What's here

- `scripts/audit_shorelines.py` — scans the data folder, builds a per-file metadata table (CRS, bounds, year, geometry, source).
- `scripts/make_slide_figures.py` — generates the 6 main slide figures from the audit table. Reports exact file accounting to stdout.
- `scripts/make_fig07_crs_by_era.py` — backup figure showing how CRS conventions changed across data-collection eras.
- `scripts/make_fig09_data_availability.py` — heatmap showing every file by year, region, and processing tier.
- `scripts/exploratory/` — one-off exploratory scripts.
- `slides/shoreshop_song_0506.pdf` — compiled slides.
- `audit_results/shapefile_audit.csv` — output of the audit script (one row per shapefile).
- `slide_figs/` — pre-rendered figures.

## Where the data lives

The shapefiles are NOT in this repo — they're on Duke Compute Cluster:

    /hpc/group/abmurraylab/Shoreline Data from Moore et al 2013/

To run the scripts locally, mirror that folder into `./data/`:

    rsync -avz dcc:'/hpc/group/abmurraylab/Shoreline\ Data\ from\ Moore\ et\ al\ 2013/' ./data/

## Reference

Moore, L. J., McNamara, D. E., Murray, A. B., & Brenner, O. (2013).
Observed changes in hurricane-driven waves explain the dynamics of modern
cuspate shorelines. *Geophysical Research Letters*, 40(22), 5867-5871.
https://doi.org/10.1002/2013GL057311

## Setup

Python 3.10+. Install dependencies:

    pip install -r requirements.txt

Run audit (produces `audit_results/shapefile_audit.csv`):

    python scripts/audit_shorelines.py

Generate slide figures:

    python scripts/make_slide_figures.py

Compile slides (requires LaTeX):

    cd slides
    pdflatex slides.tex
    pdflatex slides.tex

## Author

Y. Song, May 2026.
