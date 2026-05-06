"""
audit_shorelines_v2.py — Moore et al. 2013 shoreline data audit & visualization
                        (Fixed: handles multi-feature shapefiles correctly,
                                guesses CRS for UNDEFINED files, more robust)

Changes from v1:
- Bug fix: pass `color=color` instead of `color=[color]` (geopandas 1.0+ compatibility)
- For shapefiles with no CRS defined, guess EPSG:4269 if bounds look like lat/lon
- Verbose: prints WHY a shapefile is skipped (so you can debug)
- Auto-detect plot extents instead of hard-coded bbox
- Adds a 4th plot: just the "reprojected_shorelines" subset (cleanest data)

How to run:
    cd ~/Desktop/shoreline_recon
    python3 audit_shorelines_v2.py
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

import geopandas as gpd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent.parent
DATA_ROOT = HERE / "data"
OUTPUT_DIR = HERE / "audit_results"
OUTPUT_DIR.mkdir(exist_ok=True)

SCAN_DIRS = [
    DATA_ROOT / "shorelines",
    DATA_ROOT / "DSAS_cpHat",
    DATA_ROOT / "DSAS_cpLO",
]

# Common analysis CRS — NC State Plane NAD83 (meters), EPSG:32119
TARGET_CRS = "EPSG:32119"

# CRS guess for files with no .prj — based on inspecting bounds in CSV,
# the UNDEFINED files have lat/lon-looking bounds (~-77, ~34), so NAD83 is a safe bet.
CRS_GUESS_IF_UNDEFINED = "EPSG:4269"

VERBOSE = True   # set False to suppress per-file skip messages


# ---------------------------------------------------------------------------
# Year + category extraction (same as v1)
# ---------------------------------------------------------------------------

RANGE_PATTERN = re.compile(r"(?<!\d)(1[89]\d{2}|20\d{2})[_-](\d{2,4})(?!\d)")
SINGLE_YEAR_PATTERN = re.compile(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)")
SHORT_RANGE_PATTERN = re.compile(r"(\d{2})to(\d{2})")


def extract_years(name: str) -> tuple[int | None, int | None]:
    base = name.lower()
    m = RANGE_PATTERN.search(base)
    if m:
        y1 = int(m.group(1))
        y2_raw = m.group(2)
        y2 = (y1 // 100) * 100 + int(y2_raw) if len(y2_raw) == 2 else int(y2_raw)
        if 1840 < y1 <= 2030 and 1840 < y2 <= 2030 and y2 >= y1:
            return y1, y2
    m = SHORT_RANGE_PATTERN.search(base)
    if m:
        y1_raw, y2_raw = int(m.group(1)), int(m.group(2))
        y1 = 1900 + y1_raw if y1_raw >= 30 else 2000 + y1_raw
        y2 = 1900 + y2_raw if y2_raw >= 30 else 2000 + y2_raw
        return y1, y2
    matches = SINGLE_YEAR_PATTERN.findall(base)
    if matches:
        years = [int(y) for y in matches if 1840 < int(y) <= 2030]
        if years:
            return min(years), max(years)
    return None, None


def classify(path: Path) -> str:
    n = path.stem.lower()
    if any(k in n for k in ("baseline", "_base", "base.")):
        return "baseline"
    if any(k in n for k in ("trans", "transect", "tran74", "tran52")):
        return "transect"
    if any(k in n for k in ("setback", "factor")):
        return "setback"
    if any(k in n for k in ("er_points", "erosion_rate")):
        return "erosion_rate_points"
    if any(k in n for k in ("ncshrln", "stauble")):
        return "shoreline_composite"
    if "intersect" in n:
        return "intersection"
    if any(k in n for k in ("append", "app_")):
        return "shoreline_appended"
    if any(k in n for k in ("shoreline", "shrline", "shrl", "noaa", "nc18", "nc19", "nc20",
                            "obx", "oceanfront", "okracoke", "rodanthe")):
        return "shoreline"
    return "other"


def primary_year(year_start, year_end):
    if year_end is not None:
        return year_end
    if year_start is not None:
        return year_start
    return None


# ---------------------------------------------------------------------------
# Audit metadata
# ---------------------------------------------------------------------------

def find_all_shapefiles(roots: list[Path]) -> list[Path]:
    paths = []
    for root in roots:
        if not root.exists():
            continue
        paths.extend(sorted(root.rglob("*.shp")))
    return paths


def read_metadata(shp: Path) -> dict:
    rec = {
        "path": str(shp.relative_to(DATA_ROOT)),
        "name": shp.stem,
        "size_kb": round(shp.stat().st_size / 1024, 1),
        "n_features": None,
        "geom_type": None,
        "crs": None,
        "minx": None, "miny": None, "maxx": None, "maxy": None,
        "year_start": None, "year_end": None,
        "category": classify(shp),
        "error": None,
    }
    y1, y2 = extract_years(shp.stem)
    rec["year_start"] = y1
    rec["year_end"] = y2

    try:
        gdf = gpd.read_file(shp)
        rec["n_features"] = len(gdf)
        if not gdf.empty:
            rec["geom_type"] = str(gdf.geometry.iloc[0].geom_type)
            rec["crs"] = str(gdf.crs) if gdf.crs else "UNDEFINED"
            try:
                bx = gdf.total_bounds
                if np.all(np.isfinite(bx)):
                    rec["minx"], rec["miny"], rec["maxx"], rec["maxy"] = [float(b) for b in bx]
            except Exception as e:
                rec["error"] = f"bounds: {e}"
    except Exception as e:
        rec["error"] = str(e)[:200]
    return rec


# ---------------------------------------------------------------------------
# Plotting (FIXED + more verbose)
# ---------------------------------------------------------------------------

def safe_reproject(shp: Path, target_crs: str, verbose: bool = False):
    """Read and reproject. Returns None on failure with reason printed if verbose."""
    try:
        gdf = gpd.read_file(shp)
    except Exception as e:
        if verbose:
            print(f"      [read fail] {shp.name}: {str(e)[:80]}")
        return None

    if gdf.empty:
        if verbose:
            print(f"      [empty] {shp.name}")
        return None

    # Handle missing CRS
    if gdf.crs is None:
        # Guess based on bounds
        bx = gdf.total_bounds
        if -180 <= bx[0] <= 180 and -90 <= bx[1] <= 90:
            gdf = gdf.set_crs(CRS_GUESS_IF_UNDEFINED)
            if verbose:
                print(f"      [guessed CRS={CRS_GUESS_IF_UNDEFINED}] {shp.name}")
        else:
            if verbose:
                print(f"      [no CRS, can't guess] {shp.name} bounds={bx}")
            return None

    try:
        return gdf.to_crs(target_crs)
    except Exception as e:
        if verbose:
            print(f"      [reproject fail] {shp.name}: {str(e)[:80]}")
        return None


def plot_shorelines_by_year(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    bbox: tuple | None = None,
    plot_baselines: bool = True,
    plot_transects: bool = False,
    only_reprojected: bool = False,
    verbose: bool = False,
):
    """Plot all shorelines colored by year. Bbox in TARGET_CRS units."""
    fig, ax = plt.subplots(figsize=(13, 9))

    # Filter to shoreline-like categories
    shoreline_cats = ("shoreline", "shoreline_composite", "shoreline_appended")
    sl_df = df[df["category"].isin(shoreline_cats)].copy()
    
    if only_reprojected:
        # Use only the "Project" / "Projected" reprojected versions to avoid
        # plotting the same data multiple times in slightly different forms
        sl_df = sl_df[sl_df["path"].str.contains("reprojected", case=False, na=False)]

    valid = sl_df.dropna(subset=["primary_year"])
    if valid.empty:
        print(f"  [skip plot] no valid shorelines for {title}")
        plt.close(fig)
        return

    ymin, ymax = valid["primary_year"].min(), valid["primary_year"].max()
    norm = mcolors.Normalize(vmin=ymin, vmax=ymax)
    cmap = cm.viridis

    plotted = 0
    skipped = 0
    fail_reasons = {}

    for _, row in sl_df.iterrows():
        path = DATA_ROOT / row["path"]
        gdf = safe_reproject(path, TARGET_CRS, verbose=verbose)
        if gdf is None or gdf.empty:
            skipped += 1
            continue

        if bbox is not None:
            try:
                gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            except Exception:
                pass
            if gdf.empty:
                continue

        yr = row["primary_year"]
        color = cmap(norm(yr)) if pd.notna(yr) else (0.7, 0.7, 0.7, 1.0)

        # FIX: pass single color, not list-wrapped
        try:
            gdf.plot(ax=ax, color=color, linewidth=0.8, alpha=0.6)
            plotted += 1
        except Exception as e:
            skipped += 1
            err = str(e)[:60]
            fail_reasons[err] = fail_reasons.get(err, 0) + 1

    # Overlay baselines (red dashed)
    if plot_baselines:
        baselines = df[df["category"] == "baseline"]
        for _, row in baselines.iterrows():
            path = DATA_ROOT / row["path"]
            gdf = safe_reproject(path, TARGET_CRS, verbose=False)
            if gdf is not None and not gdf.empty:
                if bbox is not None:
                    gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                if not gdf.empty:
                    try:
                        gdf.plot(ax=ax, color="red", linewidth=1.5,
                                 linestyle="--", alpha=0.7)
                    except Exception:
                        pass

    # Overlay transects (light gray)
    if plot_transects:
        transects = df[df["category"] == "transect"]
        for _, row in transects.iterrows():
            path = DATA_ROOT / row["path"]
            gdf = safe_reproject(path, TARGET_CRS, verbose=False)
            if gdf is not None and not gdf.empty:
                if bbox is not None:
                    gdf = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                if not gdf.empty:
                    try:
                        gdf.plot(ax=ax, color="gray", linewidth=0.2, alpha=0.3)
                    except Exception:
                        pass

    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])
    # NB: we do NOT call set_aspect("equal") because it conflicts with set_xlim
    # and matplotlib was warning "Ignoring fixed limits". Equal aspect for
    # geographic data is ideal but in this case clarity matters more.

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Shoreline year")

    ax.set_title(f"{title}\n({plotted} shoreline files plotted, {skipped} skipped)")
    ax.set_xlabel(f"Easting (m, {TARGET_CRS})")
    ax.set_ylabel(f"Northing (m, {TARGET_CRS})")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path.name}  ({plotted} plotted, {skipped} skipped)")
    if fail_reasons:
        print(f"    Fail reasons:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"      [{count:3d}] {reason}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DATA_ROOT.exists():
        print(f"ERROR: data folder not found: {DATA_ROOT}")
        sys.exit(1)

    print("=" * 70)
    print(f"Auditing shapefiles under {DATA_ROOT}")
    print("=" * 70)
    shp_paths = find_all_shapefiles(SCAN_DIRS)
    print(f"Found {len(shp_paths)} .shp files")
    print()

    print("Reading metadata from each shapefile...")
    records = []
    for i, shp in enumerate(shp_paths, 1):
        if i % 25 == 0:
            print(f"  [{i}/{len(shp_paths)}]")
        records.append(read_metadata(shp))
    df = pd.DataFrame(records)
    df["primary_year"] = df.apply(
        lambda r: primary_year(r["year_start"], r["year_end"]), axis=1
    )

    csv_path = OUTPUT_DIR / "shapefile_audit.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAudit CSV saved: {csv_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total shapefiles:           {len(df)}")
    print(f"Successfully read:          {df['n_features'].notna().sum()}")
    print(f"Failed to read:             {df['error'].notna().sum()}")
    print()
    print("By category:")
    print(df["category"].value_counts().to_string())
    print()

    # Plot 1: Cape Hatteras region with auto-detected extent
    # Plot 2: Cape Lookout region with auto-detected extent
    # Auto-detect extents from the actual data instead of hard-coded bbox
    
    print("\n" + "=" * 70)
    print("Generating plots...")
    print("=" * 70)

    # Plot 1: NC coast overview, no bbox (let it auto-fit)
    plot_shorelines_by_year(
        df,
        title="All NC coast shorelines (Moore et al. 2013 dataset)",
        output_path=OUTPUT_DIR / "01_overview_NC_coast.png",
        bbox=None,
        plot_baselines=True,
        plot_transects=False,
        verbose=False,
    )

    # Plot 2: Cape Hatteras region — use the bounds of NOAA Cape_Hatteras shorelines
    # which are already in EPSG:32119, to define the bbox
    cphat_files = df[df["path"].str.contains("Cape_Hatteras", na=False)]
    cphat_files = cphat_files[cphat_files["crs"] == "EPSG:32119"]
    cphat_files = cphat_files.dropna(subset=["minx", "miny", "maxx", "maxy"])
    if not cphat_files.empty:
        bbox_hat = (
            cphat_files["minx"].min() - 5000,
            cphat_files["miny"].min() - 5000,
            cphat_files["maxx"].max() + 5000,
            cphat_files["maxy"].max() + 5000,
        )
        print(f"\n  Cape Hatteras bbox: {bbox_hat}")
        plot_shorelines_by_year(
            df,
            title="Cape Hatteras region",
            output_path=OUTPUT_DIR / "02_cape_hatteras.png",
            bbox=bbox_hat,
            plot_baselines=True,
            plot_transects=True,
        )
    
    # Plot 3: Cape Lookout region
    cplo_files = df[df["path"].str.contains("Cape_Lookout", na=False)]
    cplo_files = cplo_files[cplo_files["crs"] == "EPSG:32119"]
    cplo_files = cplo_files.dropna(subset=["minx", "miny", "maxx", "maxy"])
    if not cplo_files.empty:
        bbox_lo = (
            cplo_files["minx"].min() - 5000,
            cplo_files["miny"].min() - 5000,
            cplo_files["maxx"].max() + 5000,
            cplo_files["maxy"].max() + 5000,
        )
        print(f"\n  Cape Lookout bbox: {bbox_lo}")
        plot_shorelines_by_year(
            df,
            title="Cape Lookout region",
            output_path=OUTPUT_DIR / "03_cape_lookout.png",
            bbox=bbox_lo,
            plot_baselines=True,
            plot_transects=True,
        )

    # Plot 4: Cleaner version — only the reprojected_shorelines folder.
    # These are Laura's hand-curated, projection-corrected shorelines
    # that exactly correspond to Table S1 in Moore 2013.
    plot_shorelines_by_year(
        df,
        title="Reprojected shorelines only (Laura's curated set, NAD83 NC State Plane)",
        output_path=OUTPUT_DIR / "04_reprojected_only.png",
        bbox=None,
        plot_baselines=True,
        plot_transects=True,
        only_reprojected=True,
    )

    print("\n" + "=" * 70)
    print(f"Done. All outputs are in: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
