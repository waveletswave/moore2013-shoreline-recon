"""
inspect_baseline_overlap.py — visually and quantitatively compare
the USGS Morton 2005 baseline with the Moore 2013 baselines in their
overlap regions (Cape Hatteras, Cape Lookout).

Answers two questions:
  1. Are the baselines geometrically identical, or different?
  2. If different, by how much (in meters)?

Run from the repo root:
    python scripts/inspect_baseline_overlap.py

Outputs:
  dylan_transects/baseline_comparison.png — visual side-by-side
  dylan_transects/baseline_comparison.txt — distance summary
"""

from pathlib import Path
import warnings
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=UserWarning)

HERE = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = HERE / "data"
PARENT = HERE.parent
OUT_DIR = PARENT / "dylan_transects"
OUT_DIR.mkdir(exist_ok=True)

TARGET_CRS = "EPSG:32119"

BASELINES = [
    ("Moore CapeHatteras 1852-1974",
     DATA_ROOT / "DSAS_cpHat/reprojected_shorelines/baseline_tailored1852_1974.shp",
     "#3a7bd5"),
    ("Moore CapeHatteras 1974-2004",
     DATA_ROOT / "DSAS_cpHat/reprojected_shorelines/baseline_tailored1974_2004.shp",
     "#2e8b57"),
    ("Moore CapeLookout",
     DATA_ROOT / "DSAS_cpLO/Reprojected_shorelines/Base.shp",
     "#d9633b"),
    ("USGS NC full coast",
     DATA_ROOT / "shorelines/USGS-nc_coast/nc_zip/nc_zip/nc_baseline.shp",
     "#c0392b"),
]


def sample_points_along_line(geom, spacing_m=100):
    """Sample points along a LineString or MultiLineString every `spacing_m` meters."""
    points = []
    geoms = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
    for g in geoms:
        n = max(2, int(g.length / spacing_m))
        for d in np.linspace(0, g.length, n):
            points.append(g.interpolate(d))
    return points


def main():
    # Load and reproject all baselines
    loaded = []
    for label, path, color in BASELINES:
        if not path.exists():
            print(f"WARNING: {path} not found")
            continue
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            bx = gdf.total_bounds
            if -180 <= bx[0] <= 180 and -90 <= bx[1] <= 90:
                gdf = gdf.set_crs("EPSG:4269")
        gdf = gdf.to_crs(TARGET_CRS)
        loaded.append((label, gdf, color))
        print(f"Loaded: {label}  ({len(gdf)} feature(s))")

    # Identify USGS for distance comparison
    usgs = next((gdf for label, gdf, _ in loaded if "USGS" in label), None)
    if usgs is None:
        raise RuntimeError("USGS baseline not loaded.")
    usgs_geom = usgs.union_all()

    # --- Distance analysis ---
    print()
    print("=" * 70)
    print("Distance from each Moore baseline to the USGS baseline")
    print("(sampled every 100 m along the Moore baseline)")
    print("=" * 70)
    summary_lines = []
    for label, gdf, _ in loaded:
        if "Moore" not in label:
            continue
        moore_geom = gdf.union_all()
        sample_pts = sample_points_along_line(moore_geom, spacing_m=100)
        distances = np.array([p.distance(usgs_geom) for p in sample_pts])
        n_zero = int((distances < 1).sum())
        line = (f"{label}\n"
                f"  Sampled {len(sample_pts)} points along Moore baseline\n"
                f"  Distance to USGS baseline (m):\n"
                f"    min:    {distances.min():.1f}\n"
                f"    median: {np.median(distances):.1f}\n"
                f"    mean:   {distances.mean():.1f}\n"
                f"    max:    {distances.max():.1f}\n"
                f"    points within 1 m of USGS: {n_zero} / {len(sample_pts)}"
                f" ({100*n_zero/len(sample_pts):.1f}%)\n")
        print(line)
        summary_lines.append(line)

    with open(OUT_DIR / "baseline_comparison.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print(f"Saved: {OUT_DIR / 'baseline_comparison.txt'}")

    # --- Visualization: 3 panels (full NC, Hatteras zoom, Lookout zoom) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # Panel 1: full NC coast
    ax = axes[0]
    for label, gdf, color in loaded:
        gdf.plot(ax=ax, color=color, linewidth=1.5,
                 linestyle="--" if "USGS" in label else "-",
                 label=label, alpha=0.85)
    ax.set_title("Full NC coast")
    ax.set_xlabel("Easting (m, EPSG:32119)")
    ax.set_ylabel("Northing (m)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")

    # Panel 2: Cape Hatteras zoom
    ax = axes[1]
    for label, gdf, color in loaded:
        gdf.plot(ax=ax, color=color, linewidth=2.0,
                 linestyle="--" if "USGS" in label else "-",
                 label=label, alpha=0.85)
    ax.set_xlim(880_000, 945_000)
    ax.set_ylim(140_000, 215_000)
    ax.set_title("Cape Hatteras zoom")
    ax.set_xlabel("Easting (m)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")

    # Panel 3: Cape Lookout zoom
    ax = axes[2]
    for label, gdf, color in loaded:
        gdf.plot(ax=ax, color=color, linewidth=2.0,
                 linestyle="--" if "USGS" in label else "-",
                 label=label, alpha=0.85)
    ax.set_xlim(780_000, 890_000)
    ax.set_ylim(85_000, 155_000)
    ax.set_title("Cape Lookout zoom")
    ax.set_xlabel("Easting (m)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal")

    fig.suptitle(
        "USGS Morton 2005 (red dashed) vs Moore 2013 baselines",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()
    out_png = OUT_DIR / "baseline_comparison.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
