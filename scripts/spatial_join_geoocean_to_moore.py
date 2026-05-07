"""
spatial_join_geoocean_to_moore.py — spatial join between GeoOcean's
NC transects (1980-2023 hindcast) and historical shoreline baselines.

For each GeoOcean transect, finds a baseline within MAX_DISTANCE_M and
attaches:
  - matched_baseline   : the baseline label
  - baseline_tier      : the data source group (Moore_2013 / USGS_Morton / ...)
  - distance_m         : distance to that baseline (meters)
  - matched            : True if a baseline was found within MAX_DISTANCE_M

JOIN MODES
----------
Set JOIN_MODE below to choose how matches are decided when multiple
baselines could apply to the same transect:

  "usgs_only"
      Match against USGS Morton 2005 only. Simplest reference frame:
      one baseline for the entire NC coast. Use this if you want a
      single, uniform "distance from baseline" across all transects.

  "nearest_all"
      Match against all baselines (Moore + USGS), nearest-neighbor.
      Each transect picks whichever baseline is closest, regardless
      of tier. Useful for inspecting which baseline is geometrically
      closest in each region; can produce mixed-tier output along
      the coast.

  "moore_priority"
      Try Moore baselines first, fall through to USGS only where
      Moore doesn't reach. Use this if you want to keep Moore-region
      distances consistent with Moore 2013's own DSAS analysis.

Note: Moore and USGS baselines are NOT geometrically the same in the
cape regions (median offset ~166 m at Cape Hatteras, ~1.3 km at Cape
Lookout — see scripts/exploratory/inspect_baseline_overlap.py output).
The choice of mode therefore has real implications for any downstream
distance-based analysis.

USAGE
-----
Run from the repo root:
    python scripts/spatial_join_geoocean_to_moore.py

Outputs go to ../dylan_transects/ (parent shoreline_recon workspace),
since GeoOcean transects are external lab data, not part of this repo.

EXTENDING
---------
Add new baselines in the BASELINES list below. Each entry is:
    {"path": Path(...), "label": "...", "tier": "..."}
"""

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent.parent      # repo root
DATA_ROOT = HERE / "data"                          # symlink -> data
PARENT = HERE.parent                               # ~/Desktop/shoreline_recon
TRANSECTS_PATH = PARENT / "dylan_transects" / "NC_transects.geojson"
OUT_DIR = PARENT / "dylan_transects"

TARGET_CRS = "EPSG:32119"
MAX_DISTANCE_M = 1000  # only attach a baseline if within this distance

# ---------------------------------------------------------------------------
# Join mode — change this single line to switch behavior
# ---------------------------------------------------------------------------
JOIN_MODE = "usgs_only"        # uniform reference frame, full NC coast
# JOIN_MODE = "nearest_all"      # whichever baseline is closest, any tier
# JOIN_MODE = "moore_priority"   # Moore in capes, USGS elsewhere

# ---------------------------------------------------------------------------
# Baselines available for joining. The script will use whichever subset
# is required by the chosen JOIN_MODE.
# ---------------------------------------------------------------------------
BASELINES = [
    # Moore et al. 2013, two capes
    {
        "path": DATA_ROOT / "DSAS_cpHat/reprojected_shorelines/baseline_tailored1852_1974.shp",
        "label": "CapeHatteras_1852_1974",
        "tier": "Moore_2013",
    },
    {
        "path": DATA_ROOT / "DSAS_cpHat/reprojected_shorelines/baseline_tailored1974_2004.shp",
        "label": "CapeHatteras_1974_2004",
        "tier": "Moore_2013",
    },
    {
        "path": DATA_ROOT / "DSAS_cpLO/Reprojected_shorelines/Base.shp",
        "label": "CapeLookout",
        "tier": "Moore_2013",
    },
    # USGS Morton 2005, full NC coast (6 segments)
    {
        "path": DATA_ROOT / "shorelines/USGS-nc_coast/nc_zip/nc_zip/nc_baseline.shp",
        "label": "USGS_NC_full_coast",
        "tier": "USGS_Morton",
    },
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_baseline_entry(entry):
    """Read one baseline file, reproject to TARGET_CRS, attach metadata."""
    path = entry["path"]
    if not path.exists():
        print(f"  WARNING: baseline not found, skipping: {path}")
        return None
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        bx = gdf.total_bounds
        if -180 <= bx[0] <= 180 and -90 <= bx[1] <= 90:
            gdf = gdf.set_crs("EPSG:4269")
            print(f"  Inferred EPSG:4269 from bounds for {path.name}")
        else:
            print(f"  WARNING: cannot determine CRS for {path.name}, skipping")
            return None
    gdf = gdf.to_crs(TARGET_CRS)
    gdf["matched_baseline"] = entry["label"]
    gdf["baseline_tier"] = entry["tier"]
    print(f"  Loaded {path.name}: {len(gdf)} feature(s) "
          f"-> {entry['label']} (tier: {entry['tier']})")
    return gdf[["matched_baseline", "baseline_tier", "geometry"]]


def load_baselines(filter_tiers=None):
    """Load all baselines (or only those in filter_tiers if given)."""
    parts = []
    for entry in BASELINES:
        if filter_tiers is not None and entry["tier"] not in filter_tiers:
            continue
        gdf = load_baseline_entry(entry)
        if gdf is not None:
            parts.append(gdf)
    if not parts:
        raise RuntimeError("No baselines loaded.")
    return pd.concat(parts, ignore_index=True).pipe(gpd.GeoDataFrame, crs=TARGET_CRS)


# ---------------------------------------------------------------------------
# Join strategies
# ---------------------------------------------------------------------------
def nearest_all_join(transects, baselines):
    """Match every transect to its nearest baseline regardless of tier."""
    joined = gpd.sjoin_nearest(
        transects, baselines, how="left",
        max_distance=MAX_DISTANCE_M,
        distance_col="distance_m",
    )
    n_dup = joined.index.duplicated().sum()
    if n_dup > 0:
        print(f"  Note: {n_dup} ties between baselines; kept the first match")
    joined = joined.loc[~joined.index.duplicated(keep="first")].copy()
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])
    return joined


def priority_join(transects, baselines, tier_order):
    """Match against tiers in order; later tiers only fill remaining gaps."""
    print(f"  Priority order: {tier_order}")

    result = transects.copy()
    result["matched_baseline"] = pd.NA
    result["baseline_tier"] = pd.NA
    result["distance_m"] = float("nan")

    for tier in tier_order:
        sub = baselines[baselines["baseline_tier"] == tier]
        if sub.empty:
            print(f"    (no baselines in tier '{tier}', skipping)")
            continue

        unmatched_mask = result["matched_baseline"].isna()
        candidates = result.loc[unmatched_mask, [transects.geometry.name]]
        candidates = gpd.GeoDataFrame(candidates,
                                      geometry=transects.geometry.name,
                                      crs=transects.crs)
        if candidates.empty:
            print(f"    (all transects already matched before tier '{tier}')")
            break

        joined = gpd.sjoin_nearest(
            candidates, sub, how="left",
            max_distance=MAX_DISTANCE_M,
            distance_col="distance_m",
        )
        joined = joined.loc[~joined.index.duplicated(keep="first")]
        if "index_right" in joined.columns:
            joined = joined.drop(columns=["index_right"])

        new_matches = joined[joined["matched_baseline"].notna()]
        for col in ["matched_baseline", "baseline_tier", "distance_m"]:
            result.loc[new_matches.index, col] = new_matches[col]
        print(f"    Tier '{tier}': {len(new_matches)} new matches")

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(joined):
    n_total = len(joined)
    n_matched = int(joined["matched"].sum())
    n_unmatched = n_total - n_matched
    print()
    print("=" * 70)
    print("Join results:")
    print("=" * 70)
    print(f"  Total GeoOcean transects:    {n_total}")
    print(f"  Matched to a baseline:       {n_matched} "
          f"({100*n_matched/n_total:.1f}%)")
    print(f"  Outside any baseline region: {n_unmatched} "
          f"({100*n_unmatched/n_total:.1f}%)")
    print()
    print("By baseline tier:")
    print(joined["baseline_tier"].value_counts(dropna=False).to_string())
    print()
    print("By matched baseline:")
    print(joined["matched_baseline"].value_counts(dropna=False).to_string())
    print()
    if n_matched > 0:
        d = joined.loc[joined["matched"], "distance_m"]
        print("Distance to baseline (m), among matched transects:")
        print(f"  min:    {d.min():.1f}")
        print(f"  median: {d.median():.1f}")
        print(f"  mean:   {d.mean():.1f}")
        print(f"  max:    {d.max():.1f}")
    print()


def output_filename_for_mode(mode):
    """Each mode writes to its own output filename so results don't collide."""
    return {
        "usgs_only":      "NC_transects_joined_usgs_only",
        "nearest_all":    "NC_transects_joined_nearest_all",
        "moore_priority": "NC_transects_joined_moore_priority",
    }[mode]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"Spatial join: GeoOcean transects -> historical baselines")
    print(f"Mode: {JOIN_MODE}")
    print("=" * 70)
    print()

    # 1. Load transects
    print(f"Loading GeoOcean transects from {TRANSECTS_PATH}...")
    if not TRANSECTS_PATH.exists():
        raise FileNotFoundError(f"GeoOcean transects not found at {TRANSECTS_PATH}.")
    transects = gpd.read_file(TRANSECTS_PATH)
    print(f"  {len(transects)} transects, original CRS: {transects.crs}")
    transects = transects.to_crs(TARGET_CRS)
    print(f"  Reprojected to {TARGET_CRS}")
    print()

    # 2. Load baselines (subset depends on mode)
    print("Loading baselines...")
    if JOIN_MODE == "usgs_only":
        baselines = load_baselines(filter_tiers={"USGS_Morton"})
    else:
        baselines = load_baselines()
    print(f"  {len(baselines)} baseline feature(s) loaded")
    print()

    # 3. Run the join
    print(f"Running join (max distance: {MAX_DISTANCE_M} m)...")
    if JOIN_MODE == "usgs_only":
        joined = nearest_all_join(transects, baselines)
    elif JOIN_MODE == "nearest_all":
        joined = nearest_all_join(transects, baselines)
    elif JOIN_MODE == "moore_priority":
        joined = priority_join(transects, baselines,
                               tier_order=["Moore_2013", "USGS_Morton"])
    else:
        raise ValueError(f"Unknown JOIN_MODE: {JOIN_MODE!r}")

    joined["matched"] = joined["matched_baseline"].notna()

    # 4. Report
    report(joined)

    # 5. Save (each mode has its own filenames so they don't overwrite)
    stem = output_filename_for_mode(JOIN_MODE)
    out_geojson = OUT_DIR / f"{stem}.geojson"
    out_csv = OUT_DIR / f"{stem}.csv"
    joined.to_file(out_geojson, driver="GeoJSON")
    joined.drop(columns=["geometry"]).to_csv(out_csv, index=False)
    print(f"Saved: {out_geojson}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
