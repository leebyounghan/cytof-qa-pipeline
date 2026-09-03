"""
Lyoplate GatingSet-extracted CSV → Parquet conversion.
GatingSet data is already compensated.

Output structure:
  {sample}.parquet  : wide table — channel columns (float32) + Step* gating columns (category)
  meta.json         : cohort-level metadata (channels, marker info, cofactor, dataset_config)
  samples.csv       : per-sample metadata (parquet_file, source_csv, sample_name, site, n_cells)

Usage:
    python3 pipeline/convert_to_parquet.py \
        --csv_dir pipeline/bcell/csv_tmp \
        --out_dir pipeline/bcell/parquet \
        --gating_plan pipeline/bcell/gating_plan.json \
        --config pipeline/bcell/dataset_config.yaml \
        --max_samples 2
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import yaml


def _load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_gating_plan(path):
    with open(path) as f:
        return json.load(f)


def _infer_marker_type(channel, config):
    for mtype, channels in config.get("markers", {}).items():
        if channel in channels:
            return mtype
    return "unknown"


def _find_gate_col(df_cols, leaf_name, parent_hint=None):
    matches = [c for c in df_cols if c.endswith("__" + leaf_name) or c == leaf_name]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1 and parent_hint:
        for m in matches:
            if parent_hint in m:
                return m
    return matches[0] if matches else None


def _build_step_labels(df, gating_plan, marker_channels):
    n = len(df)
    gate_cols = [c for c in df.columns if c not in marker_channels
                 and c not in ["sample_name", "site"]]

    obs = pd.DataFrame(index=range(n))

    for step in gating_plan["steps"]:
        col_name = step["annotation column name"]
        categories = step["annotation categories"]
        parent_ref = step["parent"]

        if parent_ref == "ALL":
            parent_mask = np.ones(n, dtype=bool)
        else:
            parts = parent_ref.split(" == ")
            pcol, pcat = parts[0].strip(), parts[1].strip()
            if pcol in obs.columns:
                parent_mask = (obs[pcol] == pcat).values
            else:
                parent_mask = np.zeros(n, dtype=bool)

        labels = np.full(n, np.nan, dtype=object)

        for cat in categories:
            gate_col = _find_gate_col(gate_cols, cat)

            if gate_col is None:
                cat_to_gate = {
                    "Lymphocytes": "Lymphocytes",
                    "Live": "Live",
                    "CD19pos": "CD19", "CD3pos": "CD3",
                    "CD19_CD20pos": "CD19 AND CD20", "CD19_CD20neg": "CD19 AND NOT CD20",
                    "Transitional": "Transitional",
                    "Memory_IgDneg": "Memory IgD-", "Memory_IgDpos": "Memory IgD+",
                    "Naive": "Naive", "IgDneg_CD27neg": "IgD-CD27-",
                    "Plasmablasts": "Plasmablasts",
                    "CD4": "CD4", "CD8": "CD8", "DNT": "DNT", "DPT": "DPT",
                    "CD4_Naive": "CD4 Naive", "CD4_Central_Memory": "CD4 Central Memory",
                    "CD4_Effector_Memory": "CD4 Effector Memory", "CD4_Effector": "CD4 Effector",
                    "CD4_Activated": "CD4 Activated",
                    "CD4_38negDRpos": "38- DR+", "CD4_38posDRneg": "38+ DR-", "CD4_38negDRneg": "38- DR-",
                    "CD8_Naive": "CD8 Naive", "CD8_Central_Memory": "CD8 Central Memory",
                    "CD8_Effector_Memory": "CD8 Effector Memory", "CD8_Effector": "CD8 Effector",
                    "CD8_Activated": "CD8 Activated",
                    "CD8_38negDRpos": "38- DR+", "CD8_38posDRneg": "38+ DR-", "CD8_38negDRneg": "38- DR-",
                    "Monocytes": "Monocytes",
                    "Lin_neg_CD14pos": "Lin-CD14+", "Lin_neg_CD14neg": "Lin-CD14-",
                    "CD14pos_CD16pos": "CD14+CD16+",
                    "CD16neg_CD56pos": "CD16-CD56+", "CD16pos_CD56pos": "CD16+CD56+",
                    "CD16pos_CD56neg": "CD16+CD56-", "CD16neg_CD56neg": "CD16-CD56-",
                    "HLADRpos": "HLADR+",
                    "pDC": "CD11c-CD123+", "mDC": "CD11c+CD123-",
                    "CD11cpos_CD123pos": "CD11c+CD123+", "CD11cneg_CD123neg": "CD11c-CD123-",
                    "CD4pos": "CD4",
                    "Treg": "Lo127Hi25",
                    "CCR4neg_HLADRpos": "CCR4-HLADR+", "CCR4pos_HLADRpos": "CCR4+HLADR+",
                    "CCR4pos_HLADRneg": "CCR4+HLADR-", "CCR4neg_HLADRneg": "CCR4-HLADR-",
                    "CCR4neg_CD45ROpos": "CCR4-CD45RO+", "CCR4pos_CD45ROpos": "CCR4+CD45RO+",
                    "CCR4pos_CD45ROneg": "CCR4+CD45RO-", "CCR4neg_CD45ROneg": "CCR4-CD45RO-",
                    "Activated": "Activated", "Memory": "Memory", "Total_Treg": "Total Treg",
                }
                gate_name = cat_to_gate.get(cat)
                if gate_name:
                    hint = None
                    if cat.startswith("CD8_"):
                        hint = "__CD8__"
                    elif cat.startswith("CD4_"):
                        hint = "__CD4__"
                    gate_col = _find_gate_col(gate_cols, gate_name, parent_hint=hint)

            if gate_col is None:
                continue

            mask = parent_mask & df[gate_col].values.astype(bool)
            labels[mask] = cat

        obs[col_name] = pd.Categorical(labels)

    return obs


def _write_meta(out_dir, config, marker_channels):
    channel_marker = config.get("channel_marker_map", {})
    cofactor = config.get("cofactor", 150.0)
    meta = {
        "cohort": config.get("name", "unknown"),
        "cofactor": cofactor,
        "transform": {
            "bio": f"arcsinh(x/{cofactor})",
            "scatter_tech": "clip(p1,p99)+minmax(0,10)",
            "dtype": "float16",
            "compression": "zstd",
        },
        "channels": {
            ch: [channel_marker.get(ch, ch), _infer_marker_type(ch, config)]
            for ch in marker_channels
        },
        "category_map": config.get("category_map", {}),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json written → {meta_path}")


def _write_samples_csv(out_dir, sample_rows):
    if not sample_rows:
        return
    df = pd.DataFrame(sample_rows)
    path = os.path.join(out_dir, "samples.csv")
    df.to_csv(path, index=False)
    print(f"  samples.csv written → {path}")


def convert_file(csv_path, out_dir, gating_plan, config):
    df = pd.read_csv(csv_path)
    csv_dir = os.path.dirname(csv_path)
    marker_map = pd.read_csv(os.path.join(csv_dir, "marker_map.csv"))
    marker_channels = marker_map["channel"].tolist()

    obs = _build_step_labels(df, gating_plan, marker_channels)

    # Collect sample metadata
    sample_name_raw = df["sample_name"].iloc[0] if "sample_name" in df.columns else os.path.basename(csv_path).replace(".csv.gz", "").replace(".csv", "")
    site = df["site"].iloc[0] if "site" in df.columns else ""

    # Parquet filename = original FCS name with .fcs → .parquet
    sample_name_str = str(sample_name_raw)
    fcs_stem = sample_name_str.replace('.fcs', '') if '.fcs' in sample_name_str else sample_name_str

    # Wide table: channels + step labels only
    result = df[marker_channels].astype(np.float32).reset_index(drop=True)

    # Transform channels by marker type
    cofactor = config.get('cofactor', 150.0)
    for ch in marker_channels:
        mtype = _infer_marker_type(ch, config)
        if mtype in ('protein', 'livedead', 'dna', 'bead', 'gaussian', 'background'):
            result[ch] = np.arcsinh(result[ch].values / cofactor).astype(np.float16)
        elif mtype in ('scatter', 'tech'):
            vals = result[ch].values
            vmin, vmax = np.percentile(vals, [1, 99])
            vals = np.clip(vals, vmin, vmax)
            result[ch] = ((vals - vmin) / (vmax - vmin) * 10 if vmax > vmin
                          else np.zeros_like(vals)).astype(np.float16)
        else:
            result[ch] = np.arcsinh(result[ch].values / cofactor).astype(np.float16)

    for col in obs.columns:
        result[col] = pd.Categorical(obs[col].values)

    out_path = os.path.join(out_dir, f"{fcs_stem}.parquet")
    result.to_parquet(out_path, index=False, compression='zstd')
    n_cells = len(result)
    print(f"  {fcs_stem}: {n_cells:,} cells × {len(marker_channels)} channels → {out_path}")

    info = {
        "parquet_file": f"{fcs_stem}.parquet",
        "source_csv": os.path.basename(csv_path),
        "sample_name": sample_name_str,
        "site": site,
        "n_cells": n_cells,
    }
    return out_path, marker_channels, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_dir",     required=True)
    parser.add_argument("--out_dir",     required=True)
    parser.add_argument("--gating_plan", required=True)
    parser.add_argument("--config",      required=True)
    parser.add_argument("--max_samples", type=int, default=0, help="Max samples (0 = all)")
    args = parser.parse_args()

    gating_plan = _load_gating_plan(args.gating_plan)
    config = _load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    manifest_path = os.path.join(args.csv_dir, "manifest.csv")
    manifest = pd.read_csv(manifest_path)

    csv_files = manifest["csv_file"].tolist()
    if args.max_samples > 0:
        csv_files = csv_files[:args.max_samples]

    print(f"Dataset: {config.get('name')}")
    print(f"Converting {len(csv_files)} files → {args.out_dir}\n")

    results, errors = [], []
    marker_channels = None
    sample_rows = []
    for fname in csv_files:
        csv_path = os.path.join(args.csv_dir, fname)
        try:
            out_path, ch, info = convert_file(csv_path, args.out_dir, gating_plan, config)
            results.append(out_path)
            sample_rows.append(info)
            if marker_channels is None:
                marker_channels = ch
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")
            import traceback; traceback.print_exc()
            errors.append(fname)

    if marker_channels is not None:
        _write_meta(args.out_dir, config, marker_channels)
    _write_samples_csv(args.out_dir, sample_rows)

    print(f"\nDone. {len(results)} converted, {len(errors)} errors.")


if __name__ == "__main__":
    main()
