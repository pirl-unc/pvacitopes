#!/usr/bin/env python3
"""
A tool to process TSV files of predicted mutant epitopes and rank them.

Pipeline:
  1. Reads an input TSV file.
  2. Computes MHC threshold boolean columns (with clean names starting with "filter:")
     for five metrics:
         - MHCflurryEL Presentation MT Score (> threshold)
         - NetMHCpanEL MT Percentile (< threshold)
         - MHCflurryEL Presentation MT Percentile (< threshold)
         - NetMHCpan MT IC50 Score (< threshold)
         - NetMHCpan MT Percentile (< threshold)
     For MHCflurry Presentation Score, an adaptive scheme (by default) is used so that
     for the default range (0.5 to 0.95) and 4 thresholds you get values close to:
         [0.50, 0.75, 0.90, 0.95]; if the number increases (e.g. to 5) then roughly
         [0.50, 0.75, 0.85, 0.90, 0.95]. (For the other metrics, the default step type is “log2”.)
  3. Sums these boolean columns into a per‐row "Sum_MHC_Score" and discards rows with a zero score.
  4. Groups the remaining rows by mutation (using canonical columns)
     and selects the best transcript based on:
         - Highest Transcript Expression (within 10% of max)
         - Preference for protein_coding
         - Best (lowest) Transcript Support Level
         - Highest Protein Position
         - Lexicographic tie-breaker on Transcript.
  5. Computes additional scores:
         - "filter: frameshift": a numeric (0/1) flag for frameshift mutations.
         - "RNA_Score": computed from Transcript Expression using a transformation chosen
           by the parameter `rna_transform` ("linear", "sqrt", or "log2").
         - "DNA_Score": defined as min(1, Tumor DNA VAF / median(Tumor DNA VAF)).
         - "Bonus": applied if the mutation is a frameshift.
         - Final "Ranking_Score" = Sum_MHC_Score * DNA_Score * RNA_Score * Bonus.
  6. Outputs a full TSV (sorted descending by Ranking_Score) and a top‑n TSV.
  7. Logs summary counts for HLA Allele and Gene Name among the top‑n.

Usage:
    python rank_epitopes.py --input-file INPUT.tsv [other options]

New options allow:
  - Overriding the names of key columns via CLI arguments.
  - Logging the merged CLI arguments for reproducibility.
  - Saving the merged CLI args to a JSON/YAML file.
  - Loading a set of defaults from a JSON/YAML file, which can be overridden by CLI args.
"""

import sys
import re
import json
import os
import inspect
import numpy as np
import pandas as pd
import argh
from loguru import logger
import loguru  # for version logging

# Try importing YAML support.
try:
    import yaml

    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

# Configure loguru.
logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")
logger.info("Pandas version: {}", pd.__version__)
logger.info("NumPy version: {}", np.__version__)
logger.info("Loguru version: {}", loguru.__version__)


def load_defaults_from_file(file_path: str) -> dict:
    """Load defaults from a JSON or YAML file based on the file extension."""
    _, ext = os.path.splitext(file_path)
    try:
        with open(file_path) as f:
            if ext.lower() == ".json":
                return json.load(f)
            elif ext.lower() in [".yaml", ".yml"]:
                if HAVE_YAML:
                    return yaml.safe_load(f)
                else:
                    logger.error("YAML file provided but PyYAML is not installed.")
                    sys.exit(1)
            else:
                # Default to JSON
                return json.load(f)
    except Exception as e:
        logger.error("Error loading defaults from {}: {}", file_path, e)
        sys.exit(1)


def save_args_to_file(
    args_dict: dict, file_path: str, file_format: str = "json"
) -> None:
    """Save the CLI arguments dictionary to a file in JSON or YAML format."""
    try:
        with open(file_path, "w") as f:
            if file_format.lower() == "yaml":
                if HAVE_YAML:
                    yaml.dump(args_dict, f)
                else:
                    logger.error("YAML format selected but PyYAML is not installed.")
                    sys.exit(1)
            else:
                json.dump(args_dict, f, indent=2)
        logger.info("Saved CLI args to {}", file_path)
    except Exception as e:
        logger.error("Error saving CLI args to {}: {}", file_path, e)


def get_tsl_value(tsl: str) -> float:
    """
    Convert a Transcript Support Level (TSL) value to a numeric value.

    Args:
        tsl (str): The TSL value, e.g. "TSL1" or "1".

    Returns:
        float: A numeric value.
    """
    try:
        return float(tsl)
    except ValueError:
        m = re.search(r"\d+", str(tsl))
        if m:
            return float(m.group(0))
    return np.inf


def select_best_transcript(group: pd.DataFrame, log_ctx: logger) -> pd.Series:
    """
    Select the best transcript from a group (all rows for a given mutation)
    based on:
      1. Transcript Expression (within 10% of max)
      2. Preference for protein_coding biotype
      3. Lowest Transcript Support Level
      4. Highest Protein Position
      5. Lexicographic order on Transcript.

    Args:
        group (pd.DataFrame): Group of transcripts for one mutation.
        log_ctx (logger): Logger for decision messages.

    Returns:
        pd.Series: The selected transcript row.
    """
    group = group.copy()
    decision_notes = []
    # Convert to numeric.
    group["Transcript Expression"] = pd.to_numeric(
        group["Transcript Expression"], errors="coerce"
    )
    group["Protein Position"] = pd.to_numeric(
        group["Protein Position"], errors="coerce"
    )

    # 1. Filter: keep transcripts within 10% of max expression.
    max_expr = group["Transcript Expression"].max()
    expr_threshold = 0.9 * max_expr
    candidates = group[group["Transcript Expression"] >= expr_threshold]
    decision_notes.append(
        f"Max Expression: {max_expr:.3f} (threshold: {expr_threshold:.3f}), candidates: {len(candidates)}"
    )

    # 2. Prefer protein_coding.
    if (candidates["Biotype"] == "protein_coding").any():
        candidates_pc = candidates[candidates["Biotype"] == "protein_coding"]
        decision_notes.append(f"Filtered protein_coding: {len(candidates_pc)}")
        candidates = candidates_pc

    # 3. Prefer lowest TSL.
    candidates["TSL_numeric"] = candidates["Transcript Support Level"].apply(
        get_tsl_value
    )
    best_tsl = candidates["TSL_numeric"].min()
    candidates = candidates[candidates["TSL_numeric"] == best_tsl]
    decision_notes.append(f"Best TSL: {best_tsl}, candidates: {len(candidates)}")

    # 4. Prefer highest protein position.
    best_protein_pos = candidates["Protein Position"].max()
    candidates = candidates[candidates["Protein Position"] == best_protein_pos]
    decision_notes.append(
        f"Highest Protein Position: {best_protein_pos}, candidates: {len(candidates)}"
    )

    # 5. Lexicographic order on Transcript.
    candidates = candidates.sort_values("Transcript")
    chosen = candidates.iloc[0]
    decision_notes.append(f"Chosen transcript: {chosen['Transcript']}")

    mutation_id = group.iloc[0][
        ["Chromosome", "Start", "Stop", "Reference", "Variant", "Gene Name"]
    ].to_dict()
    log_ctx.info("Mutation {}: {}", mutation_id, " | ".join(decision_notes))
    return chosen


def adaptive_thresholds(min_val: float, max_val: float, num: int) -> np.ndarray:
    """
    Generate adaptive thresholds for MHCflurry presentation score.

    The first value is fixed to min_val and the last to max_val.
    For num>=3, intermediate thresholds are chosen so that for the default
    range (0.5 to 0.95) you get values close to:
       n=4: [0.50, 0.75, 0.90, 0.95]
       n=5: [0.50, 0.75, 0.85, 0.90, 0.95]
    For other ranges, intermediate thresholds are linearly interpolated using
    normalized anchors 0.5556 and 0.8889.

    Args:
        min_val (float): Minimum threshold.
        max_val (float): Maximum threshold.
        num (int): Total number of thresholds.

    Returns:
        np.ndarray: Array of thresholds.
    """
    if num == 1:
        return np.array([min_val])
    if num == 2:
        return np.array([min_val, max_val])
    intermediate = np.linspace(0.5556, 0.8889, num - 2)
    thresholds = (
        [min_val]
        + [min_val + (max_val - min_val) * t for t in intermediate]
        + [max_val]
    )
    return np.array(thresholds)


def main(
    input_file: str,
    # Output options:
    all_output: str = "all-epitopes.tsv",
    top_output: str = "top-epitopes.tsv",
    top_n: int = 50,
    bonus_multiplier: int = 5,
    log_file: str = "log.txt",
    rna_transform: str = "sqrt",
    # Threshold parameters for MHC metrics:
    min_threshold_mhcflurry_presentation_score: float = 0.5,
    max_threshold_mhcflurry_presentation_score: float = 0.95,
    num_threshold_mhcflurry_presentation_score: int = 4,
    step_type_mhcflurry_pres: str = "adaptive",
    min_threshold_netmhcpanel_el_percentile: float = 0.25,
    max_threshold_netmhcpanel_el_percentile: float = 2,
    num_threshold_netmhcpanel_el_percentile: int = 4,
    step_type_netmhcpanel_el: str = "log2",
    min_threshold_mhcflurry_presentation_percentile: float = 0.25,
    max_threshold_mhcflurry_presentation_percentile: float = 2,
    num_threshold_mhcflurry_presentation_percentile: int = 4,
    step_type_mhcflurry_pres_pct: str = "log2",
    min_threshold_netmhcpan_ba: float = 125,
    max_threshold_netmhcpan_ba: float = 1000,
    num_threshold_netmhcpan_ba: int = 4,
    step_type_netmhcpan_ba: str = "log2",
    min_threshold_netmhcpan_percentile: float = 0.25,
    max_threshold_netmhcpan_percentile: float = 2,
    num_threshold_netmhcpan_percentile: int = 4,
    step_type_netmhcpan_pct: str = "log2",
    # Column name options:
    col_chromosome: str = "Chromosome",
    col_start: str = "Start",
    col_stop: str = "Stop",
    col_reference: str = "Reference",
    col_variant: str = "Variant",
    col_transcript: str = "Transcript",
    col_transcript_support: str = "Transcript Support Level",
    col_transcript_expression: str = "Transcript Expression",
    col_biotype: str = "Biotype",
    col_protein_position: str = "Protein Position",
    col_gene_name: str = "Gene Name",
    col_mutation: str = "Mutation",
    col_mhcflurry_presentation: str = "MHCflurryEL Presentation MT Score",
    col_netmhcpanel_el: str = "NetMHCpanEL MT Percentile",
    col_mhcflurry_presentation_percentile: str = "MHCflurryEL Presentation MT Percentile",
    col_netmhcpan_ba: str = "NetMHCpan MT IC50 Score",
    col_netmhcpan_percentile: str = "NetMHCpan MT Percentile",
    col_tumor_dna_vaf: str = "Tumor DNA VAF",
    col_hla_allele: str = "HLA Allele",
    # Defaults file options:
    load_defaults: str = None,
    save_args: str = None,
    save_args_format: str = "json",
) -> None:
    """
    Process the mutant epitope TSV file and output ranked epitopes.

    New options allow overriding column names as well as loading/saving default CLI args.
    """
    # --- Implicitly build built-in defaults from main's signature ---
    builtin_defaults = {
        k: v.default
        for k, v in inspect.signature(main).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    # Build a dictionary of CLI args from the parameters of main.
    cli_args = {k: locals()[k] for k in inspect.signature(main).parameters.keys()}

    # If a defaults file is provided, load it.
    if load_defaults is not None:
        loaded_defaults = load_defaults_from_file(load_defaults)
    else:
        loaded_defaults = {}

    # Merge: built‑in defaults < loaded defaults < CLI args.
    merged = {}
    for key in cli_args:
        if key in builtin_defaults:
            # If the CLI arg equals the built‐in default and a loaded default exists, use that.
            if cli_args[key] == builtin_defaults[key] and key in loaded_defaults:
                merged[key] = loaded_defaults[key]
            else:
                merged[key] = cli_args[key]
        else:
            # For required arguments (e.g. input_file) not in builtin_defaults.
            merged[key] = cli_args[key]

    # Log and optionally save the merged CLI arguments.
    logger.info("Merged CLI arguments:")
    for k, v in merged.items():
        logger.info("  {}: {}", k, v)

    if merged.get("save_args") is not None:
        save_args_to_file(
            merged, merged["save_args"], merged.get("save_args_format", "json")
        )

    # --- Use merged values in the rest of the code ---
    input_file = merged["input_file"]
    all_output = merged["all_output"]
    top_output = merged["top_output"]
    top_n = merged["top_n"]
    bonus_multiplier = merged["bonus_multiplier"]
    log_file = merged["log_file"]
    rna_transform = merged["rna_transform"]
    min_threshold_mhcflurry_presentation_score = merged[
        "min_threshold_mhcflurry_presentation_score"
    ]
    max_threshold_mhcflurry_presentation_score = merged[
        "max_threshold_mhcflurry_presentation_score"
    ]
    num_threshold_mhcflurry_presentation_score = merged[
        "num_threshold_mhcflurry_presentation_score"
    ]
    step_type_mhcflurry_pres = merged["step_type_mhcflurry_pres"]
    min_threshold_netmhcpanel_el_percentile = merged[
        "min_threshold_netmhcpanel_el_percentile"
    ]
    max_threshold_netmhcpanel_el_percentile = merged[
        "max_threshold_netmhcpanel_el_percentile"
    ]
    num_threshold_netmhcpanel_el_percentile = merged[
        "num_threshold_netmhcpanel_el_percentile"
    ]
    step_type_netmhcpanel_el = merged["step_type_netmhcpanel_el"]
    min_threshold_mhcflurry_presentation_percentile = merged[
        "min_threshold_mhcflurry_presentation_percentile"
    ]
    max_threshold_mhcflurry_presentation_percentile = merged[
        "max_threshold_mhcflurry_presentation_percentile"
    ]
    num_threshold_mhcflurry_presentation_percentile = merged[
        "num_threshold_mhcflurry_presentation_percentile"
    ]
    step_type_mhcflurry_pres_pct = merged["step_type_mhcflurry_pres_pct"]
    min_threshold_netmhcpan_ba = merged["min_threshold_netmhcpan_ba"]
    max_threshold_netmhcpan_ba = merged["max_threshold_netmhcpan_ba"]
    num_threshold_netmhcpan_ba = merged["num_threshold_netmhcpan_ba"]
    step_type_netmhcpan_ba = merged["step_type_netmhcpan_ba"]
    min_threshold_netmhcpan_percentile = merged["min_threshold_netmhcpan_percentile"]
    max_threshold_netmhcpan_percentile = merged["max_threshold_netmhcpan_percentile"]
    num_threshold_netmhcpan_percentile = merged["num_threshold_netmhcpan_percentile"]
    step_type_netmhcpan_pct = merged["step_type_netmhcpan_pct"]

    # Column names (original names in input file):
    col_chromosome = merged["col_chromosome"]
    col_start = merged["col_start"]
    col_stop = merged["col_stop"]
    col_reference = merged["col_reference"]
    col_variant = merged["col_variant"]
    col_transcript = merged["col_transcript"]
    col_transcript_support = merged["col_transcript_support"]
    col_transcript_expression = merged["col_transcript_expression"]
    col_biotype = merged["col_biotype"]
    col_protein_position = merged["col_protein_position"]
    col_gene_name = merged["col_gene_name"]
    col_mutation = merged["col_mutation"]
    col_mhcflurry_presentation = merged["col_mhcflurry_presentation"]
    col_netmhcpanel_el = merged["col_netmhcpanel_el"]
    col_mhcflurry_presentation_percentile = merged[
        "col_mhcflurry_presentation_percentile"
    ]
    col_netmhcpan_ba = merged["col_netmhcpan_ba"]
    col_netmhcpan_percentile = merged["col_netmhcpan_percentile"]
    col_tumor_dna_vaf = merged["col_tumor_dna_vaf"]
    col_hla_allele = merged["col_hla_allele"]

    # Add log file handler.
    logger.add(
        log_file,
        level="INFO",
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        mode="w",
    )
    logger.info("Processing input file: {}", input_file)

    # Read input TSV.
    try:
        df = pd.read_csv(input_file, sep="\t")
    except Exception as e:
        logger.error("Error reading {}: {}", input_file, e)
        sys.exit(1)

    # --- Rename input columns to canonical names ---
    col_mapping = {
        col_chromosome: "Chromosome",
        col_start: "Start",
        col_stop: "Stop",
        col_reference: "Reference",
        col_variant: "Variant",
        col_transcript: "Transcript",
        col_transcript_support: "Transcript Support Level",
        col_transcript_expression: "Transcript Expression",
        col_biotype: "Biotype",
        col_protein_position: "Protein Position",
        col_gene_name: "Gene Name",
        col_mutation: "Mutation",
        col_mhcflurry_presentation: "MHCflurryEL Presentation MT Score",
        col_netmhcpanel_el: "NetMHCpanEL MT Percentile",
        col_mhcflurry_presentation_percentile: "MHCflurryEL Presentation MT Percentile",
        col_netmhcpan_ba: "NetMHCpan MT IC50 Score",
        col_netmhcpan_percentile: "NetMHCpan MT Percentile",
        col_tumor_dna_vaf: "Tumor DNA VAF",
        col_hla_allele: "HLA Allele",
    }
    df.rename(columns=col_mapping, inplace=True)

    # Check required columns.
    required_columns = list(col_mapping.values())
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error("Missing required columns: {}", missing)
        sys.exit(1)

    # === Compute MHC boolean columns on the full DataFrame ===
    # MHCflurry Presentation Score:
    stype = step_type_mhcflurry_pres.lower()
    if stype == "adaptive":
        thr_mhcflurry_pres = adaptive_thresholds(
            min_threshold_mhcflurry_presentation_score,
            max_threshold_mhcflurry_presentation_score,
            num_threshold_mhcflurry_presentation_score,
        )
    elif stype == "linear":
        thr_mhcflurry_pres = np.linspace(
            min_threshold_mhcflurry_presentation_score,
            max_threshold_mhcflurry_presentation_score,
            num_threshold_mhcflurry_presentation_score,
        )
    elif stype == "log2":
        thr_mhcflurry_pres = np.geomspace(
            max_threshold_mhcflurry_presentation_score,
            min_threshold_mhcflurry_presentation_score,
            num=num_threshold_mhcflurry_presentation_score,
        )
    else:
        logger.error(
            "Invalid step type for mhcflurry presentation score: {}",
            step_type_mhcflurry_pres,
        )
        sys.exit(1)

    # NetMHCpanEL Percentile:
    if step_type_netmhcpanel_el.lower() == "linear":
        thr_netmhcpanel_el = np.linspace(
            max_threshold_netmhcpanel_el_percentile,
            min_threshold_netmhcpanel_el_percentile,
            num_threshold_netmhcpanel_el_percentile,
        )
    elif step_type_netmhcpanel_el.lower() == "log2":
        thr_netmhcpanel_el = np.geomspace(
            max_threshold_netmhcpanel_el_percentile,
            min_threshold_netmhcpanel_el_percentile,
            num=num_threshold_netmhcpanel_el_percentile,
        )
    else:
        logger.error(
            "Invalid step type for netmhcpanel el percentile: {}",
            step_type_netmhcpanel_el,
        )
        sys.exit(1)

    # MHCflurry Presentation Percentile:
    if step_type_mhcflurry_pres_pct.lower() == "linear":
        thr_mhcflurry_pres_pct = np.linspace(
            max_threshold_mhcflurry_presentation_percentile,
            min_threshold_mhcflurry_presentation_percentile,
            num_threshold_mhcflurry_presentation_percentile,
        )
    elif step_type_mhcflurry_pres_pct.lower() == "log2":
        thr_mhcflurry_pres_pct = np.geomspace(
            max_threshold_mhcflurry_presentation_percentile,
            min_threshold_mhcflurry_presentation_percentile,
            num=num_threshold_mhcflurry_presentation_percentile,
        )
    else:
        logger.error(
            "Invalid step type for mhcflurry presentation percentile: {}",
            step_type_mhcflurry_pres_pct,
        )
        sys.exit(1)

    # NetMHCpan BA Score:
    if step_type_netmhcpan_ba.lower() == "linear":
        thr_netmhcpan_ba = np.linspace(
            max_threshold_netmhcpan_ba,
            min_threshold_netmhcpan_ba,
            num_threshold_netmhcpan_ba,
        )
    elif step_type_netmhcpan_ba.lower() == "log2":
        thr_netmhcpan_ba = np.geomspace(
            max_threshold_netmhcpan_ba,
            min_threshold_netmhcpan_ba,
            num=num_threshold_netmhcpan_ba,
        )
    else:
        logger.error("Invalid step type for netmhcpan BA: {}", step_type_netmhcpan_ba)
        sys.exit(1)

    # NetMHCpan Percentile:
    if step_type_netmhcpan_pct.lower() == "linear":
        thr_netmhcpan_pct = np.linspace(
            max_threshold_netmhcpan_percentile,
            min_threshold_netmhcpan_percentile,
            num_threshold_netmhcpan_percentile,
        )
    elif step_type_netmhcpan_pct.lower() == "log2":
        thr_netmhcpan_pct = np.geomspace(
            max_threshold_netmhcpan_percentile,
            min_threshold_netmhcpan_percentile,
            num=num_threshold_netmhcpan_percentile,
        )
    else:
        logger.error(
            "Invalid step type for netmhcpan percentile: {}", step_type_netmhcpan_pct
        )
        sys.exit(1)

    # Round thresholds to 2 decimals.
    thr_mhcflurry_pres = np.round(thr_mhcflurry_pres, 2)
    thr_netmhcpanel_el = np.round(thr_netmhcpanel_el, 2)
    thr_mhcflurry_pres_pct = np.round(thr_mhcflurry_pres_pct, 2)
    thr_netmhcpan_ba = np.round(thr_netmhcpan_ba, 2)
    thr_netmhcpan_pct = np.round(thr_netmhcpan_pct, 2)

    logger.info("Generated thresholds:")
    logger.info("filter: mhcflurry presentation > thresholds: {}", thr_mhcflurry_pres)
    logger.info("filter: netmhcpan el percentile < thresholds: {}", thr_netmhcpanel_el)
    logger.info(
        "filter: mhcflurry presentation percentile < thresholds: {}",
        thr_mhcflurry_pres_pct,
    )
    logger.info("filter: netmhcpan BA < thresholds: {}", thr_netmhcpan_ba)
    logger.info("filter: netmhcpan percentile < thresholds: {}", thr_netmhcpan_pct)

    # Create boolean columns with clean names.
    bool_cols = []
    for thr in thr_mhcflurry_pres:
        colname = f"filter: mhcflurry presentation > {thr:.2f}"
        df[colname] = (df["MHCflurryEL Presentation MT Score"] > thr).astype(int)
        bool_cols.append(colname)
    for thr in thr_netmhcpanel_el:
        colname = f"filter: netmhcpan el percentile < {thr:.2f}"
        df[colname] = (df["NetMHCpanEL MT Percentile"] < thr).astype(int)
        bool_cols.append(colname)
    for thr in thr_mhcflurry_pres_pct:
        colname = f"filter: mhcflurry presentation percentile < {thr:.2f}"
        df[colname] = (df["MHCflurryEL Presentation MT Percentile"] < thr).astype(int)
        bool_cols.append(colname)
    for thr in thr_netmhcpan_ba:
        colname = f"filter: netmhcpan BA < {thr:.2f}"
        df[colname] = (df["NetMHCpan MT IC50 Score"] < thr).astype(int)
        bool_cols.append(colname)
    for thr in thr_netmhcpan_pct:
        colname = f"filter: netmhcpan percentile < {thr:.2f}"
        df[colname] = (df["NetMHCpan MT Percentile"] < thr).astype(int)
        bool_cols.append(colname)

    # Compute Sum_MHC_Score for each row.
    df["Sum_MHC_Score"] = df[bool_cols].sum(axis=1)
    logger.info("Computed Sum_MHC_Score for all rows.")

    # Filter out rows with Sum_MHC_Score == 0.
    df = df[df["Sum_MHC_Score"] > 0]
    logger.info("Filtered to {} rows with Sum_MHC_Score > 0.", len(df))

    # === Group by mutation and select best transcript per mutation ===
    mutation_cols = ["Chromosome", "Start", "Stop", "Reference", "Variant", "Gene Name"]
    groups = df.groupby(mutation_cols, as_index=False)
    selected = []
    for name, group in groups:
        try:
            chosen = select_best_transcript(group, logger)
            selected.append(chosen)
        except Exception as ex:
            logger.error("Error selecting transcript for mutation {}: {}", name, ex)
    if not selected:
        logger.error("No transcripts selected after filtering; exiting.")
        sys.exit(1)
    df_best = pd.DataFrame(selected)
    logger.info("Selected best transcript for {} mutations.", len(df_best))

    # === Compute additional scores ===
    # Create a numeric frameshift flag with "filter:" prefix.
    df_best["filter: frameshift"] = (
        df_best["Mutation"]
        .str.contains("fs|frameshift", flags=re.IGNORECASE, na=False)
        .astype(int)
    )

    # Compute DNA_Score = min(1, Tumor DNA VAF / median(Tumor DNA VAF)).
    median_dna_vaf = df_best["Tumor DNA VAF"].median()
    if median_dna_vaf == 0:
        logger.warning(
            "Median Tumor DNA VAF is 0; setting DNA_Score to 1 for all entries."
        )
        df_best["DNA_Score"] = 1
    else:
        df_best["DNA_Score"] = (df_best["Tumor DNA VAF"] / median_dna_vaf).clip(upper=1)

    # Compute RNA_Score from Transcript Expression based on transformation.
    try:
        expr = df_best["Transcript Expression"].astype(float)
    except Exception as e:
        logger.error("Error converting Transcript Expression to float: {}", e)
        sys.exit(1)
    rna_trans = rna_transform.lower()
    if rna_trans == "linear":
        df_best["RNA_Score"] = expr
    elif rna_trans == "sqrt":
        df_best["RNA_Score"] = np.sqrt(expr)
    elif rna_trans == "log2":
        df_best["RNA_Score"] = np.log2(expr + 1)
    else:
        logger.error("Invalid RNA transform: {}", rna_transform)
        sys.exit(1)

    # Apply bonus for frameshift mutations.
    df_best["Bonus"] = np.where(df_best["filter: frameshift"] == 1, bonus_multiplier, 1)

    # Compute final Ranking_Score using RNA_Score.
    df_best["Ranking_Score"] = (
        df_best["Sum_MHC_Score"]
        * df_best["DNA_Score"]
        * df_best["RNA_Score"]
        * df_best["Bonus"]
    )
    logger.info("Computed Ranking_Score for all selected epitopes.")

    # Sort by Ranking_Score descending.
    df_best.sort_values("Ranking_Score", ascending=False, inplace=True)
    if df_best["Ranking_Score"].max() == 0:
        logger.warning("All ranking scores are 0. Check thresholds and input data.")

    # Write full ranked TSV.
    try:
        df_best.to_csv(all_output, sep="\t", index=False)
        logger.info("Wrote all epitopes to {}", all_output)
    except Exception as e:
        logger.error("Error writing {}: {}", all_output, e)

    # Write top-n TSV.
    try:
        top_df = df_best.head(top_n)
        top_df.to_csv(top_output, sep="\t", index=False)
        logger.info("Wrote top {} epitopes to {}", top_n, top_output)
    except Exception as e:
        logger.error("Error writing {}: {}", top_output, e)

    # Log summary counts for HLA Allele and Gene Name among the top n.
    if "HLA Allele" in df_best.columns:
        hla_counts = top_df["HLA Allele"].value_counts()
        logger.info("Top {} HLA Allele counts:\n{}", top_n, hla_counts.to_string())
    else:
        logger.warning("Column 'HLA Allele' not found.")
    if "Gene Name" in df_best.columns:
        gene_counts = top_df["Gene Name"].value_counts()
        logger.info("Top {} Gene Name counts:\n{}", top_n, gene_counts.to_string())
    else:
        logger.warning("Column 'Gene Name' not found.")

    logger.info("Processing complete.")


if __name__ == "__main__":
    argh.dispatch_command(main)
