#!/usr/bin/env python3
"""
A tool to process TSV files of predicted mutant epitopes and rank them.

Pipeline:
  1) Load CLI arguments (possibly merged with defaults from a JSON/YAML file).
  2) Read a TSV file of predicted mutant epitopes.
  3) Apply basic filters:
       - Normal VAF <= normal_vaf_max
       - Tumor DNA Depth >= tumor_dna_depth_min
       - Normal Depth >= normal_depth_min
       - Tumor DNA VAF >= tumor_dna_vaf_min
  4) Compute "Mutant_RNA_Expression" = (Tumor RNA VAF × Gene Expression).
  5) For each mutation, select up to max_peptides_per_mutation transcripts:
       - Filter to transcripts with Transcript Expression ≥ 90% of the group's maximum.
       - If any candidate is protein_coding, restrict to those.
       - Use TSL (transcript support level) for tie-breaking (with "NA" treated as 6).
       - Then, use Transcript Length (descending) as a tie-breaker, followed by Transcript (alphabetically).
  6) Compute boolean columns for MHC thresholds and Sum_MHC_Score.
  7) Compute additional scores (DNA_Score, RNA_Score using Mutant_RNA_Expression, frameshift bonus)
       and the final Ranking_Score.
  8) Construct a new "Mutation Description" column as "Gene Name" + "_" + "Mutation".
  9) Collapse duplicate peptides in two steps:
       a) For each unique peptide (identified by the column specified by col_mt_epitope_seq),
          group by "Mutation Description" and select the mutation with the highest Mutant_RNA_Expression.
       b) Then, within that mutation, select the row (i.e. the HLA allele) with the highest Sum_MHC_Score.
 10) Finally, re‑sort the collapsed results by Ranking_Score (descending) before writing outputs.
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
    Treat "NA" (case-insensitive) as 6.
    """
    tsl_str = str(tsl).strip().upper()
    if tsl_str == "NA":
        return 6.0
    try:
        return float(tsl)
    except ValueError:
        m = re.search(r"\d+", tsl_str)
        if m:
            return float(m.group(0))
    return np.inf


def adaptive_thresholds(min_val: float, max_val: float, num: int) -> np.ndarray:
    """
    Generate adaptive thresholds for MHCflurry presentation score.
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


def generate_thresholds(
    min_val: float, max_val: float, num: int, step_type: str, reverse: bool = False
) -> np.ndarray:
    """
    Generate an array of thresholds based on the given parameters.
    """
    stype = step_type.lower()
    if stype == "adaptive":
        vals = adaptive_thresholds(min_val, max_val, num)
    elif stype == "linear":
        vals = (
            np.linspace(max_val, min_val, num)
            if reverse
            else np.linspace(min_val, max_val, num)
        )
    elif stype == "log2":
        vals = (
            np.geomspace(max_val, min_val, num=num)
            if reverse
            else np.geomspace(min_val, max_val, num=num)
        )
    else:
        logger.error("Invalid step type: {}", step_type)
        sys.exit(1)
    return vals


def select_transcripts(
    group: pd.DataFrame, max_peptides: int, log_ctx: logger
) -> pd.DataFrame:
    """
    Select up to max_peptides transcripts for a given mutation group.

    Selection steps:
      1. Convert Transcript Expression and Transcript Length to numeric.
      2. Keep transcripts with Transcript Expression within 90% of the group's maximum.
      3. If any candidate is protein_coding, restrict to those.
      4. Compute a numeric TSL (treating "NA" as 6).
      5. Sort candidates by:
           - Sum_MHC_Score (descending),
           - TSL_numeric (ascending),
           - Transcript Length (descending),
           - Transcript (ascending).
      6. Return the top max_peptides rows.
    """
    group = group.copy()
    group["Transcript Expression"] = pd.to_numeric(
        group["Transcript Expression"], errors="coerce"
    )
    group["Transcript Length"] = pd.to_numeric(
        group["Transcript Length"], errors="coerce"
    )
    max_expr = group["Transcript Expression"].max()
    expr_threshold = 0.9 * max_expr
    candidates = group[group["Transcript Expression"] >= expr_threshold]

    if (candidates["Biotype"] == "protein_coding").any():
        candidates = candidates[candidates["Biotype"] == "protein_coding"]

    candidates["TSL_numeric"] = candidates["Transcript Support Level"].apply(
        get_tsl_value
    )
    # Sort by Sum_MHC_Score descending, TSL_numeric ascending, Transcript Length descending, and then Transcript ascending.
    candidates = candidates.sort_values(
        by=["Sum_MHC_Score", "TSL_numeric", "Transcript Length", "Transcript"],
        ascending=[False, True, False, True],
    )

    mutation_id = group.iloc[0][
        ["Chromosome", "Start", "Stop", "Reference", "Variant", "Gene Name"]
    ].to_dict()
    log_ctx.info(
        "Mutation {}: selected {} transcripts out of {} candidates after filtering",
        mutation_id,
        min(len(candidates), max_peptides),
        len(candidates),
    )
    return candidates.head(max_peptides)


def merge_args(cli_args: dict, load_defaults=None):
    """
    Merge CLI arguments with built-in defaults and optionally loaded defaults.
    Priority: built-in defaults < loaded defaults < CLI args.
    """
    builtin_defaults = {
        k: v.default
        for k, v in inspect.signature(main).parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    loaded_defaults = (
        load_defaults_from_file(load_defaults) if load_defaults is not None else {}
    )
    merged = {}
    for key in cli_args:
        if key in builtin_defaults:
            merged[key] = (
                loaded_defaults.get(key, cli_args[key])
                if cli_args[key] == builtin_defaults[key]
                else cli_args[key]
            )
        else:
            merged[key] = cli_args[key]
    logger.info("Merged CLI arguments:")
    for k, v in merged.items():
        logger.info("  {}: {}", k, v)
    return merged


def rename_input_columns(merged_args: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename user-provided column names to canonical names.
    """
    col_mapping = {
        merged_args["col_chromosome"]: "Chromosome",
        merged_args["col_start"]: "Start",
        merged_args["col_stop"]: "Stop",
        merged_args["col_reference"]: "Reference",
        merged_args["col_variant"]: "Variant",
        merged_args["col_transcript"]: "Transcript",
        merged_args["col_transcript_support"]: "Transcript Support Level",
        merged_args["col_transcript_expression"]: "Transcript Expression",
        merged_args["col_biotype"]: "Biotype",
        merged_args[
            "col_protein_position"
        ]: "Protein Position",  # Still available if needed elsewhere.
        merged_args["col_transcript_length"]: "Transcript Length",
        merged_args["col_gene_name"]: "Gene Name",
        merged_args["col_mutation"]: "Mutation",
        merged_args["col_mhcflurry_presentation"]: "MHCflurryEL Presentation MT Score",
        merged_args["col_netmhcpan_el"]: "NetMHCpanEL MT Percentile",
        merged_args[
            "col_mhcflurry_presentation_percentile"
        ]: "MHCflurryEL Presentation MT Percentile",
        merged_args["col_netmhcpan_ba"]: "NetMHCpan MT IC50 Score",
        merged_args["col_netmhcpan_percentile"]: "NetMHCpan MT Percentile",
        merged_args["col_tumor_dna_vaf"]: "Tumor DNA VAF",
        merged_args["col_hla_allele"]: "HLA Allele",
        # New columns:
        merged_args["col_normal_vaf"]: "Normal VAF",
        merged_args["col_tumor_dna_depth"]: "Tumor DNA Depth",
        merged_args["col_normal_depth"]: "Normal Depth",
        merged_args["col_tumor_rna_vaf"]: "Tumor RNA VAF",
        merged_args["col_gene_expression"]: "Gene Expression",
        merged_args["col_mt_epitope_seq"]: "MT Epitope Seq",
    }
    return df.rename(columns=col_mapping, inplace=False)


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
    min_threshold_netmhcpan_el_percentile: float = 0.25,
    max_threshold_netmhcpan_el_percentile: float = 2,
    num_threshold_netmhcpan_el_percentile: int = 4,
    step_type_netmhcpan_el: str = "log2",
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
    # New filters:
    normal_vaf_max: float = 0.01,
    tumor_dna_depth_min: int = 30,
    normal_depth_min: int = 30,
    tumor_dna_vaf_min: float = 0.025,
    # New argument: allow up to this many peptides per mutation.
    max_peptides_per_mutation: int = 3,
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
    col_transcript_length: str = "Transcript Length",
    col_gene_name: str = "Gene Name",
    col_mutation: str = "Mutation",
    col_mhcflurry_presentation: str = "MHCflurryEL Presentation MT Score",
    col_netmhcpan_el: str = "NetMHCpanEL MT Percentile",
    col_mhcflurry_presentation_percentile: str = "MHCflurryEL Presentation MT Percentile",
    col_netmhcpan_ba: str = "NetMHCpan MT IC50 Score",
    col_netmhcpan_percentile: str = "NetMHCpan MT Percentile",
    col_tumor_dna_vaf: str = "Tumor DNA VAF",
    col_hla_allele: str = "HLA Allele",
    col_normal_vaf: str = "Normal VAF",
    col_tumor_dna_depth: str = "Tumor DNA Depth",
    col_normal_depth: str = "Normal Depth",
    col_tumor_rna_vaf: str = "Tumor RNA VAF",
    col_gene_expression: str = "Gene Expression",
    # New peptide column parameter:
    col_mt_epitope_seq: str = "MT Epitope Seq",
    # Defaults file options:
    load_defaults: str = None,
    save_args: str = None,
    save_args_format: str = "json",
) -> None:
    """
    Process the mutant epitope TSV file and output ranked epitopes.
    """
    cli_args = {k: locals()[k] for k in inspect.signature(main).parameters.keys()}
    merged = merge_args(cli_args=cli_args, load_defaults=load_defaults)

    if merged.get("save_args") is not None:
        save_args_to_file(
            merged, merged["save_args"], merged.get("save_args_format", "json")
        )

    # --- Use merged values ---
    input_file = merged["input_file"]
    all_output = merged["all_output"]
    top_output = merged["top_output"]
    top_n = merged["top_n"]
    bonus_multiplier = merged["bonus_multiplier"]
    log_file = merged["log_file"]
    rna_transform = merged["rna_transform"]
    normal_vaf_max = merged["normal_vaf_max"]
    tumor_dna_depth_min = merged["tumor_dna_depth_min"]
    normal_depth_min = merged["normal_depth_min"]
    tumor_dna_vaf_min = merged["tumor_dna_vaf_min"]
    max_peptides_per_mutation = merged["max_peptides_per_mutation"]

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
    min_threshold_netmhcpan_el_percentile = merged[
        "min_threshold_netmhcpan_el_percentile"
    ]
    max_threshold_netmhcpan_el_percentile = merged[
        "max_threshold_netmhcpan_el_percentile"
    ]
    num_threshold_netmhcpan_el_percentile = merged[
        "num_threshold_netmhcpan_el_percentile"
    ]
    step_type_netmhcpan_el = merged["step_type_netmhcpan_el"]
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

    # --- Rename input columns ---
    df = rename_input_columns(merged, df)

    # --- Construct "Mutation Description" ---
    if "Mutation Description" not in df.columns:
        df["Mutation Description"] = (
            df["Gene Name"].astype(str) + "_" + df["Mutation"].astype(str)
        )

    # Check required columns.
    required_columns = [
        "Chromosome",
        "Start",
        "Stop",
        "Reference",
        "Variant",
        "Transcript",
        "Transcript Support Level",
        "Transcript Expression",
        "Transcript Length",
        "Biotype",
        "Protein Position",
        "Gene Name",
        "Mutation",
        "Mutation Description",
        "MHCflurryEL Presentation MT Score",
        "NetMHCpanEL MT Percentile",
        "MHCflurryEL Presentation MT Percentile",
        "NetMHCpan MT IC50 Score",
        "NetMHCpan MT Percentile",
        "Tumor DNA VAF",
        "HLA Allele",
        "Normal VAF",
        "Tumor DNA Depth",
        "Normal Depth",
        "Tumor RNA VAF",
        "Gene Expression",
        "MT Epitope Seq",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error("Missing required columns: {}", missing)
        sys.exit(1)

    # --- Apply new filters ---
    df_before = len(df)
    df = df[df["Normal VAF"] <= normal_vaf_max]
    logger.info(
        "Filtered by Normal VAF <= {}: {} -> {} rows",
        normal_vaf_max,
        df_before,
        len(df),
    )

    df_before = len(df)
    df = df[df["Tumor DNA Depth"] >= tumor_dna_depth_min]
    logger.info(
        "Filtered by Tumor DNA Depth >= {}: {} -> {} rows",
        tumor_dna_depth_min,
        df_before,
        len(df),
    )

    df_before = len(df)
    df = df[df["Normal Depth"] >= normal_depth_min]
    logger.info(
        "Filtered by Normal Depth >= {}: {} -> {} rows",
        normal_depth_min,
        df_before,
        len(df),
    )

    df_before = len(df)
    df = df[df["Tumor DNA VAF"] >= tumor_dna_vaf_min]
    logger.info(
        "Filtered by Tumor DNA VAF >= {}: {} -> {} rows",
        tumor_dna_vaf_min,
        df_before,
        len(df),
    )

    # --- Compute Mutant_RNA_Expression ---
    df["Mutant_RNA_Expression"] = df["Tumor RNA VAF"] * df["Gene Expression"]

    # === Compute MHC boolean columns ===
    thr_mhcflurry_pres = generate_thresholds(
        min_threshold_mhcflurry_presentation_score,
        max_threshold_mhcflurry_presentation_score,
        num_threshold_mhcflurry_presentation_score,
        step_type_mhcflurry_pres,
        reverse=(step_type_mhcflurry_pres.lower() == "log2"),
    )
    thr_netmhcpan_el = generate_thresholds(
        min_threshold_netmhcpan_el_percentile,
        max_threshold_netmhcpan_el_percentile,
        num_threshold_netmhcpan_el_percentile,
        step_type_netmhcpan_el,
        reverse=True,
    )
    thr_mhcflurry_pres_pct = generate_thresholds(
        min_threshold_mhcflurry_presentation_percentile,
        max_threshold_mhcflurry_presentation_percentile,
        num_threshold_mhcflurry_presentation_percentile,
        step_type_mhcflurry_pres_pct,
        reverse=True,
    )
    thr_netmhcpan_ba = generate_thresholds(
        min_threshold_netmhcpan_ba,
        max_threshold_netmhcpan_ba,
        num_threshold_netmhcpan_ba,
        step_type_netmhcpan_ba,
        reverse=True,
    )
    thr_netmhcpan_pct = generate_thresholds(
        min_threshold_netmhcpan_percentile,
        max_threshold_netmhcpan_percentile,
        num_threshold_netmhcpan_percentile,
        step_type_netmhcpan_pct,
        reverse=True,
    )

    thr_mhcflurry_pres = np.round(thr_mhcflurry_pres, 2)
    thr_netmhcpan_el = np.round(thr_netmhcpan_el, 2)
    thr_mhcflurry_pres_pct = np.round(thr_mhcflurry_pres_pct, 2)
    thr_netmhcpan_ba = np.round(thr_netmhcpan_ba, 2)
    thr_netmhcpan_pct = np.round(thr_netmhcpan_pct, 2)

    logger.info("Generated thresholds:")
    logger.info("  MHCflurry presentation > {}", thr_mhcflurry_pres)
    logger.info("  NetMHCpan EL percentile < {}", thr_netmhcpan_el)
    logger.info("  MHCflurry presentation percentile < {}", thr_mhcflurry_pres_pct)
    logger.info("  NetMHCpan BA < {}", thr_netmhcpan_ba)
    logger.info("  NetMHCpan percentile < {}", thr_netmhcpan_pct)

    bool_cols = []
    for thr in thr_mhcflurry_pres:
        colname = f"filter: mhcflurry presentation > {thr:.2f}"
        df[colname] = (df["MHCflurryEL Presentation MT Score"] > thr).astype(int)
        bool_cols.append(colname)
    for thr in thr_netmhcpan_el:
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

    df["Sum_MHC_Score"] = df[bool_cols].sum(axis=1)
    logger.info("Computed Sum_MHC_Score for all rows.")

    df_before = len(df)
    df = df[df["Sum_MHC_Score"] > 0]
    logger.info(
        "Filtered to rows with Sum_MHC_Score > 0: {} -> {} rows", df_before, len(df)
    )

    # === Group by mutation and select transcripts per mutation ===
    mutation_cols = ["Chromosome", "Start", "Stop", "Reference", "Variant", "Gene Name"]
    groups = df.groupby(mutation_cols, as_index=False)
    selected = []
    for name, group_df in groups:
        try:
            selected_group = select_transcripts(
                group_df, max_peptides_per_mutation, logger
            )
            selected.append(selected_group)
        except Exception as ex:
            logger.error("Error selecting transcripts for mutation {}: {}", name, ex)
    if not selected:
        logger.error("No transcripts selected after grouping; exiting.")
        sys.exit(1)
    df_selected = pd.concat(selected, ignore_index=True)
    logger.info(
        "Selected transcripts from {} mutations.",
        df_selected[mutation_cols].drop_duplicates().shape[0],
    )

    # === Compute additional scores on df_selected ===
    df_selected["filter: frameshift"] = (
        df_selected["Mutation"]
        .str.contains("fs|frameshift", flags=re.IGNORECASE, na=False)
        .astype(int)
    )
    median_dna_vaf = df_selected["Tumor DNA VAF"].median()
    if median_dna_vaf == 0:
        logger.warning(
            "Median Tumor DNA VAF is 0; setting DNA_Score to 1 for all entries."
        )
        df_selected["DNA_Score"] = 1
    else:
        df_selected["DNA_Score"] = (df_selected["Tumor DNA VAF"] / median_dna_vaf).clip(
            upper=1
        )
    try:
        expr = df_selected["Mutant_RNA_Expression"].astype(float)
    except Exception as e:
        logger.error("Error converting Mutant_RNA_Expression to float: {}", e)
        sys.exit(1)
    rna_trans = rna_transform.lower()
    if rna_trans == "linear":
        df_selected["RNA_Score"] = expr
    elif rna_trans == "sqrt":
        df_selected["RNA_Score"] = np.sqrt(expr)
    elif rna_trans == "log2":
        df_selected["RNA_Score"] = np.log2(expr + 1)
    else:
        logger.error("Invalid RNA transform: {}", rna_transform)
        sys.exit(1)
    df_selected["Bonus"] = np.where(
        df_selected["filter: frameshift"] == 1, bonus_multiplier, 1
    )
    df_selected["Ranking_Score"] = (
        df_selected["Sum_MHC_Score"]
        * df_selected["DNA_Score"]
        * df_selected["RNA_Score"]
        * df_selected["Bonus"]
    )
    logger.info("Computed Ranking_Score for all selected epitopes.")

    df_selected.sort_values("Ranking_Score", ascending=False, inplace=True)
    if df_selected["Ranking_Score"].max() == 0:
        logger.warning("All ranking scores are 0. Check thresholds and input data.")

    # === Collapse duplicate peptides in two steps ===
    # Use the peptide column as specified by the parameter.
    if merged["col_mt_epitope_seq"] not in df_selected.columns:
        logger.error(
            "Required peptide column '{}' not found in data.",
            merged["col_mt_epitope_seq"],
        )
        sys.exit(1)
    peptide_col = merged["col_mt_epitope_seq"]

    def collapse_peptide(group):
        # Collapse across mutations using "Mutation Description"
        muts = group.groupby("Mutation Description", as_index=False)[
            "Mutant_RNA_Expression"
        ].max()
        best_mut_desc = muts.loc[muts["Mutant_RNA_Expression"].idxmax()][
            "Mutation Description"
        ]
        subset = group[group["Mutation Description"] == best_mut_desc]
        best_row = subset.loc[subset["Sum_MHC_Score"].idxmax()]
        return best_row

    df_collapsed = df_selected.groupby(peptide_col, as_index=False).apply(
        collapse_peptide
    )
    df_collapsed.reset_index(drop=True, inplace=True)
    # Re-sort collapsed results by Ranking_Score descending
    df_collapsed = df_collapsed.sort_values("Ranking_Score", ascending=False)

    logger.info(
        "Collapsed duplicate peptides: {} rows remain after collapsing.",
        df_collapsed.shape[0],
    )

    # Write outputs.
    try:
        df_collapsed.to_csv(all_output, sep="\t", index=False)
        logger.info("Wrote all collapsed epitopes to {}", all_output)
    except Exception as e:
        logger.error("Error writing {}: {}", all_output, e)
    try:
        top_df = df_collapsed.head(top_n)
        top_df.to_csv(top_output, sep="\t", index=False)
        logger.info("Wrote top {} collapsed epitopes to {}", top_n, top_output)
    except Exception as e:
        logger.error("Error writing {}: {}", top_output, e)
    if "HLA Allele" in df_collapsed.columns:
        hla_counts = top_df["HLA Allele"].value_counts()
        logger.info("Top {} HLA Allele counts:\n{}", top_n, hla_counts.to_string())
    else:
        logger.warning("Column 'HLA Allele' not found.")
    logger.info("Processing complete.")


if __name__ == "__main__":
    argh.dispatch_command(main)
