#!/usr/bin/env python3
"""
A tool to process TSV files of predicted mutant epitopes and rank them.

It:
  - Reads an input TSV file.
  - Groups rows by mutation.
  - For each mutation, selects the best transcript based on expression, biotype,
    transcript support level, and mutation position.
  - Logs decisions using Loguru (including dependency versions).
  - Creates per-threshold boolean columns for several MHC metrics. For each metric you can choose a step type:
       * For MHCflurry Presentation Score (MT Score), step type can be:
             - "linear": evenly spaced between min and max,
             - "log2": geometric (log₂–spaced) between max and min,
             - "adaptive": an adaptive scheme that (for the default range) yields values as close as possible to [0.5, 0.75, 0.9, 0.95] for 4 thresholds and [0.5, 0.75, 0.85, 0.9, 0.95] for 5.
       * For the other metrics (NetMHCpanEL Percentile, MHCflurry Presentation Percentile,
         NetMHCpan MT IC50 Score, and NetMHCpan MT Percentile) the step type can be "linear" or "log2" (defaulting to log2).
  - Computes Sum_MHC_Score as the sum of these boolean columns.
  - Flags frameshift mutations and computes a DNA_Score.
  - Computes the final Ranking_Score.
  - Outputs the full ranking TSV and the top‑n epitopes TSV.
  - Logs a summary of HLA allele counts and gene counts for the top‑n epitopes.
  
Usage:
    python rank_epitopes.py --input-file INPUT.tsv [options]
"""

import sys
import re
import numpy as np
import pandas as pd
import argh
from loguru import logger
import loguru  # to log its version

# Configure loguru: log dependency versions.
logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO")
logger.info("Pandas version: {}", pd.__version__)
logger.info("NumPy version: {}", np.__version__)
logger.info("Loguru version: {}", loguru.__version__)


def get_tsl_value(tsl: str) -> float:
    """
    Convert a Transcript Support Level (TSL) value to a numeric value.

    Args:
        tsl (str): The TSL value, e.g. "TSL1" or "1".

    Returns:
        float: A numeric value corresponding to the TSL.
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
    Select the best transcript from a group of rows representing the same mutation.

    Selection is performed by:
      1. Filtering for transcripts with Transcript Expression within 10% of the maximum.
      2. Preferring transcripts with 'protein_coding' biotype.
      3. Choosing the transcript with the lowest Transcript Support Level.
      4. Choosing the transcript with the highest Protein Position.
      5. Finally, breaking ties lexicographically by Transcript.

    Args:
        group (pd.DataFrame): DataFrame of transcripts for a mutation.
        log_ctx (logger): Logger for recording decision steps.

    Returns:
        pd.Series: The chosen transcript row.
    """
    group = group.copy()
    decision_notes = []

    # Ensure numeric conversion.
    group["Transcript Expression"] = pd.to_numeric(
        group["Transcript Expression"], errors="coerce"
    )
    group["Protein Position"] = pd.to_numeric(
        group["Protein Position"], errors="coerce"
    )

    # 1. Filter for highest Transcript Expression (within 10% of max).
    max_expr = group["Transcript Expression"].max()
    expr_threshold = 0.9 * max_expr
    candidates = group[group["Transcript Expression"] >= expr_threshold]
    decision_notes.append(
        f"Max Expression: {max_expr:.3f}, threshold: {expr_threshold:.3f} (candidates: {len(candidates)})"
    )

    # 2. Prefer protein_coding biotype.
    if (candidates["Biotype"] == "protein_coding").any():
        candidates_pc = candidates[candidates["Biotype"] == "protein_coding"]
        decision_notes.append(f"Filtered by protein_coding: {len(candidates_pc)}")
        candidates = candidates_pc

    # 3. Prefer best (lowest) TSL.
    candidates["TSL_numeric"] = candidates["Transcript Support Level"].apply(
        get_tsl_value
    )
    best_tsl = candidates["TSL_numeric"].min()
    candidates = candidates[candidates["TSL_numeric"] == best_tsl]
    decision_notes.append(f"Best TSL: {best_tsl}, candidates: {len(candidates)}")

    # 4. Prefer highest Protein Position.
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

    The scheme fixes the first threshold to min_val and the last to max_val.
    For num>=3, the intermediate thresholds are chosen such that for the default range
    (min=0.5, max=0.95) you get values close to:
        n=4: [0.5, 0.75, 0.90, 0.95]
        n=5: [0.5, 0.75, 0.85, 0.90, 0.95]
    For other ranges, the intermediate thresholds are linearly spaced between
    positions 0.5556 and 0.8889 (the normalized anchors corresponding to the default).

    Args:
        min_val (float): The minimum threshold value.
        max_val (float): The maximum threshold value.
        num (int): The total number of thresholds.

    Returns:
        np.ndarray: An array of thresholds.
    """
    if num == 1:
        return np.array([min_val])
    if num == 2:
        return np.array([min_val, max_val])
    # For n>=3, fix first and last; interpolate intermediate normalized positions.
    intermediate = np.linspace(0.5556, 0.8889, num - 2)
    thresholds = (
        [min_val]
        + [min_val + (max_val - min_val) * t for t in intermediate]
        + [max_val]
    )
    return np.array(thresholds)


def main(
    input_file: str,
    all_output: str = "all-epitopes.tsv",
    top_output: str = "top-epitopes.tsv",
    top_n: int = 50,
    bonus_multiplier: int = 5,
    log_file: str = "log.txt",
    # For MHCflurry Presentation MT Score thresholds (condition: value > threshold)
    min_threshold_mhcflurry_presentation_score: float = 0.5,
    max_threshold_mhcflurry_presentation_score: float = 0.95,
    num_threshold_mhcflurry_presentation_score: int = 4,
    step_type_mhcflurry_pres: str = "adaptive",  # "linear", "log2", or "adaptive"
    # For NetMHCpanEL MT Percentile thresholds (condition: value < threshold)
    min_threshold_netmhcpanel_el_percentile: float = 0.25,
    max_threshold_netmhcpanel_el_percentile: float = 2,
    num_threshold_netmhcpanel_el_percentile: int = 4,
    step_type_netmhcpanel_el: str = "log2",  # "linear" or "log2"
    # For MHCflurry Presentation MT Percentile thresholds.
    min_threshold_mhcflurry_presentation_percentile: float = 0.25,
    max_threshold_mhcflurry_presentation_percentile: float = 2,
    num_threshold_mhcflurry_presentation_percentile: int = 4,
    step_type_mhcflurry_pres_pct: str = "log2",  # "linear" or "log2"
    # For NetMHCpan MT IC50 Score thresholds (condition: value < threshold)
    min_threshold_netmhcpan_ba: float = 125,
    max_threshold_netmhcpan_ba: float = 1000,
    num_threshold_netmhcpan_ba: int = 4,
    step_type_netmhcpan_ba: str = "log2",  # "linear" or "log2"
    # For NetMHCpan MT Percentile thresholds.
    min_threshold_netmhcpan_percentile: float = 0.25,
    max_threshold_netmhcpan_percentile: float = 2,
    num_threshold_netmhcpan_percentile: int = 4,
    step_type_netmhcpan_pct: str = "log2",  # "linear" or "log2"
) -> None:
    """
    Process the mutant epitope TSV file and output ranked epitopes.

    The function:
      1. Reads the input TSV file.
      2. Groups rows by mutation (Chromosome, Start, Stop, Reference, Variant, Gene Name).
      3. For each mutation, selects the best transcript based on:
         - Highest Transcript Expression (within 10% of max),
         - Preference for protein_coding,
         - Lowest Transcript Support Level,
         - Highest Protein Position,
         - Lexicographic tie-breaker on Transcript.
      4. Logs decisions and dependency versions.
      5. For five MHC metrics, creates per-threshold boolean columns. For the
         MHCflurry Presentation MT Score, the step type can be:
             "linear" (even spacing), "log2" (geometric spacing), or "adaptive"
             (an adaptive scheme that approximates default values such as
             [0.5, 0.75, 0.9, 0.95] when min=0.5 and max=0.95).
         For the other metrics, the step type can be "linear" or "log2" (default "log2").
      6. Computes Sum_MHC_Score as the sum of these boolean columns.
      7. Flags frameshift mutations and computes a DNA_Score.
      8. Computes the final Ranking_Score.
      9. Outputs the full ranking and the top‑n epitopes as TSV files.
      10. Logs summary counts for HLA alleles and gene names for the top‑n epitopes.

    Args:
        input_file (str): Path to the input TSV file.
        all_output (str): Output TSV file for all epitopes.
        top_output (str): Output TSV file for top epitopes.
        top_n (int): Number of top epitopes to output.
        bonus_multiplier (int): Bonus multiplier if the mutation is a frameshift.
        log_file (str): Path to the log file.
        (Threshold parameters follow; note that min and max may be anywhere between 0 and 1.)

    Returns:
        None
    """
    # Add file handler for loguru.
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
        logger.error("Error reading input file {}: {}", input_file, e)
        sys.exit(1)

    # Check for required columns.
    required_columns = [
        "Chromosome",
        "Start",
        "Stop",
        "Reference",
        "Variant",
        "Transcript",
        "Transcript Support Level",
        "Transcript Expression",
        "Biotype",
        "Protein Position",
        "Gene Name",
        "Mutation",
        "MHCflurryEL Presentation MT Score",
        "NetMHCpanEL MT Percentile",
        "MHCflurryEL Presentation MT Percentile",
        "NetMHCpan MT IC50 Score",
        "NetMHCpan MT Percentile",
        "Tumor DNA VAF",
        "HLA Allele",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error("Missing required columns: {}", missing)
        sys.exit(1)

    # Group by mutation.
    mutation_cols = ["Chromosome", "Start", "Stop", "Reference", "Variant", "Gene Name"]
    groups = df.groupby(mutation_cols, as_index=False)

    selected = []
    for name, group in groups:
        try:
            chosen = select_best_transcript(group, logger)
            selected.append(chosen)
        except Exception as ex:
            logger.error(
                "Error selecting best transcript for mutation {}: {}", name, ex
            )
    if not selected:
        logger.error("No transcripts selected; exiting.")
        sys.exit(1)
    df_best = pd.DataFrame(selected)
    logger.info("Selected best transcript for {} mutations.", len(df_best))

    # === Build threshold arrays ===
    # MHCflurry Presentation MT Score thresholds.
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
            "Invalid step type for MHCflurry presentation score: {}",
            step_type_mhcflurry_pres,
        )
        sys.exit(1)

    # For NetMHCpanEL MT Percentile thresholds.
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
            "Invalid step type for NetMHCpanEL percentile: {}", step_type_netmhcpanel_el
        )
        sys.exit(1)

    # For MHCflurry Presentation MT Percentile thresholds.
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
            "Invalid step type for MHCflurry presentation percentile: {}",
            step_type_mhcflurry_pres_pct,
        )
        sys.exit(1)

    # For NetMHCpan MT IC50 Score thresholds.
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
        logger.error("Invalid step type for NetMHCpan BA: {}", step_type_netmhcpan_ba)
        sys.exit(1)

    # For NetMHCpan MT Percentile thresholds.
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
            "Invalid step type for NetMHCpan percentile: {}", step_type_netmhcpan_pct
        )
        sys.exit(1)

    # Round threshold values to two decimals for display and for column names.
    thr_mhcflurry_pres = np.round(thr_mhcflurry_pres, 2)
    thr_netmhcpanel_el = np.round(thr_netmhcpanel_el, 2)
    thr_mhcflurry_pres_pct = np.round(thr_mhcflurry_pres_pct, 2)
    thr_netmhcpan_ba = np.round(thr_netmhcpan_ba, 2)
    thr_netmhcpan_pct = np.round(thr_netmhcpan_pct, 2)

    logger.info("Generated thresholds:")
    logger.info("MHCflurry Presentation Score thresholds (>): {}", thr_mhcflurry_pres)
    logger.info("NetMHCpanEL Percentile thresholds (<): {}", thr_netmhcpanel_el)
    logger.info(
        "MHCflurry Presentation Percentile thresholds (<): {}", thr_mhcflurry_pres_pct
    )
    logger.info("NetMHCpan BA Score thresholds (<) [log2 spaced]: {}", thr_netmhcpan_ba)
    logger.info("NetMHCpan Percentile thresholds (<): {}", thr_netmhcpan_pct)

    # === Create boolean columns for each threshold (with searchable prefixes) ===
    threshold_columns = []
    for thr in thr_mhcflurry_pres:
        colname = f"thr_mhcflurry_pres_gt_{thr:.2f}"
        df_best[colname] = (df_best["MHCflurryEL Presentation MT Score"] > thr).astype(
            int
        )
        threshold_columns.append(colname)
    for thr in thr_netmhcpanel_el:
        colname = f"thr_netmhcpanel_el_lt_{thr:.2f}"
        df_best[colname] = (df_best["NetMHCpanEL MT Percentile"] < thr).astype(int)
        threshold_columns.append(colname)
    for thr in thr_mhcflurry_pres_pct:
        colname = f"thr_mhcflurry_pres_pct_lt_{thr:.2f}"
        df_best[colname] = (
            df_best["MHCflurryEL Presentation MT Percentile"] < thr
        ).astype(int)
        threshold_columns.append(colname)
    for thr in thr_netmhcpan_ba:
        colname = f"thr_netmhcpan_ba_lt_{thr:.2f}"
        df_best[colname] = (df_best["NetMHCpan MT IC50 Score"] < thr).astype(int)
        threshold_columns.append(colname)
    for thr in thr_netmhcpan_pct:
        colname = f"thr_netmhcpan_pct_lt_{thr:.2f}"
        df_best[colname] = (df_best["NetMHCpan MT Percentile"] < thr).astype(int)
        threshold_columns.append(colname)

    # Compute Sum_MHC_Score from the threshold columns.
    df_best["Sum_MHC_Score"] = df_best[threshold_columns].sum(axis=1)
    logger.info("Computed Sum_MHC_Score for all epitopes.")

    # Flag frameshift mutations.
    df_best["Is_Frameshift"] = df_best["Mutation"].str.contains(
        "fs|frameshift", flags=re.IGNORECASE, na=False
    )

    # Compute DNA_Score: min(1, Tumor DNA VAF / median(Tumor DNA VAF)).
    median_dna_vaf = df_best["Tumor DNA VAF"].median()
    if median_dna_vaf == 0:
        logger.warning(
            "Median Tumor DNA VAF is 0; setting DNA_Score to 1 for all entries."
        )
        df_best["DNA_Score"] = 1
    else:
        df_best["DNA_Score"] = (df_best["Tumor DNA VAF"] / median_dna_vaf).clip(upper=1)

    # Apply bonus for frameshift mutations.
    df_best["Bonus"] = np.where(df_best["Is_Frameshift"], bonus_multiplier, 1)

    # Compute final Ranking_Score.
    df_best["Ranking_Score"] = (
        df_best["Sum_MHC_Score"]
        * df_best["DNA_Score"]
        * df_best["Transcript Expression"]
        * df_best["Bonus"]
    )
    logger.info("Computed Ranking_Score for all epitopes.")

    # Sort by Ranking_Score descending (higher is better).
    df_best.sort_values("Ranking_Score", ascending=False, inplace=True)

    if df_best["Ranking_Score"].max() == 0:
        logger.warning(
            "All ranking scores are 0. Please check thresholds and input data."
        )

    # Write the complete ranked file.
    try:
        df_best.to_csv(all_output, sep="\t", index=False)
        logger.info("Wrote all epitopes to {}", all_output)
    except Exception as e:
        logger.error("Error writing all epitopes file {}: {}", all_output, e)

    # Write top-N epitopes.
    try:
        top_df = df_best.head(top_n)
        top_df.to_csv(top_output, sep="\t", index=False)
        logger.info("Wrote top {} epitopes to {}", top_n, top_output)
    except Exception as e:
        logger.error("Error writing top epitopes file {}: {}", top_output, e)

    # Log summary counts for HLA allele and gene (Gene Name) for the top n.
    if "HLA Allele" in df_best.columns:
        hla_counts = top_df["HLA Allele"].value_counts()
        logger.info("Top {} HLA Allele counts:\n{}", top_n, hla_counts.to_string())
    else:
        logger.warning("Column 'HLA Allele' not found in input data.")

    if "Gene Name" in df_best.columns:
        gene_counts = top_df["Gene Name"].value_counts()
        logger.info("Top {} Gene Name counts:\n{}", top_n, gene_counts.to_string())
    else:
        logger.warning("Column 'Gene Name' not found in input data.")

    logger.info("Processing complete.")


if __name__ == "__main__":
    argh.dispatch_command(main)
