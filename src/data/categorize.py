"""
Categorize ProteinGym DMS assays by property type.

Maps 217 assays to 4 categories:
- Stability
- Binding
- Expression
- Activity

Challenge: ProteinGym's coarse_selection_type has "OrganismalFitness" (~90-100 assays)
which needs reclassification based on phenotype keywords.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def reclassify_organismal_fitness(
    row: pd.Series,
    keyword_maps: Dict[str, List[str]],
    default: str = "activity",
) -> Tuple[str, List[str], str]:
    """
    Reclassify an OrganismalFitness assay based on keyword matching.

    Args:
        row: DataFrame row with assay metadata
        keyword_maps: Dict mapping category -> list of keywords
        default: Default category if no keywords match

    Returns:
        (assigned_category, matched_keywords, confidence)

    Confidence levels:
        - "high": Direct coarse_selection_type match
        - "medium": Keyword match in phenotype/assay description
        - "low": Default assignment (no keyword match)
    """
    # Concatenate searchable text fields (lowercase for case-insensitive matching)
    searchable_fields = [
        str(row.get("raw_DMS_phenotype_name", "")),
        str(row.get("selection_assay", "")),
        str(row.get("selection_type", "")),
        str(row.get("molecule_name", "")),
        str(row.get("title", "")),
    ]
    text_to_search = " ".join(searchable_fields).lower()

    # Score each category by number of keyword matches
    category_scores = {}
    for category, keywords in keyword_maps.items():
        matched = [kw for kw in keywords if kw in text_to_search]
        if matched:
            category_scores[category] = (len(matched), matched)

    # Return category with most keyword matches
    if category_scores:
        best_category = max(category_scores, key=lambda k: category_scores[k][0])
        return best_category, category_scores[best_category][1], "medium"
    else:
        return default, [], "low"


def categorize_assays(
    reference_df: pd.DataFrame,
    config: Dict,
) -> Dict[str, List[str]]:
    """
    Categorize all DMS assays into property categories.

    Strategy:
        1. Direct mapping for Stability, Binding, Expression, Activity in coarse_selection_type
        2. Reclassify OrganismalFitness assays via keyword matching
        3. Exclude assays with < min_mutations_per_assay

    Args:
        reference_df: ProteinGym reference DataFrame
        config: Configuration dict

    Returns:
        categorization: Dict with keys:
            - stability: List[DMS_id]
            - binding: List[DMS_id]
            - expression: List[DMS_id]
            - activity: List[DMS_id]
            - excluded: List[DMS_id]
            - categorization_log: List[Dict] - detailed decision log
    """
    # Initialize categories
    categories = {
        "stability": [],
        "binding": [],
        "expression": [],
        "activity": [],
        "excluded": [],
    }

    categorization_log = []

    # Get configuration
    primary_map = config["categories"]["primary_map"]
    keyword_config = config["categories"]["organismal_fitness_reclassification"]
    min_mutations = config["data"]["min_mutations_per_assay"]

    # Convert keyword lists to lowercase and strip "_keywords" suffix
    keyword_maps = {}
    for key, keywords in keyword_config.items():
        if key == "default_category":
            continue
        # Extract category name by removing "_keywords" suffix
        category_name = key.replace("_keywords", "")
        keyword_maps[category_name] = [kw.lower() for kw in keywords]

    default_category = keyword_config.get("default_category", "activity")

    # Process each assay
    for _, row in reference_df.iterrows():
        dms_id = row["DMS_id"]
        coarse_type = row.get("coarse_selection_type", "Unknown")
        num_mutants = row.get("DMS_total_number_mutants", 0)

        # Check if assay meets minimum size requirement
        if num_mutants < min_mutations:
            categories["excluded"].append(dms_id)
            categorization_log.append({
                "dms_id": dms_id,
                "original_type": coarse_type,
                "assigned_category": "excluded",
                "reason": f"too_few_mutations ({num_mutants} < {min_mutations})",
                "matched_keywords": [],
                "confidence": "N/A",
            })
            continue

        # Categorize based on coarse_selection_type
        if coarse_type in primary_map:
            # Direct mapping
            category = primary_map[coarse_type].lower()
            confidence = "high"
            matched_keywords = []
        elif coarse_type == "OrganismalFitness":
            # Reclassify via keyword matching
            category, matched_keywords, confidence = reclassify_organismal_fitness(
                row, keyword_maps, default_category
            )
        else:
            # Unknown type - assign to default
            category = default_category
            confidence = "low"
            matched_keywords = []

        # Add to category
        categories[category].append(dms_id)

        # Log decision
        categorization_log.append({
            "dms_id": dms_id,
            "original_type": coarse_type,
            "assigned_category": category,
            "matched_keywords": matched_keywords,
            "confidence": confidence,
            "raw_DMS_phenotype_name": str(row.get("raw_DMS_phenotype_name", "")),
            "num_mutants": num_mutants,
        })

    # Add log to output
    categories["categorization_log"] = categorization_log

    # Print summary
    print("\n=== Assay Categorization Summary ===")
    for cat in ["stability", "binding", "expression", "activity"]:
        count = len(categories[cat])
        print(f"  {cat.capitalize()}: {count} assays")

    excluded_count = len(categories["excluded"])
    print(f"  Excluded: {excluded_count} assays")

    # Check minimum category sizes
    min_required = config["categories"]["min_assays_per_category"]
    warnings = []
    for cat in ["stability", "binding", "expression", "activity"]:
        count = len(categories[cat])
        if count < min_required:
            warnings.append(f"{cat.capitalize()} has only {count} assays (< {min_required})")

    if warnings:
        print("\n⚠ Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    # Print confidence distribution
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    for log_entry in categorization_log:
        conf = log_entry["confidence"]
        if conf in confidence_counts:
            confidence_counts[conf] += 1

    print("\nConfidence distribution:")
    for conf, count in confidence_counts.items():
        print(f"  {conf}: {count} assays")

    return categories


def save_categorization(
    categories: Dict,
    output_path: Path,
):
    """
    Save categorization results to JSON file.

    Args:
        categories: Categorization dict from categorize_assays()
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(categories, f, indent=2)

    print(f"\n✓ Saved categorization to: {output_path}")


def load_categorization(categorization_path: Path) -> Dict:
    """
    Load saved categorization from JSON file.

    Args:
        categorization_path: Path to categorization JSON

    Returns:
        categories: Categorization dict
    """
    with open(categorization_path) as f:
        categories = json.load(f)

    print(f"Loaded categorization from: {categorization_path}")
    for cat in ["stability", "binding", "expression", "activity"]:
        print(f"  {cat.capitalize()}: {len(categories[cat])} assays")

    return categories


def main():
    """CLI entry point for categorizing assays."""
    import yaml
    import argparse
    from .download import load_reference_file

    parser = argparse.ArgumentParser(description="Categorize ProteinGym assays by property")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load reference file
    reference_df = load_reference_file(config)

    # Categorize
    categories = categorize_assays(reference_df, config)

    # Save
    output_path = Path(config["data"]["processed_dir"]) / "category_assignments.json"
    save_categorization(categories, output_path)

    print("\n✓ Categorization complete!")


if __name__ == "__main__":
    main()
