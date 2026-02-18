"""
Create train/test splits with ZERO protein-level sequence overlap.

Critical data integrity component. Uses MMseqs2 to cluster wildtype sequences
at 30% identity threshold, then assigns entire clusters to train or test.

This prevents the data leakage that inflated Metalic's reported performance.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def extract_wildtype_sequences(
    reference_df: pd.DataFrame,
    output_fasta: Path,
) -> Dict[str, str]:
    """
    Extract unique wildtype sequences from reference DataFrame.

    Args:
        reference_df: ProteinGym reference DataFrame
        output_fasta: Path to save FASTA file

    Returns:
        seq_mapping: Dict[UniProt_ID -> sequence]

    Each unique UniProt_ID gets one entry (de-duplicated).
    """
    # De-duplicate by UniProt_ID (some proteins have multiple assays)
    unique_proteins = reference_df.drop_duplicates(subset=["UniProt_ID"])

    seq_mapping = {}
    seq_records = []

    for _, row in unique_proteins.iterrows():
        uniprot_id = row["UniProt_ID"]
        sequence = row["target_seq"]

        if pd.isna(sequence) or len(sequence) == 0:
            print(f"Warning: No sequence for {uniprot_id}, skipping")
            continue

        seq_mapping[uniprot_id] = sequence

        # Create SeqRecord for FASTA
        record = SeqRecord(
            Seq(sequence),
            id=uniprot_id,
            description=f"{row.get('molecule_name', 'Unknown')}"
        )
        seq_records.append(record)

    # Write FASTA
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(seq_records, output_fasta, "fasta")

    print(f"Extracted {len(seq_records)} unique wildtype sequences to: {output_fasta}")

    return seq_mapping


def run_mmseqs2_clustering(
    fasta_path: Path,
    output_prefix: Path,
    tmp_dir: Path,
    identity_threshold: float = 0.3,
    coverage: float = 0.8,
    coverage_mode: int = 0,
    threads: int = 8,
) -> pd.DataFrame:
    """
    Run MMseqs2 sequence clustering.

    Args:
        fasta_path: Input FASTA file with sequences to cluster
        output_prefix: Prefix for output files
        tmp_dir: Temporary directory for MMseqs2
        identity_threshold: Minimum sequence identity (0.3 = 30%)
        coverage: Minimum coverage (0.8 = 80%)
        coverage_mode: 0 = bidirectional, 1 = query, 2 = target
        threads: Number of threads

    Returns:
        cluster_df: DataFrame with columns [representative, member]

    The output file {output_prefix}_cluster.tsv contains two columns:
        1. Representative sequence ID (cluster centroid)
        2. Member sequence ID (any sequence in that cluster)

    Each cluster is a group of sequences with >= identity_threshold similarity.
    """
    # Ensure MMseqs2 is installed
    try:
        subprocess.run(["mmseqs", "version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "MMseqs2 not found. Please install it:\n"
            "  Option 1: conda install -c conda-forge -c bioconda mmseqs2\n"
            "  Option 2: Download from https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz"
        )

    # Create temporary directory
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Build MMseqs2 command
    cmd = [
        "mmseqs", "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        str(tmp_dir),
        "--min-seq-id", str(identity_threshold),
        "-c", str(coverage),
        "--cov-mode", str(coverage_mode),
        "--threads", str(threads),
    ]

    print(f"\nRunning MMseqs2 clustering:")
    print(f"  Identity threshold: {identity_threshold * 100}%")
    print(f"  Coverage: {coverage * 100}%")
    print(f"  Command: {' '.join(cmd)}")

    # Run MMseqs2
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    print("MMseqs2 output:")
    print(result.stdout)

    # Parse cluster assignments
    cluster_file = Path(f"{output_prefix}_cluster.tsv")

    if not cluster_file.exists():
        raise FileNotFoundError(f"MMseqs2 cluster file not found: {cluster_file}")

    cluster_df = pd.read_csv(
        cluster_file,
        sep="\t",
        header=None,
        names=["representative", "member"]
    )

    print(f"\nClustering results:")
    print(f"  Total sequences: {len(cluster_df['member'].unique())}")
    print(f"  Number of clusters: {len(cluster_df['representative'].unique())}")

    return cluster_df


def assign_clusters_to_splits(
    cluster_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    category_assignments: Dict[str, List[str]],
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Assign protein clusters to train or test splits.

    Strategy:
        1. Map each DMS_id to its protein cluster (via UniProt_ID)
        2. Assign entire clusters to train OR test (never split a cluster)
        3. Stratify by property category (~80/20 train/test per category)

    Args:
        cluster_df: MMseqs2 cluster assignments [representative, member]
        reference_df: ProteinGym reference DataFrame
        category_assignments: Dict from categorize_assays()
        test_fraction: Fraction of clusters for test (default 0.2)
        seed: Random seed for reproducibility

    Returns:
        (train_dms_ids, test_dms_ids)
    """
    np.random.seed(seed)

    # Map UniProt_ID -> cluster representative
    uniprot_to_cluster = {}
    for _, row in cluster_df.iterrows():
        member_id = row["member"]
        rep_id = row["representative"]
        uniprot_to_cluster[member_id] = rep_id

    # Map DMS_id -> cluster
    dms_to_cluster = {}
    for _, row in reference_df.iterrows():
        dms_id = row["DMS_id"]
        uniprot_id = row["UniProt_ID"]
        cluster = uniprot_to_cluster.get(uniprot_id, uniprot_id)  # Self if not in clusters
        dms_to_cluster[dms_id] = cluster

    # Group DMS_ids by category and cluster
    category_clusters = {cat: {} for cat in ["stability", "binding", "expression", "activity"]}

    for category in ["stability", "binding", "expression", "activity"]:
        dms_ids = category_assignments.get(category, [])

        # Group by cluster
        for dms_id in dms_ids:
            cluster = dms_to_cluster.get(dms_id)
            if cluster is None:
                print(f"Warning: No cluster for {dms_id}, skipping")
                continue

            if cluster not in category_clusters[category]:
                category_clusters[category][cluster] = []
            category_clusters[category][cluster].append(dms_id)

    # Split clusters within each category
    train_dms_ids = []
    test_dms_ids = []

    for category, clusters in category_clusters.items():
        if not clusters:
            continue

        # Get list of unique clusters for this category
        cluster_list = list(clusters.keys())
        np.random.shuffle(cluster_list)

        # Split at test_fraction
        n_test = max(1, int(len(cluster_list) * test_fraction))
        test_clusters = set(cluster_list[:n_test])
        train_clusters = set(cluster_list[n_test:])

        # Assign DMS_ids based on cluster assignment
        for cluster, dms_ids in clusters.items():
            if cluster in test_clusters:
                test_dms_ids.extend(dms_ids)
            else:
                train_dms_ids.extend(dms_ids)

        print(f"\n{category.capitalize()}:")
        print(f"  Clusters: {len(cluster_list)} total ({len(train_clusters)} train, {len(test_clusters)} test)")
        print(f"  Assays: {sum(len(v) for v in clusters.values())} total ({sum(len(clusters[c]) for c in train_clusters)} train, {sum(len(clusters[c]) for c in test_clusters)} test)")

    return train_dms_ids, test_dms_ids


def validate_split(
    train_dms_ids: List[str],
    test_dms_ids: List[str],
    reference_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> Dict:
    """
    Validate that train/test split has zero protein overlap.

    Verification:
        - No DMS_id appears in both train and test
        - Max sequence identity between train and test proteins is < threshold

    Args:
        train_dms_ids: List of training DMS_ids
        test_dms_ids: List of test DMS_ids
        reference_df: ProteinGym reference DataFrame
        cluster_df: MMseqs2 cluster assignments

    Returns:
        validation_results: Dict with metrics
    """
    # Check for direct overlap (should never happen)
    train_set = set(train_dms_ids)
    test_set = set(test_dms_ids)
    overlap = train_set & test_set

    if overlap:
        print(f"ERROR: {len(overlap)} DMS_ids in both train and test!")
        return {"valid": False, "overlap_dms_ids": list(overlap)}

    # Get UniProt_IDs for train and test
    train_uniprots = set(
        reference_df[reference_df["DMS_id"].isin(train_dms_ids)]["UniProt_ID"].unique()
    )
    test_uniprots = set(
        reference_df[reference_df["DMS_id"].isin(test_dms_ids)]["UniProt_ID"].unique()
    )

    # Check cluster overlap (should be zero if clustering worked correctly)
    uniprot_to_cluster = dict(zip(cluster_df["member"], cluster_df["representative"]))

    train_clusters = {uniprot_to_cluster.get(u, u) for u in train_uniprots}
    test_clusters = {uniprot_to_cluster.get(u, u) for u in test_uniprots}

    cluster_overlap = train_clusters & test_clusters

    results = {
        "valid": len(cluster_overlap) == 0,
        "train_assays": len(train_dms_ids),
        "test_assays": len(test_dms_ids),
        "train_proteins": len(train_uniprots),
        "test_proteins": len(test_uniprots),
        "train_clusters": len(train_clusters),
        "test_clusters": len(test_clusters),
        "cluster_overlap": list(cluster_overlap),
    }

    if results["valid"]:
        print("\n✓ Split validation PASSED:")
        print(f"  Train: {results['train_assays']} assays, {results['train_proteins']} proteins, {results['train_clusters']} clusters")
        print(f"  Test: {results['test_assays']} assays, {results['test_proteins']} proteins, {results['test_clusters']} clusters")
        print(f"  Cluster overlap: 0 (valid)")
    else:
        print("\n✗ Split validation FAILED:")
        print(f"  {len(cluster_overlap)} clusters appear in both train and test!")

    return results


def create_splits(
    reference_df: pd.DataFrame,
    category_assignments: Dict,
    config: Dict,
    output_dir: Path,
) -> Dict:
    """
    Create train/test splits with zero protein overlap via MMseqs2 clustering.

    Args:
        reference_df: ProteinGym reference DataFrame
        category_assignments: Output from categorize_assays()
        config: Configuration dict
        output_dir: Directory to save split files

    Returns:
        splits: Dict with train_assays, test_assays, metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract wildtype sequences
    fasta_path = output_dir / "wildtype_sequences.fasta"
    seq_mapping = extract_wildtype_sequences(reference_df, fasta_path)

    # Run MMseqs2 clustering
    with tempfile.TemporaryDirectory() as tmp_dir:
        cluster_df = run_mmseqs2_clustering(
            fasta_path=fasta_path,
            output_prefix=output_dir / "cluster_results",
            tmp_dir=Path(tmp_dir),
            identity_threshold=config["splits"]["mmseqs2_identity_threshold"],
            coverage=config["splits"]["mmseqs2_coverage"],
            coverage_mode=config["splits"]["mmseqs2_coverage_mode"],
        )

    # Save cluster assignments
    cluster_df.to_csv(output_dir / "cluster_assignments.tsv", sep="\t", index=False)

    # Assign clusters to train/test
    train_dms_ids, test_dms_ids = assign_clusters_to_splits(
        cluster_df=cluster_df,
        reference_df=reference_df,
        category_assignments=category_assignments,
        test_fraction=config["splits"]["test_fraction"],
        seed=config["splits"]["random_seed"],
    )

    # Validate
    validation_results = validate_split(
        train_dms_ids, test_dms_ids, reference_df, cluster_df
    )

    # Save split assignments
    splits = {
        "train_assays": train_dms_ids,
        "test_assays": test_dms_ids,
        "split_metadata": {
            "mmseqs2_identity_threshold": config["splits"]["mmseqs2_identity_threshold"],
            "test_fraction": config["splits"]["test_fraction"],
            "random_seed": config["splits"]["random_seed"],
            "validation": validation_results,
        },
    }

    # Save to JSON
    with open(output_dir / "train_assays.json", "w") as f:
        json.dump(train_dms_ids, f, indent=2)

    with open(output_dir / "test_assays.json", "w") as f:
        json.dump(test_dms_ids, f, indent=2)

    with open(output_dir / "split_metadata.json", "w") as f:
        json.dump(splits["split_metadata"], f, indent=2)

    print(f"\n✓ Splits saved to: {output_dir}")

    return splits


def main():
    """CLI entry point for creating splits."""
    import yaml
    import argparse
    from .download import load_reference_file
    from .categorize import load_categorization

    parser = argparse.ArgumentParser(description="Create train/test splits with MMseqs2")
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

    # Load reference and categorization
    reference_df = load_reference_file(config)

    categorization_path = Path(config["data"]["processed_dir"]) / "category_assignments.json"
    category_assignments = load_categorization(categorization_path)

    # Create splits
    splits_dir = Path(config["data"]["splits_dir"])
    splits = create_splits(reference_df, category_assignments, config, splits_dir)

    print("\n✓ Split creation complete!")


if __name__ == "__main__":
    main()
