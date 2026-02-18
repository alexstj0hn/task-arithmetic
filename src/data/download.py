"""
Download and validate ProteinGym v1.3 benchmark data.

Downloads:
- DMS_substitutions.csv (reference metadata, 217 assays)
- DMS_ProteinGym_substitutions.zip (~500MB, 217 individual assay CSVs)
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict
import zipfile
import pandas as pd
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path, desc: str = "Downloading"):
    """
    Download a file from URL with progress bar.

    Args:
        url: URL to download from
        output_path: Local path to save to
        desc: Description for progress bar
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)

    print(f"Downloaded to: {output_path}")


def download_proteingym(config: Dict, force: bool = False) -> Path:
    """
    Download ProteinGym v1.3 benchmark data.

    Args:
        config: Configuration dict with data.download_url, data.reference_url, data.raw_dir
        force: If True, re-download even if files exist

    Returns:
        raw_dir: Path to directory containing downloaded data

    Directory structure after download:
        data/raw/
        ├── DMS_substitutions.csv  (reference file)
        └── DMS_ProteinGym_substitutions/  (217 assay CSVs)
            ├── A0A140D2T1_ZIKV_Sourisseau_2019.csv
            ├── ACE2_HUMAN_Chan_2020.csv
            └── ...
    """
    raw_dir = Path(config["data"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download reference CSV
    reference_url = config["data"]["reference_url"]
    reference_path = raw_dir / "DMS_substitutions.csv"

    if not reference_path.exists() or force:
        print(f"\nDownloading reference file from {reference_url}")
        download_url(reference_url, reference_path, desc="Reference CSV")
    else:
        print(f"\nReference file already exists: {reference_path}")

    # Download DMS assays ZIP
    zip_url = config["data"]["download_url"]
    zip_path = raw_dir / "DMS_ProteinGym_substitutions.zip"
    assays_dir = raw_dir / "DMS_ProteinGym_substitutions"

    if not assays_dir.exists() or force:
        # Download ZIP if needed
        if not zip_path.exists() or force:
            print(f"\nDownloading DMS assays from {zip_url}")
            print("This may take a few minutes (~500MB)...")
            download_url(zip_url, zip_path, desc="DMS Assays ZIP")

        # Unzip
        print(f"\nExtracting ZIP to {assays_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)

        # Clean up ZIP file
        zip_path.unlink()
        print(f"Removed ZIP file: {zip_path}")
    else:
        print(f"\nDMS assays directory already exists: {assays_dir}")

    # Validate
    print("\nValidating download...")
    reference_df = pd.read_csv(reference_path)
    validation_results = validate_download(assays_dir, reference_df)

    if validation_results["all_found"]:
        print(f"✓ Validation successful!")
        print(f"  Found {validation_results['found_count']} / {validation_results['expected_count']} assay files")
    else:
        print(f"✗ Validation warning:")
        print(f"  Found {validation_results['found_count']} / {validation_results['expected_count']} assay files")
        if validation_results["missing"]:
            print(f"  Missing files ({len(validation_results['missing'])}):")
            for missing in validation_results["missing"][:10]:  # Show first 10
                print(f"    - {missing}")

    return raw_dir


def validate_download(assays_dir: Path, reference_df: pd.DataFrame) -> Dict:
    """
    Validate that all expected assay files are present.

    Args:
        assays_dir: Directory containing assay CSV files
        reference_df: Reference dataframe with DMS_filename column

    Returns:
        validation_results: Dict with keys:
            - all_found: bool
            - expected_count: int
            - found_count: int
            - missing: List[str]
            - file_mapping: Dict[DMS_id -> file_path]
    """
    # Get expected filenames from reference
    expected_files = set(reference_df["DMS_filename"].values)

    # Get actual files
    actual_files = {f.name for f in assays_dir.glob("*.csv")}

    # Find missing
    missing = expected_files - actual_files
    found_count = len(expected_files) - len(missing)

    # Create DMS_id -> file_path mapping for found files
    file_mapping = {}
    for _, row in reference_df.iterrows():
        filename = row["DMS_filename"]
        if filename in actual_files:
            file_mapping[row["DMS_id"]] = assays_dir / filename

    return {
        "all_found": len(missing) == 0,
        "expected_count": len(expected_files),
        "found_count": found_count,
        "missing": sorted(list(missing)),
        "file_mapping": file_mapping,
    }


def load_reference_file(config: Dict) -> pd.DataFrame:
    """
    Load the ProteinGym reference CSV file.

    Args:
        config: Configuration dict

    Returns:
        reference_df: DataFrame with assay metadata (217 rows)

    Columns include:
        - DMS_id: Unique assay identifier
        - DMS_filename: CSV filename
        - UniProt_ID: Protein UniProt ID
        - target_seq: Wild-type protein sequence
        - coarse_selection_type: Coarse property category
        - raw_DMS_phenotype_name: Original phenotype description
        - (and 40+ more metadata columns)
    """
    raw_dir = Path(config["data"]["raw_dir"])
    reference_path = raw_dir / "DMS_substitutions.csv"

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference file not found: {reference_path}\n"
            f"Run download_proteingym() first."
        )

    df = pd.read_csv(reference_path)
    print(f"Loaded reference file: {reference_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:10]}...")  # Show first 10 columns

    return df


def main():
    """CLI entry point for downloading data."""
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description="Download ProteinGym v1.3 data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Download
    raw_dir = download_proteingym(config, force=args.force)
    print(f"\n✓ Download complete. Data saved to: {raw_dir}")


if __name__ == "__main__":
    main()
