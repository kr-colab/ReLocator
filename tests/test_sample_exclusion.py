"""Tests for sample exclusion functionality."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from locator import Locator


class TestSampleExclusion:
    """Test sample exclusion features."""

    def setup_method(self):
        """Create test data for each test."""
        # Create sample data
        self.n_samples = 20
        self.n_snps = 100

        # Generate sample IDs
        self.sample_ids = [f"sample_{i:03d}" for i in range(self.n_samples)]

        # Create sample metadata with some NA locations
        self.sample_data = pd.DataFrame(
            {
                "sampleID": self.sample_ids,
                "x": np.random.uniform(-10, 10, self.n_samples),
                "y": np.random.uniform(-10, 10, self.n_samples),
            }
        )
        # Make some samples have NA coordinates
        self.sample_data.loc[15:17, ["x", "y"]] = np.nan

        # Create genotype data
        self.genotypes = np.random.randint(0, 3, size=(self.n_snps, self.n_samples))

    def test_exclude_samples_from_list(self):
        """Test loading exclusions from a list."""
        # Create locator with excluded samples
        exclude_list = ["sample_001", "sample_005", "sample_010"]
        locator = Locator(
            {"exclude_samples": exclude_list, "sample_data": self.sample_data}
        )

        # Check that samples were loaded
        assert len(locator._excluded_sample_ids) == 3
        assert "sample_001" in locator._excluded_sample_ids
        assert "sample_005" in locator._excluded_sample_ids
        assert "sample_010" in locator._excluded_sample_ids

        # Check exclusion source
        assert locator._exclusion_source["sample_001"] == "config"

    def test_exclude_samples_from_file(self):
        """Test loading exclusions from a file."""
        # Create temporary exclusion file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# This is a comment\n")
            f.write("sample_002\n")
            f.write("sample_007\n")
            f.write("sample_012\n")
            f.write("\n")  # Empty line
            exclude_file = f.name

        try:
            # Create locator with exclusion file
            locator = Locator(
                {"exclude_samples": exclude_file, "sample_data": self.sample_data}
            )

            # Check that samples were loaded
            assert len(locator._excluded_sample_ids) == 3
            assert "sample_002" in locator._excluded_sample_ids
            assert "sample_007" in locator._excluded_sample_ids
            assert "sample_012" in locator._excluded_sample_ids

            # Check exclusion source
            assert locator._exclusion_source["sample_002"].startswith("file:")

        finally:
            os.unlink(exclude_file)

    def test_interactive_exclusion(self):
        """Test interactive exclude/include methods."""
        locator = Locator({"sample_data": self.sample_data})

        # Initially no exclusions
        assert len(locator._excluded_sample_ids) == 0

        # Exclude some samples
        locator.exclude_samples(["sample_003", "sample_008"], reason="quality_control")
        assert len(locator._excluded_sample_ids) == 2
        assert "sample_003" in locator._excluded_sample_ids
        assert locator._exclusion_source["sample_003"] == "quality_control"

        # Exclude single sample
        locator.exclude_samples("sample_011", reason="outlier")
        assert len(locator._excluded_sample_ids) == 3

        # Include sample back
        n_removed = locator.include_samples(["sample_003"])
        assert n_removed == 1
        assert len(locator._excluded_sample_ids) == 2
        assert "sample_003" not in locator._excluded_sample_ids

        # Clear all exclusions
        locator.clear_exclusions()
        assert len(locator._excluded_sample_ids) == 0

    def test_get_excluded_samples(self):
        """Test getting excluded samples as DataFrame."""
        locator = Locator({"sample_data": self.sample_data})

        # No exclusions initially
        df = locator.get_excluded_samples()
        assert len(df) == 0
        assert list(df.columns) == ["sampleID", "reason"]

        # Add some exclusions
        locator.exclude_samples(["sample_001", "sample_002"], reason="manual")
        locator.exclude_samples(["sample_003"], reason="high_error")

        df = locator.get_excluded_samples()
        assert len(df) == 3
        assert df[df["sampleID"] == "sample_001"]["reason"].iloc[0] == "manual"
        assert df[df["sampleID"] == "sample_003"]["reason"].iloc[0] == "high_error"

    def test_exclude_by_condition(self):
        """Test excluding samples by condition."""
        # Add an error column to sample data
        sample_data_with_error = self.sample_data.copy()
        sample_data_with_error["error"] = np.random.uniform(0, 150, self.n_samples)

        locator = Locator({"sample_data": self.sample_data})

        # Exclude samples with high error
        locator.exclude_samples_by_condition(
            lambda df: df["error"] > 100,
            sample_df=sample_data_with_error,
            reason="high_error",
        )

        # Check that some samples were excluded
        excluded_df = locator.get_excluded_samples()
        assert len(excluded_df) > 0
        assert all(excluded_df["reason"] == "high_error")

    def test_sample_status_with_exclusions(self):
        """Test that get_sample_status reports exclusions correctly."""
        # Create locator with exclusions
        exclude_list = ["sample_001", "sample_005", "sample_010"]
        locator = Locator(
            {
                "exclude_samples": exclude_list,
                "sample_data": self.sample_data,
                "verbose": 0,
            }
        )

        # Get sample status (don't pass sample_data to trigger exclusion)
        status = locator.get_sample_status(samples=np.array(self.sample_ids))

        # Check basic counts
        assert status["total"] == self.n_samples
        assert status["n_excluded"] == 3

        # Check that excluded samples with coords are counted
        assert status["n_excluded_with_coords"] == 3  # All excluded samples have coords
        # n_known is already post-exclusion, so n_available equals n_known
        assert status["n_available"] == status["n_known"]
        # Total known before exclusion would be n_known + n_excluded_with_coords
        assert (
            status["n_known"] + status["n_excluded_with_coords"] == 17
        )  # 20 total - 3 NA

    def test_check_data_reports_exclusions(self, capsys):
        """Test that check_data reports exclusions."""
        # Create locator with exclusions
        exclude_list = ["sample_001", "sample_005"]
        locator = Locator(
            {"exclude_samples": exclude_list, "sample_data": self.sample_data}
        )

        # Check data
        locator.check_data(self.genotypes, np.array(self.sample_ids))

        # Capture output
        captured = capsys.readouterr()
        assert "Excluded samples: 2" in captured.out
        assert "Excluded samples with coordinates: 2" in captured.out
        assert "Available samples for training:" in captured.out

    def test_sort_samples_excludes_correctly(self):
        """Test that sort_samples excludes samples."""
        # Create locator with exclusions
        exclude_list = ["sample_001", "sample_005", "sample_010"]
        locator = Locator(
            {
                "exclude_samples": exclude_list,
                "sample_data": self.sample_data,
                "verbose": 1,
            }
        )

        # Sort samples
        sorted_data, locs = locator.sort_samples(
            samples=np.array(self.sample_ids), reorder=True
        )

        # Check that excluded samples are removed
        assert len(sorted_data) == self.n_samples - 3
        assert "sample_001" not in sorted_data["sampleID"].values
        assert "sample_005" not in sorted_data["sampleID"].values
        assert "sample_010" not in sorted_data["sampleID"].values

        # Check that locations match
        assert len(locs) == len(sorted_data)

    def test_nonexistent_exclusion_file(self):
        """Test error handling for non-existent exclusion file."""
        with pytest.raises(FileNotFoundError):
            Locator(
                {
                    "exclude_samples": "/nonexistent/file.txt",
                    "sample_data": self.sample_data,
                }
            )
