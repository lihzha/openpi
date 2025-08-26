#!/usr/bin/env python3
"""
Hash File Comparison Tool

This script compares two hash files to check if they contain the same hashes,
regardless of the order they appear in the files.
"""

from pathlib import Path
import sys


def read_hash_file(file_path: str) -> set[str]:
    """
    Read a hash file and return a set of unique hashes.

    Args:
        file_path: Path to the hash file

    Returns:
        Set of hash strings

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or contains invalid data
    """
    try:
        with open(file_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    if not lines:
        raise ValueError(f"File is empty: {file_path}")

    # Extract hashes, strip whitespace, and filter out empty lines
    hashes = set()
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if line:  # Skip empty lines
            # Basic validation: check if line looks like a hash (40 hex chars)
            if len(line) == 40 and all(c in "0123456789abcdef" for c in line.lower()):
                hashes.add(line)
            else:
                print(f"Warning: Line {line_num} in {file_path} doesn't look like a valid hash: {line}")

    if not hashes:
        raise ValueError(f"No valid hashes found in {file_path}")

    return hashes


def compare_hash_files(file1: str, file2: str) -> tuple[bool, dict]:
    """
    Compare two hash files and return comparison results.

    Args:
        file1: Path to first hash file
        file2: Path to second hash file

    Returns:
        Tuple of (are_equal, comparison_details)
    """
    try:
        hashes1 = read_hash_file(file1)
        hashes2 = read_hash_file(file2)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error reading files: {e}")
        return False, {}

    # Check if the sets are equal (same hashes regardless of order)
    are_equal = hashes1 == hashes2

    # Calculate differences for detailed reporting
    only_in_file1 = hashes1 - hashes2
    only_in_file2 = hashes2 - hashes1
    common_hashes = hashes1 & hashes2

    comparison_details = {
        "file1_count": len(hashes1),
        "file2_count": len(hashes2),
        "common_count": len(common_hashes),
        "only_in_file1": only_in_file1,
        "only_in_file2": only_in_file2,
        "common_hashes": common_hashes,
    }

    return are_equal, comparison_details


def print_comparison_results(file1: str, file2: str, are_equal: bool, details: dict):
    """Print formatted comparison results."""
    print("=" * 60)
    print("HASH FILE COMPARISON RESULTS")
    print("=" * 60)
    print(f"File 1: {file1}")
    print(f"File 2: {file2}")
    print(f"Files contain same hashes: {'YES' if are_equal else 'NO'}")
    print()

    print("SUMMARY:")
    print(f"  Hashes in {file1}: {details['file1_count']}")
    print(f"  Hashes in {file2}: {details['file2_count']}")
    print(f"  Common hashes: {details['common_count']}")
    print()

    if not are_equal:
        print("DIFFERENCES:")
        if details["only_in_file1"]:
            print(f"  Hashes only in {file1}: {len(details['only_in_file1'])}")
            for hash_val in sorted(details["only_in_file1"]):
                print(f"    {hash_val}")
            print()

        if details["only_in_file2"]:
            print(f"  Hashes only in {file2}: {len(details['only_in_file2'])}")
            for hash_val in sorted(details["only_in_file2"]):
                print(f"    {hash_val}")
            print()

    print("=" * 60)


def main():
    """Main function to run the hash comparison."""

    file1 = "hash_test3.txt"
    file2 = "hash_train3.txt"
    # file2 = "visualizations/hash31.txt"

    # Check if files exist
    if not Path(file1).exists():
        print(f"Error: File '{file1}' does not exist")
        sys.exit(1)

    if not Path(file2).exists():
        print(f"Error: File '{file2}' does not exist")
        sys.exit(1)

    # Compare the files
    are_equal, details = compare_hash_files(file1, file2)

    # Print results
    print_comparison_results(file1, file2, are_equal, details)

    # Exit with appropriate code
    sys.exit(0 if are_equal else 1)


if __name__ == "__main__":
    main()
