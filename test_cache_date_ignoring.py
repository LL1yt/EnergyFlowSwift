#!/usr/bin/env python3
"""
Test script to verify that cache date ignoring works correctly
"""

import os
import tempfile
import shutil
import glob
from pathlib import Path

def test_pattern_matching():
    """Test the pattern matching logic for cache files"""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "cache"
        cache_dir.mkdir()
        
        # Test parameters
        lattice_str = "15x15x15"
        adaptive_radius_str = "6.00"
        adaptive_radius_ratio = "unknown"
        hash_key = "0c72488a"
        
        # Create test cache files with different dates
        test_files = [
            f"connection_cache_20250724_{lattice_str}_r{adaptive_radius_str}_ar{adaptive_radius_ratio}_{hash_key}.pkl",
            f"connection_cache_20250723_{lattice_str}_r{adaptive_radius_str}_ar{adaptive_radius_ratio}_{hash_key}.pkl",
            f"connection_cache_20250722_{lattice_str}_r{adaptive_radius_str}_ar{adaptive_radius_ratio}_{hash_key}.pkl",
            f"connection_cache_20250721_{lattice_str}_r{adaptive_radius_str}_ar{adaptive_radius_ratio}_{hash_key}.pkl",
            # This should NOT match (different hash)
            f"connection_cache_20250724_{lattice_str}_r{adaptive_radius_str}_ar{adaptive_radius_ratio}_different.pkl",
            # This should NOT match (different dimensions)
            f"connection_cache_20250724_16x16x16_r{adaptive_radius_str}_ar{adaptive_radius_ratio}_{hash_key}.pkl",
        ]
        
        # Create the files
        for filename in test_files:
            (cache_dir / filename).touch()
        
        # Test the pattern matching
        pattern = f"connection_cache_*_{lattice_str}_r{adaptive_radius_str}_ar{adaptive_radius_ratio}_{hash_key}.pkl"
        matching_files = glob.glob(str(cache_dir / pattern))
        
        print("Pattern:", pattern)
        print("Matching files:")
        for f in matching_files:
            print(f"  {Path(f).name}")
        
        # Verify we found the expected files
        expected_matches = [
            f"connection_cache_20250724_{lattice_str}_r{adaptive_radius_str}_ar{adaptive_radius_ratio}_{hash_key}.pkl",
            f"connection_cache_20250723_{lattice_str}_r{adaptive_radius_str}_ar{adaptive_radius_ratio}_{hash_key}.pkl",
            f"connection_cache_20250722_{lattice_str}_r{adaptive_radius_str}_ar{adaptive_radius_ratio}_{hash_key}.pkl",
            f"connection_cache_20250721_{lattice_str}_r{adaptive_radius_str}_ar{adaptive_radius_ratio}_{hash_key}.pkl",
        ]
        
        found_basenames = [Path(f).name for f in matching_files]
        
        # Check if all expected files are found
        all_expected_found = all(expected in found_basenames for expected in expected_matches)
        no_unexpected_found = not any("different" in f or "16x16x16" in f for f in found_basenames)
        
        print(f"\nAll expected files found: {all_expected_found}")
        print(f"No unexpected files found: {no_unexpected_found}")
        print(f"Total matches: {len(matching_files)}")
        
        return all_expected_found and no_unexpected_found and len(matching_files) == 4

if __name__ == "__main__":
    success = test_pattern_matching()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")