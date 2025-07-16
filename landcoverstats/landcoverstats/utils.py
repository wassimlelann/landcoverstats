# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:53:09 2025

@author: wassim.lelann
"""

from pyproj import CRS
import yaml

def load_config(config_path, validate=True):
    """
    Load and validate YAML config file.
    
    Args:
        config_path: Path to YAML config file
        validate: If True (default), runs config validation
    
    Returns:
        dict: Parsed configuration
        
    Raises:
        ValueError: If validation fails
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if validate:
        _validate_config(config)  # Private validation function
    return config



def _validate_config(config):
    """Internal validation logic."""
    # 1. Top-level sections
    required_sections = ["geometry_input", "crs_handling", "raster_input"]
    if missing := [s for s in required_sections if s not in config]:
        raise ValueError(f"Missing required sections: {missing}")

    # 2. Geometry validation
    if not isinstance(config["geometry_input"].get("buffer_radius_meters"), (int, float)):
        raise ValueError("buffer_radius_meters must be a number")

    # 3. Raster validation
    if not isinstance(config["raster_input"]["year"], int):
        raise ValueError("Year must be an integer")

    # 4. CRS validation
    crs = config["crs_handling"]
    
    # Check all CRS flags are proper booleans
    crs_flags = ["use_shapefile_crs", "use_raster_crs", "use_custom_crs", "use_local_utm"]
    if invalid := [f for f in crs_flags if not isinstance(crs[f], bool)]:
        raise ValueError(f"CRS flags must be booleans: {invalid}")

    # Exactly one CRS selected
    if sum(crs[f] for f in crs_flags) != 1:
        raise ValueError("Exactly one CRS option must be True")

    # Validate custom CRS if used
    if crs["use_custom_crs"]:
        try:
            # Newer pyproj validation (v3.0+)
            crs_obj = CRS(crs["custom_crs"])
            if crs_obj.to_authority() is None:
                raise ValueError(f"CRS is not recognized by PROJ: {crs['custom_crs']}")
        except Exception as e:
            raise ValueError(f"CRS validation failed: {str(e)}")