# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 14:58:42 2025

@author: wassim.lelann
"""

from landcoverstats.utils import *
from landcoverstats.geospatial_analysis import *
from pathlib import Path

# Load & Parse config
config_path = r"C:\Users\wassim.lelann\Documents\My GitHub\landcoverstats\config\config.yaml"
cfg = load_config(config_path)
# Input geometry
input_geometry = load_geometry(Path(cfg["geometry_input"]["file_path"]))
# Landcover raster file path
raster = cfg['raster_input']['file_path']
# UTM Zone
utm_crs = cfg['crs_handling']['world_utm_grid']['crs_field']
world_utm = load_utm_grid(cfg['crs_handling']['world_utm_grid']['file_path'],utm_crs)

# Initialize classes & testing
input_geometry = input_geometry.to_crs("EPSG:3035")
input_test = input_geometry[input_geometry['NAME'] == "Germany"]
projection_handler = ProjectionHandler(world_utm,utm_crs)
LandCoverAnalyzer = LandCoverAnalyzer(input_test,projection_handler)

# Calculate area
cover_area,cover_pct = LandCoverAnalyzer.get_landcover(raster)
