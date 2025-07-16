# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:21:30 2025

@author: wassim.lelann
"""

from pyproj import CRS
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
import numpy as np
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as ET

def load_geometry(path):
    gdf = gpd.read_file(path)
    return gdf

def load_utm_grid(path,crs_field):
    gdf = load_geometry(path)
    return gdf[[crs_field,'geometry']]

class ProjectionHandler:
    def __init__(self, crs_grid=None, crs_info=None):
        """Initialize with optional inputs. Attributes can be set later."""
        self.crs_grid = crs_grid
        self.crs_info = crs_info
    
    def _validate_crs(self,crs1_input, crs2_input):
        """
        Compare two CRS definitions for equivalence.

        Parameters
        ----------
        crs1_input, crs2_input : any
            CRS representations (EPSG code, WKT string, PROJ string,
            rasterio dataset, geopandas object, etc.)

        Returns
        -------
        bool
            True if CRSes are equivalent, False otherwise.
        """

        # Convert inputs to pyproj CRS objects
        crs1 = CRS.from_user_input(crs1_input)
        crs2 = CRS.from_user_input(crs2_input)

        # Check EPSG codes first
        epsg1 = crs1.to_epsg()
        epsg2 = crs2.to_epsg()

        if epsg1 is not None and epsg2 is not None:
            return epsg1 == epsg2

        # Fallback: compare proj4 strings
        proj4_1 = crs1.to_proj4()
        proj4_2 = crs2.to_proj4()

        # Normalize proj4 strings:
        # - split into tokens
        # - sort alphabetically
        # - join back into a single string
        norm_proj4_1 = " ".join(sorted(proj4_1.strip().split()))
        norm_proj4_2 = " ".join(sorted(proj4_2.strip().split()))

        return norm_proj4_1 == norm_proj4_2
    
    def reproj_to_equalarea(self,gdf):
        return gdf.to_crs("ESRI:54009") #Mollweide Equal-Area Projection
    
    def split_by_utm(self,gdf):
        """Breakdown and split geometries into groups based on their UTM zones.
            Projects input geometries to equal-area CRS and intersects with UTM grid zones
            to determine optimal UTM zone for each geometry.
        
            Parameters
            ----------
            gdf : geopandas.GeoDataFrame
                Input geometries to be split by UTM zone
                
            Returns
            -------
            geopandas.GeoDataFrame
                Geometries intersected with UTM zone boundaries with an additional column indicating UTM zone affiliation 
            
            Raises
            ------
            AttributeError 
                If crs_grid was not provided during class initialization
        """
        if self.crs_grid is None:
            raise AttributeError("crs_grid has not been provided when initializing ProjectionHandler.")
        proj_gdf = self.reproj_to_equalarea(gdf)
        proj_crs_grid = self.reproj_to_equalarea(self.crs_grid)
        gdf_split_by_utm = gpd.overlay(proj_gdf,proj_crs_grid,how='intersection')
        return gdf_split_by_utm
    
    def generate_utm_proj(self,gdf):
        """Generator that yields geometries reprojected to their local UTM zones.
    
            Processes geometries one zone at a time for memory efficiency.
            
            Parameters
            ----------
            gdf : geopandas.GeoDataFrame
                Input geometries to be projected
                
            Yields
            ------
            geopandas.GeoDataFrame
                Geometries reprojected to their local UTM CRS
                
            Raises
            ------
            AttributeError
                If crs_grid or crs_info were not provided during class initialization
        """
        if self.crs_grid is None:
            raise AttributeError("crs_grid has not been provided when initializing ProjectionHandler.")
        if self.crs_info is None:
            raise AttributeError("crs_info has not been provided when initializing ProjectionHandler.")
        # Store 
        #if not hasattr(self, '_cached_split') or self._cached_split[0] is not gdf:
        #    self._cached_split = (gdf, self.split_by_utm(gdf))  # Store input + split result
        # Use cached split result
        #_, gdf_split_by_utm = self._cached_split
        
        gdf_split_by_utm = self.split_by_utm(gdf)
        for utm, group in gdf_split_by_utm.groupby(self.crs_info):
            yield group.to_crs(utm)
            
    def reproject_to_utm(self,gdf):
        """Returns a list of GeoDataFrames reprojected to their local UTM CRS from an original GeoDataFrame.
            Parameters
            ----------
            gdf : geopandas.GeoDataFrame
                Input geometries to be projected
                
            Returns
            -------
            list[geopandas.GeoDataFrame]
                List of geometries grouped and projected by UTM zone
        """
        return [gdf_utm for gdf_utm in self.generate_utm_proj(gdf)]
            
    def reproject_raster(self, src, target_crs, resampling_method=Resampling.nearest):
        """
        Reproject an already-open rasterio dataset to a new CRS.

        Parameters
        ----------
        src : rasterio.DatasetReader
            Open rasterio dataset.
        target_crs : str or dict
            Target CRS.
        resampling_method : rasterio.enums.Resampling
            Resampling method.

        Returns
        -------
        (np.ndarray, rasterio.Affine, dict)
            Tuple with:
                - array of reprojected data
                - affine transform
                - updated metadata (driver, dtype, nodata, etc.)
        """

        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds)
        
        # Prepare output metadata
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height})
        
        if src.nodata is not None:
            kwargs["nodata"] = src.nodata

        # Allocate memory for output array
        data = np.empty((src.count, height, width), dtype=src.dtypes[0])

        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=data[i - 1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=resampling_method,
                src_nodata=src.nodata,
                dst_nodata=src.nodata,)
        
        return data, transform, kwargs

class LandCoverAnalyzer:
    def __init__(self, gdf,projection_handler=None):
        self.gdf = gdf
        self.crs = gdf.crs
        self.projection_handler = projection_handler
        
    def _calculate_pixel_area(self, raster_res):
        return abs(raster_res[0] * raster_res[1])

    def _count_classes(self, masked_values):
        unique, counts = np.unique(masked_values, return_counts=True)
        return dict(zip(unique, counts))

    def _calculate_area(self,masked_values,raster_res):
        class_counts = self._count_classes(masked_values)
        total = sum(class_counts.values())
        pixel_area = self._calculate_pixel_area(raster_res)
        count_area = {k: v * pixel_area for k, v in class_counts.items()}
        count_pct = {k: v / total for k, v in class_counts.items()}
        return count_area,count_pct

    # def _counts_to_percent(self, class_counts):
    #     total = sum(class_counts.values())
    #     return {k: v / total * 100 for k, v in self._count_classes(masked_values).items()}

    def _build_raster_window(self, polygon, src):
        minx, miny, maxx, maxy = polygon.geometry.bounds
        if minx >= maxx or miny >= maxy:
            raise ValueError(f"Invalid Polygon Bounds")
        else:
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            return window

    def _read_raster_window(self, polygon, src):
        window = self._build_raster_window(polygon, src)
        data = src.read(1, window=window)
        if data.size == 0:
            return None

        # Mask raster values outside of input polygon
        mask = geometry_mask(
            [polygon.geometry],
            transform=src.window_transform(window),
            invert=True,
            out_shape=data.shape)
        
        masked_values = data[mask]
        
        if masked_values.size == 0:
            return None
        return masked_values
    
    def _build_output_dataframe(self,results_area,results_pct,valid_indices,missing_indices):
        # Construction du DataFrame
        df_area = pd.DataFrame(results_area, index=valid_indices)
        df_area = df_area.fillna(0)
        df_area.columns = [f"areaLandCover_{k}" for k in df_area.columns]
        df_pct = pd.DataFrame(results_pct, index=valid_indices)
        df_pct = df_pct.fillna(0)
        df_pct.columns = [f"pctLandCover_{k}" for k in df_pct.columns]
        
        if missing_indices:
            missing_df_area = pd.DataFrame(index=missing_indices, columns=df_area.columns).fillna(0)
            missing_df_pct = pd.DataFrame(index=missing_indices, columns=df_pct.columns).fillna(0)
            df_area = pd.concat([df_area, missing_df_area])
            df_pct = pd.concat([df_pct, missing_df_pct])
        return df_area.sort_index(),df_pct.sort_index()   
    
    
    def get_landcover(self, raster_path):
        """Classifie avec choix de sortie.
        
        Args:
            output (str): 
                - 'area' → superficie en m² (défaut)
                - 'percent' → % de couverture
        """
        results_area = []
        results_pct = []
        valid_indices = []
        missing_indices = []

        with rasterio.open(raster_path) as src:
            if not self.projection_handler._validate_crs(self.crs,src.crs):
                raise ValueError(f"Raster CRS ({src.crs}) differs from target CRS ")
            
            for idx, polygon in self.gdf.iterrows():
                self._build_raster_window(polygon, src)
                masked_data = self._read_raster_window(polygon, src)
                cover_area, cover_pct = self._calculate_area(masked_data,src.res)
                #class_data = self._process_window(polygon, window, src, output)
                
                if cover_area: #class_data:
                    results_area.append(cover_area)
                    results_pct.append(cover_pct)
                    valid_indices.append(idx)
                else:
                    missing_indices.append(idx)
        return self._build_output_dataframe(results_area,results_pct,valid_indices,missing_indices)
    
    
class LandCoverLegend:
    def __init__(self, landcover_df, classes_labels=None, classes_colors=None,no_data=False):
        """
        Initialize Land Cover Legend handler.
        
        Args:
            landcover_df: Output dataframe from LandCoverAnalyzer.get_landcover()
            classes_labels: Either:
                - Dictionary mapping land cover values to labels
                - Path to QML file (string ending with '.qml')
            classes_colors: Dictionary mapping values to hex colors (optional override)
            no_data: Land cover value for no data
        """
        self.landcover_df = landcover_df
        
        # Initialize empty containers
        self.landcover_labels = {}
        self.landcover_colors = {}
        self.no_data = no_data
        
        # Parse based on input type
        if isinstance(classes_labels, str) and classes_labels.endswith('.qml'):
            self._parse_qml(classes_labels)
        elif isinstance(classes_labels, dict):
            self.landcover_labels = classes_labels
        
        # Apply color overrides if provided
        if classes_colors:
            self.landcover_colors = classes_colors
    
    def _parse_qml(self,qml_path):
        """Parse QML file to extract labels and colors (your original implementation)"""
        tree = ET.parse(qml_path)
        root = tree.getroot()
        for element in root.findall(".//paletteEntry"):
            value = int(element.get('value'))
            self.landcover_labels[value] = element.get('label')
            self.landcover_colors[value] = element.get('color')
        
        # Add NODATA entry if not present
        if self.no_data != False:
            self.landcover_labels[self.no_data] = "999 - NODATA"
            self.landcover_colors[self.no_data] = "#000000"


