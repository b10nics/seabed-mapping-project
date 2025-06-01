# -*- coding: utf-8 -*-
"""
Seabed Classification Script: Terrain Analysis, Random Forest (Tuned) & K-Means
... (rest of the docstring) ...
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd

# --- Determine Project Root and Define Absolute Paths ---
# Get the directory where the script itself is located. This will serve as our project root.
# If your script is in a 'src' subfolder, you might use os.path.dirname(SCRIPT_DIR).
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR # Assuming the script is at the root of your project structure for this example.

# --- Configuration ---
# !!! USER ACTION: Modify these relative paths if your structure is different !!!
# data_dir_relative is relative to PROJECT_ROOT
data_dir_relative = 'input_data'
# output_dir_relative is relative to PROJECT_ROOT
output_dir_relative = 'Outputs_SeabedClassification'

# Construct absolute paths
ABS_DATA_DIR = os.path.join(PROJECT_ROOT, data_dir_relative)
ABS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, output_dir_relative)


# --- Set PROJ_LIB Environment Variable ---
# ... (PROJ_LIB setting code remains the same) ...
proj_lib_path = os.path.join(sys.prefix, 'share', 'proj')
if not os.path.exists(proj_lib_path):
    proj_lib_path = os.path.join(os.path.dirname(sys.executable), '..', 'share', 'proj') # Common for conda

if os.path.exists(proj_lib_path):
    os.environ['PROJ_LIB'] = proj_lib_path
    print(f"Set PROJ_LIB to: {proj_lib_path}")
else:
    alt_paths = ['/usr/share/proj', '/usr/local/share/proj', '/opt/conda/share/proj'] # Added /opt/conda/share/proj
    for path in alt_paths:
        if os.path.exists(path):
            os.environ['PROJ_LIB'] = path
            print(f"Set PROJ_LIB to alternative path: {path}")
            break
    else:
        print(f"WARNING: Default PROJ_LIB path not found ({proj_lib_path}) and no alternatives worked.")
        print("Ensure PROJ data files are installed and PROJ_LIB environment variable is set correctly, or CRS operations may fail.")


# --- Import GDAL/Rasterio AFTER setting PROJ_LIB ---
# ... (GDAL/Rasterio import code remains the same) ...
try:
    from osgeo import gdal, gdal_array, osr # osr for Coordinate Reference System handling
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.plot import show as rio_show
except ImportError as e:
    print(f"CRITICAL ERROR importing GDAL/Rasterio: {e}")
    print("Ensure GDAL and Rasterio are installed correctly in your Python environment.")
    print("For Debian/Ubuntu, 'sudo apt install python3-gdal' might be needed for system Python.")
    print("For Conda, 'conda install -c conda-forge gdal rasterio' is recommended.")
    sys.exit(1)

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter # For formatting plot axes
from matplotlib.colors import ListedColormap # For creating discrete colormaps for classification maps

gdal.UseExceptions()

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError


# --- Input Files (filenames only) ---
bathy_file_name = 'bathy_cube_10_filled_5x5.tiff'
backscatter_file_name = 'back_10_filled_5x5.tiff'
ground_truth_csv_name = 'ground_truth_samples_removed.csv'

# --- Create Output Directory (using absolute path) ---
if not os.path.exists(ABS_DATA_DIR):
    print(f"CRITICAL ERROR: The absolute data directory does not exist: {ABS_DATA_DIR}")
    print("Please create it and place your input files there, or correct the path in the script.")
    sys.exit(1)

if not os.path.exists(ABS_OUTPUT_DIR):
    os.makedirs(ABS_OUTPUT_DIR)
    print(f"Created output directory: {ABS_OUTPUT_DIR}")

# REMOVE or COMMENT OUT: os.chdir(data_dir)
# print(f"Working directory temporarily changed to: {os.getcwd()}") # This line can also be removed

# --- Output Filenames (now using ABS_OUTPUT_DIR) ---
terrain_output_files = {
    'Slope': os.path.join(ABS_OUTPUT_DIR, 'slope.tif'),
    'Aspect': os.path.join(ABS_OUTPUT_DIR, 'aspect.tif'),
    'TRI': os.path.join(ABS_OUTPUT_DIR, 'tri.tif'),
    'TPI': os.path.join(ABS_OUTPUT_DIR, 'tpi.tif'),
    'Roughness': os.path.join(ABS_OUTPUT_DIR, 'roughness.tif'),
    'Hillshade': os.path.join(ABS_OUTPUT_DIR, 'hillshade.tif')
}
backscatter_aligned_full_file = os.path.join(ABS_OUTPUT_DIR, 'backscatter_aligned_to_bathy.tif')
bathy_band1_vrt = os.path.join(ABS_OUTPUT_DIR, 'bathy_b1.vrt')
backscatter_band1_vrt = os.path.join(ABS_OUTPUT_DIR, 'backscatter_aligned_b1.vrt')
stacked_features_vrt = os.path.join(ABS_OUTPUT_DIR, 'stacked_features_for_classification.vrt')
stacked_features_tif = os.path.join(ABS_OUTPUT_DIR, 'stacked_features_for_classification.tif')

rf_classified_file = os.path.join(ABS_OUTPUT_DIR, 'classification_rf.tif')
kmeans_classified_file = os.path.join(ABS_OUTPUT_DIR, 'classification_kmeans.tif')

# --- Parameters ---
INITIAL_CLASSIFICATION_FEATURES = ['Depth', 'Backscatter', 'Slope', 'Aspect', 'TRI', 'TPI', 'Roughness']
n_kmeans_clusters = 7
rf_cv_folds = 5
rf_n_jobs = -1
test_size_rf = 0.3
random_state = 42
cleanup_intermediate = False

# ... (Helper functions save_raster_rio, plot_raster, remove_feature remain the same) ...
def save_raster_rio(filename, data_array, profile, nodata_value=None):
    """Saves a numpy array as a GeoTIFF using Rasterio."""
    print(f"Saving raster using Rasterio: {filename}")
    # Update the profile with specifics for this output array
    profile_out = profile.copy() # Work on a copy to avoid modifying the original profile dict
    profile_out.update(
        dtype=data_array.dtype,
        count=1, # Outputting single band rasters from this function
        nodata=nodata_value
    )
    try:
        with rasterio.open(filename, 'w', **profile_out) as dst:
            dst.write(data_array.astype(profile_out['dtype']), 1) # Ensure dtype consistency
        print(f"Successfully saved: {filename}")
    except Exception as e:
        print(f"ERROR saving raster {filename} using Rasterio: {e}")


def plot_raster(raster_path, title, cmap='viridis', label='Value', ax=None, figsize=(8, 8), vmin=None, vmax=None, **kwargs):
    """
    Plots a raster file using Rasterio and Matplotlib (simplified version).
    (rest of the function code)
    """
    try:
        with rasterio.open(raster_path) as src:
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
                show_plot_in_function = True
            else:
                fig = ax.figure
                show_plot_in_function = False

            try: # Attempt to set numeric formatter for axes
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            except AttributeError: pass # Non-critical if it fails

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            data_array = src.read(1, masked=True) # Read as masked array to handle NoData

            if vmin is not None and vmax is not None and vmin == vmax:
                print(f"Warning for '{title}': vmin equals vmax ({vmin}). Adjusting slightly.")
                vmax = vmin + 1e-6 # Avoid single-color map issue

            # Plot the raster using rasterio.plot.show
            img_artist = rio_show(src, ax=ax, cmap=cmap, title=title, vmin=vmin, vmax=vmax, **kwargs)

            # Add colorbar intelligently
            if np.ma.is_masked(data_array) and data_array.mask.all():
                print(f"Skipping colorbar for '{title}' (all data is NoData/masked).")
            elif not np.ma.is_masked(data_array) and data_array.min() == data_array.max():
                print(f"Skipping colorbar for '{title}' (all valid data has the same value: {data_array.min()}).")
            else:
                # For newer rasterio, rio_show might return the AxesImage directly.
                # If it returns a Show類 object, you might need img_artist.ax.images[0] or similar
                # Assuming img_artist is compatible or is the AxesImage itself for colorbar.
                if hasattr(img_artist, 'figure'): # Heuristic: if it's an Axes object from older rio_show
                    fig.colorbar(img_artist.images[0] if img_artist.images else img_artist, ax=ax, label=label)
                else: # Assuming img_artist is the image itself for newer versions
                    fig.colorbar(img_artist, ax=ax, label=label)


            ax.set_xlabel("Easting"); ax.set_ylabel("Northing")
            ax.ticklabel_format(style='plain', axis='both', useOffset=False)

            if show_plot_in_function:
                plt.tight_layout(); plt.show()
            return fig, ax

    except rasterio.RasterioIOError:
        print(f"ERROR: Could not open raster file '{raster_path}' for plotting.")
        return None, None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during plotting of '{raster_path}': {e}")
        return None, None

def remove_feature(feature_name, feature_list):
    """Removes a feature name from a list if present and prints a message."""
    if feature_name in feature_list:
        feature_list.remove(feature_name)
        print(f"-> Feature '{feature_name}' removed from processing list due to error or unavailability.")
    return feature_list

# --- Main Script ---
if __name__ == "__main__":
    script_start_time = time.time()
    print(f"--- Starting Seabed Classification Script ---")
    print(f"Using Project Root: {PROJECT_ROOT}")
    print(f"Using Data Directory: {ABS_DATA_DIR}")
    print(f"Using Output Directory: {ABS_OUTPUT_DIR}")

    # --- 1. Check Inputs ---
    print("\n--- 1. Checking Input Files ---")
    # Construct full paths to input files
    bathy_path = os.path.join(ABS_DATA_DIR, bathy_file_name)
    backscatter_path_orig_abs = os.path.join(ABS_DATA_DIR, backscatter_file_name)
    gt_csv_path = os.path.join(ABS_DATA_DIR, ground_truth_csv_name)

    current_classification_features = INITIAL_CLASSIFICATION_FEATURES[:]

    if not os.path.exists(bathy_path):
        sys.exit(f"CRITICAL ERROR: Bathymetry file not found: {bathy_path}")

    backscatter_path = None # Initialize to None
    if 'Backscatter' in current_classification_features:
        if not os.path.exists(backscatter_path_orig_abs):
            print(f"WARNING: Backscatter file not found: {backscatter_path_orig_abs}.")
            current_classification_features = remove_feature('Backscatter', current_classification_features)
        elif os.path.getsize(backscatter_path_orig_abs) == 0:
            print(f"WARNING: Backscatter file found but is empty: {backscatter_path_orig_abs}.")
            current_classification_features = remove_feature('Backscatter', current_classification_features)
        else:
            backscatter_path = backscatter_path_orig_abs # Use the absolute path
            print(f"Found backscatter file: {backscatter_path}")
    else:
        print("Backscatter not in INITIAL_CLASSIFICATION_FEATURES.")

    if not os.path.exists(gt_csv_path):
        sys.exit(f"CRITICAL ERROR: Ground truth CSV not found: {gt_csv_path}")

    print("Input files check complete.")
    print(f"Attempting classification with features: {current_classification_features}")

    # --- 2. Define Reference Grid from Bathymetry & Get NoData ---
    print("\n--- 2. Defining Reference Grid from Bathymetry ---")
    ref_profile = None; bathy_nodata = None; ref_gt_affine = None; ref_proj_wkt = None;
    ref_cols = None; ref_rows = None; ref_bounds = None
    try:
        with rasterio.open(bathy_path) as src: # bathy_path is relative to data_dir (CWD)
            ref_profile = src.profile   # Rasterio profile (metadata dictionary)
            bathy_nodata = src.nodata   # NoData value from bathymetry
            ref_gt_affine = src.transform # Affine transform (georeferencing)
            ref_proj_wkt = src.crs.to_wkt() if src.crs else None # CRS in WKT format
            if not ref_proj_wkt: print("WARNING: Reference bathymetry has no CRS defined in its metadata.")
            ref_cols, ref_rows = src.width, src.height
            ref_bounds = src.bounds     # Bounding box (left, bottom, right, top)

            print(f"Reference Grid Source: {bathy_path}")
            print(f"  Dimensions: {ref_cols}x{ref_rows}, Resolution: ({ref_profile['transform'].a:.2f}, {ref_profile['transform'].e:.2f})")
            print(f"  NoData Value: {bathy_nodata}")
            print(f"  Bounds (L,B,R,T): {ref_bounds}")
            print(f"  Projection (WKT snippet): {str(ref_proj_wkt)[:100]}...")
            if not ref_gt_affine: print("WARNING: Reference bathymetry has no GeoTransform (Affine) defined.")
    except Exception as e:
        sys.exit(f"CRITICAL ERROR reading reference grid info from {bathy_path}: {e}")
    if ref_profile is None: sys.exit("FATAL: Could not obtain reference profile from bathymetry.")

    # --- 3. Perform Terrain Analysis using GDAL ---
    print("\n--- 3. Performing Terrain Analysis ---")
    terrain_feature_files_generated = {} # Stores paths of successfully generated terrain rasters
    # These GDAL DEMProcessing modes require the input dataset path (bathy_path)
    # and implicitly handle georeferencing based on that input.
    potential_terrain_features_to_calc = ['Slope', 'Aspect', 'TRI', 'TPI', 'Roughness', 'Hillshade']
    for feature_name in potential_terrain_features_to_calc:
        is_needed_for_classification = feature_name in current_classification_features
        is_hillshade_for_viz = feature_name == 'Hillshade' # Hillshade is often just for visualization

        if not (is_needed_for_classification or is_hillshade_for_viz):
             continue # Skip if not in current feature list and not the optional Hillshade

        output_file_path = terrain_output_files.get(feature_name) # Get path from config dict
        if not output_file_path:
            print(f"ERROR: Output path for {feature_name} not defined. Skipping.")
            if is_needed_for_classification: remove_feature(feature_name, current_classification_features)
            continue

        # Determine GDAL processing mode and specific options
        processing_mode = feature_name.lower()
        gdal_dem_options_dict = {'computeEdges': True} # Common option for many DEM modes
        if feature_name == 'Slope': gdal_dem_options_dict['alg'] = 'ZevenbergenThorne' # Specific algorithm for slope
        elif feature_name == 'Aspect': gdal_dem_options_dict['zeroForFlat'] = True # Handle flat areas for aspect
        elif feature_name == 'Hillshade': gdal_dem_options_dict['zFactor'] = 2 # Vertical exaggeration for hillshade

        try:
            print(f"Calculating {feature_name} -> {output_file_path}...")
            gdal_options_obj = gdal.DEMProcessingOptions(**gdal_dem_options_dict)
            # bathy_path is relative to data_dir (CWD)
            ds_out = gdal.DEMProcessing(output_file_path, bathy_path, processing_mode, options=gdal_options_obj)
            if ds_out is None:
                 print(f"WARNING: gdal.DEMProcessing for {feature_name} returned None. File might not be created or valid.")
                 if is_needed_for_classification: remove_feature(feature_name, current_classification_features)
                 continue
            ds_out = None # Close/dereference the output dataset to ensure file writing is finalized

            if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
                terrain_feature_files_generated[feature_name] = output_file_path
                print(f"  Saved: {output_file_path}")
            else:
                print(f"WARNING: {feature_name} output file not found or empty after GDAL processing: {output_file_path}")
                if is_needed_for_classification: remove_feature(feature_name, current_classification_features)
        except Exception as e:
            print(f"ERROR during {feature_name} analysis: {e}. Skipping.")
            if is_needed_for_classification: remove_feature(feature_name, current_classification_features)
    print("Terrain analysis stage complete.")

    # --- 4. Align Backscatter to Reference Grid using GDAL Warp ---
    print("\n--- 4. Aligning Backscatter (if provided and valid) ---")
    aligned_backscatter_full_path = None # Path to the multi-band aligned backscatter TIF
    # Check if 'Backscatter' is still a requested feature AND its input path (backscatter_path) is valid
    if 'Backscatter' in current_classification_features and backscatter_path:
        print(f"Aligning '{backscatter_path}' to bathymetry grid ({ref_cols}x{ref_rows})...")
        if ref_proj_wkt is None or ref_gt_affine is None or ref_bounds is None:
            print("WARNING: Cannot align backscatter; reference bathymetry lacks essential georeferencing (CRS, Transform, or Bounds).")
            remove_feature('Backscatter', current_classification_features)
        else:
            try:
                # Use bathymetry's NoData value for the target, or a common default if bathy_nodata is None
                target_nodata_for_alignment = bathy_nodata if bathy_nodata is not None else -9999.0

                # GDAL Warp options to align backscatter to bathymetry's grid definition
                warp_options = gdal.WarpOptions(
                    format='GTiff',
                    width=ref_cols,           # Force output width to match reference
                    height=ref_rows,          # Force output height to match reference
                    outputBounds=(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top), # Match extent
                    dstSRS=ref_proj_wkt,      # Target CRS from reference bathymetry
                    resampleAlg='bilinear',   # Bilinear resampling for continuous data
                    dstNodata=target_nodata_for_alignment, # Set NoData for output
                    creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'] # Optimize output GeoTIFF
                )
                # backscatter_path is relative to data_dir (CWD)
                # backscatter_aligned_full_file is the output path (includes output_dir)
                ds_warp = gdal.Warp(backscatter_aligned_full_file, backscatter_path, options=warp_options)
                if ds_warp is None:
                    print(f"WARNING: GDAL Warp for backscatter returned None. File might not be created or valid.")
                    remove_feature('Backscatter', current_classification_features)
                else:
                     ds_warp = None # Close/finalize dataset
                     if os.path.exists(backscatter_aligned_full_file) and os.path.getsize(backscatter_aligned_full_file) > 0:
                        aligned_backscatter_full_path = backscatter_aligned_full_file
                        print(f"  Aligned backscatter saved: {aligned_backscatter_full_path}")
                     else:
                        print(f"WARNING: GDAL Warp output file for backscatter not found or empty: {backscatter_aligned_full_file}")
                        remove_feature('Backscatter', current_classification_features)
            except Exception as e:
                print(f"ERROR during backscatter alignment: {e}.")
                remove_feature('Backscatter', current_classification_features)
    elif 'Backscatter' in current_classification_features:
        # This handles if 'Backscatter' was in the list, but its input file was missing/empty from Step 1.
        print("Skipping backscatter alignment (input was unavailable or removed earlier).")
        remove_feature('Backscatter', current_classification_features) # Ensure it's removed
    else:
        print("Backscatter processing not requested or already removed from feature list.")


    # --- 5. Prepare Final List of Features for Stacking ---
    # This step creates VRTs for single bands of bathy/backscatter if needed,
    # and collects paths to all GeoTIFFs that will form the final multi-band raster.
    print("\n--- 5. Preparing Final Feature List for Stacking ---")
    feature_filepaths_for_stacking = [] # List of file paths (TIFs or VRTs) to stack
    band_names_for_final_stack = []     # Corresponding names for bands in the final stack

    # 5a. Add Bathymetry (Band 1)
    if 'Depth' in current_classification_features:
        try:
            print(f"Creating VRT for Bathymetry (Band 1): {bathy_band1_vrt}")
            # bathy_path is relative to data_dir (CWD)
            bathy_vrt_build_options = gdal.BuildVRTOptions(bandList=[1]) # Select only band 1
            vrt_ds = gdal.BuildVRT(bathy_band1_vrt, bathy_path, options=bathy_vrt_build_options)
            if vrt_ds:
                vrt_ds = None # Close VRT
                if os.path.exists(bathy_band1_vrt):
                    feature_filepaths_for_stacking.append(bathy_band1_vrt)
                    band_names_for_final_stack.append('Depth')
                    print(f"  -> Added Depth (from VRT): {bathy_band1_vrt}")
                else: raise IOError(f"Bathymetry VRT file not found after creation: {bathy_band1_vrt}")
            else: raise RuntimeError("gdal.BuildVRT for Bathymetry returned None.")
        except Exception as e:
            print(f"ERROR creating Bathymetry VRT: {e}.")
            remove_feature('Depth', current_classification_features)

    # 5b. Add Aligned Backscatter (Band 1 from the aligned TIF)
    if 'Backscatter' in current_classification_features and aligned_backscatter_full_path:
        try:
            print(f"Creating VRT for Aligned Backscatter (Band 1): {backscatter_band1_vrt}")
            bs_vrt_build_options = gdal.BuildVRTOptions(bandList=[1]) # Select only band 1
            vrt_bs_ds = gdal.BuildVRT(backscatter_band1_vrt, aligned_backscatter_full_path, options=bs_vrt_build_options)
            if vrt_bs_ds:
                 vrt_bs_ds = None # Close VRT
                 if os.path.exists(backscatter_band1_vrt):
                     feature_filepaths_for_stacking.append(backscatter_band1_vrt)
                     band_names_for_final_stack.append('Backscatter')
                     print(f"  -> Added Backscatter (from VRT): {backscatter_band1_vrt}")
                 else: raise IOError(f"Backscatter VRT file not found after creation: {backscatter_band1_vrt}")
            else: raise RuntimeError("gdal.BuildVRT for Backscatter returned None.")
        except Exception as e:
            print(f"ERROR creating Backscatter VRT: {e}.")
            remove_feature('Backscatter', current_classification_features)

    # 5c. Add successfully generated Terrain Derivatives (direct TIF paths)
    terrain_features_to_try_stack = ['Slope', 'Aspect', 'TRI', 'TPI', 'Roughness']
    for feature_name in terrain_features_to_try_stack:
        if feature_name in current_classification_features:
            feature_path = terrain_feature_files_generated.get(feature_name) # Get from dict of successful files
            if feature_path and os.path.exists(feature_path):
                 feature_filepaths_for_stacking.append(feature_path)
                 band_names_for_final_stack.append(feature_name)
                 print(f"  -> Added Terrain Feature: {feature_name} from {feature_path}")
            else:
                 print(f"WARNING: Skipping {feature_name} for stack (file missing or not generated successfully in Step 3).")
                 remove_feature(feature_name, current_classification_features)

    # --- 5d. Final Check and Report on Features for Stacking ---
    print(f"\nFinal features prepared for stacking ({len(feature_filepaths_for_stacking)} bands): {band_names_for_final_stack}")
    if not feature_filepaths_for_stacking:
        sys.exit("CRITICAL ERROR: No features were successfully prepared for stacking. Cannot proceed.")

    # Explicit check message for Backscatter inclusion status
    was_bs_requested = 'Backscatter' in INITIAL_CLASSIFICATION_FEATURES
    is_bs_included = 'Backscatter' in band_names_for_final_stack
    if was_bs_requested and not is_bs_included:
        print("\nIMPORTANT WARNING: 'Backscatter' was requested but is NOT included in the final stack.")
        print("  This could be due to: input file missing/empty, alignment failure, or VRT creation failure.")
        print("  Classification will proceed without Backscatter.")
    elif is_bs_included:
        print(">>> Backscatter IS INCLUDED in the final feature stack.")
    elif not was_bs_requested:
         print(">>> Backscatter was NOT REQUESTED and is not included.")


    # --- 6. Stack Prepared Features into Final Multi-band TIF using GDAL VRT and Translate ---
    print("\n--- 6. Stacking Final Features into a Multi-Band GeoTIFF ---")
    final_stack_profile = None # To store profile of the final stacked TIF for later use
    try:
        print(f"Building final VRT from {len(feature_filepaths_for_stacking)} sources: {stacked_features_vrt}")
        # Use separate=True to ensure each input file becomes a separate band in the VRT.
        vrt_stack_options = gdal.BuildVRTOptions(separate=True)
        vrt_ds_stack = gdal.BuildVRT(stacked_features_vrt, feature_filepaths_for_stacking, options=vrt_stack_options)
        if vrt_ds_stack is None: sys.exit("CRITICAL ERROR: Failed to build final VRT (gdal.BuildVRT returned None).")

        expected_band_count = len(feature_filepaths_for_stacking)
        if vrt_ds_stack.RasterCount != expected_band_count:
             actual_bands = vrt_ds_stack.RasterCount; vrt_ds_stack = None # Close before exit
             sys.exit(f"CRITICAL ERROR: Final VRT band count mismatch ({actual_bands} vs {expected_band_count}). Check VRT creation.")
        vrt_ds_stack = None # Close VRT dataset handle, it's written to disk

        print(f"Translating final VRT to TIF: {stacked_features_tif}")
        # Determine NoData value for the final stack (use bathymetry's if available, else a common default)
        final_stack_nodata_val = bathy_nodata if bathy_nodata is not None else -9999.0
        translate_options = gdal.TranslateOptions(
            format='GTiff',
            noData=final_stack_nodata_val,
            creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES', 'INTERLEAVE=BAND'] # Band interleave for efficiency
        )
        stack_ds_tif = gdal.Translate(stacked_features_tif, stacked_features_vrt, options=translate_options)
        if stack_ds_tif is None: sys.exit("CRITICAL ERROR: Failed to translate final VRT to TIF (gdal.Translate returned None).")

        # Verify band count in the final TIF
        if stack_ds_tif.RasterCount != len(band_names_for_final_stack):
             actual_bands_tif = stack_ds_tif.RasterCount; stack_ds_tif = None
             sys.exit(f"CRITICAL ERROR: Final TIF band count mismatch ({actual_bands_tif} vs {len(band_names_for_final_stack)}).")

        # Set band descriptions (names) in the final TIF for better interpretability in GIS software
        print(f"Setting band descriptions for TIF: {', '.join(band_names_for_final_stack)}")
        for i, band_name_to_set in enumerate(band_names_for_final_stack):
            try:
                band = stack_ds_tif.GetRasterBand(i + 1) # GDAL bands are 1-indexed
                band.SetDescription(band_name_to_set)
                band.FlushCache() # Ensure description is written
            except Exception as e_band_desc: print(f"WARNING: Failed to set band {i+1} ('{band_name_to_set}') description: {e_band_desc}")
        print(f"Final stacked TIF created: {stacked_features_tif}")
        stack_ds_tif = None # Close the final TIF dataset

        # Read the profile of the successfully created stacked_features_tif using Rasterio
        # This profile will be used as a template for saving classification output rasters.
        with rasterio.open(stacked_features_tif) as src_stack_final:
            final_stack_profile = src_stack_final.profile
            print(f"Successfully read profile from final stack TIF: {stacked_features_tif}")

    except Exception as e_stack: sys.exit(f"CRITICAL ERROR during final feature stacking: {e_stack}")
    if final_stack_profile is None:
         sys.exit("FATAL: Could not obtain profile from the final stacked features TIF after creation. Cannot save classifications.")


    # --- 7. Load Ground Truth (CSV) and Project Coordinates ---
    print("\n--- 7. Loading and Projecting Ground Truth Data ---")
    # gt_csv_path is relative to data_dir (CWD)
    gdf_ground_truth = None
    try:
        gt_df_initial = pd.read_csv(gt_csv_path, encoding='latin-1') # latin-1 can handle some special characters
        print(f"Loaded {len(gt_df_initial)} ground truth points from {gt_csv_path}.")
        required_gt_cols = ['Longitude', 'Latitude', 'Class']
        if not all(col in gt_df_initial.columns for col in required_gt_cols):
            sys.exit(f"CRITICAL ERROR: Ground truth CSV must contain columns: {required_gt_cols}")

        # Create GeoDataFrame, assuming input coordinates are WGS84 (EPSG:4326)
        gdf_ground_truth = gpd.GeoDataFrame(
            gt_df_initial,
            geometry=gpd.points_from_xy(gt_df_initial.Longitude, gt_df_initial.Latitude),
            crs='EPSG:4326' # Standard for lat/lon
        )

        # Target CRS for reprojection is taken from the final stacked raster's profile
        target_crs_from_stack = final_stack_profile.get('crs')
        if target_crs_from_stack is None:
             print("WARNING: Final stack raster profile has no CRS information. Ground truth cannot be reprojected accurately.")
             print("         Ensure ground truth coordinates ALREADY match the raster projection: {ref_proj_wkt or unknown}")
        else:
             raster_crs_obj = rasterio.crs.CRS.from_wkt(target_crs_from_stack.to_wkt()) \
                 if isinstance(target_crs_from_stack, rasterio.crs.CRS) else \
                 rasterio.crs.CRS.from_string(str(target_crs_from_stack))

             print(f"Ground Truth Input CRS: {gdf_ground_truth.crs.to_string()}")
             print(f"Target Raster CRS for Reprojection: {raster_crs_obj.to_string()}")

             if gdf_ground_truth.crs != raster_crs_obj:
                print(f"Projecting ground truth points to target CRS...")
                gdf_ground_truth = gdf_ground_truth.to_crs(raster_crs_obj)
                print("  Projection complete.")
             else:
                print("Ground truth CRS already matches raster CRS. No reprojection needed.")

        # Extract projected coordinates (Easting, Northing) for sampling
        gdf_ground_truth['Easting'] = gdf_ground_truth.geometry.x
        gdf_ground_truth['Northing'] = gdf_ground_truth.geometry.y
        print("Ground truth data (head) after potential reprojection:")
        print(gdf_ground_truth.head())

    except FileNotFoundError: sys.exit(f"CRITICAL ERROR: Ground truth file not found at {os.path.join(data_dir, gt_csv_path)}")
    except Exception as e_gt: sys.exit(f"CRITICAL ERROR loading or processing ground truth data: {e_gt}")


    # --- 8. Extract Training Data by Sampling Stacked Raster at Ground Truth Locations ---
    print("\n--- 8. Extracting Training Data from Stacked Raster ---")
    # Coordinates for sampling, taken from the (potentially reprojected) GeoDataFrame
    coords_for_sampling = [(x, y) for x, y in zip(gdf_ground_truth.Easting, gdf_ground_truth.Northing)]
    class_to_int_map = {}; int_to_class_map = {} # For mapping string class labels to integers
    X_features = None; y_target_int = None; training_data_df = None
    actual_band_names_from_stack_for_sampling = band_names_for_final_stack # Use the definitive list

    try:
        with rasterio.open(stacked_features_tif) as src_for_sampling:
             if len(actual_band_names_from_stack_for_sampling) != src_for_sampling.count:
                 sys.exit(f"FATAL ERROR: Mismatch between expected features for sampling ({len(actual_band_names_from_stack_for_sampling)}) "
                          f"and raster band count ({src_for_sampling.count}) in {stacked_features_tif}")

             print(f"Sampling {src_for_sampling.count} raster bands at {len(coords_for_sampling)} locations: {', '.join(actual_band_names_from_stack_for_sampling)}")
             # rasterio.sample returns a generator; convert to list of numpy arrays, then to a 2D array
             sampled_values_generator = src_for_sampling.sample(coords_for_sampling)
             sampled_values_array = np.vstack(list(sampled_values_generator))
             # Create DataFrame with columns named after the actual bands in the stack
             sampled_features_df = pd.DataFrame(sampled_values_array, columns=actual_band_names_from_stack_for_sampling)

        # Combine sampled features with original ground truth info (coordinates, class label)
        # Reset index on both DataFrames before concat to ensure correct alignment
        training_data_df = pd.concat([
            gdf_ground_truth[['Easting', 'Northing', 'Class']].reset_index(drop=True),
            sampled_features_df.reset_index(drop=True)
        ], axis=1)

        print(f"Training data shape before NoData/NaN removal: {training_data_df.shape}")
        # Use the NoData value from the final stack's profile for robust checking
        nodata_val_from_stack_profile = final_stack_profile.get('nodata')
        cols_to_check_for_nodata = actual_band_names_from_stack_for_sampling
        initial_row_count = len(training_data_df)

        # Create masks for NoData and NaN values across all feature columns
        nodata_mask = pd.DataFrame(False, index=training_data_df.index, columns=cols_to_check_for_nodata)
        if nodata_val_from_stack_profile is not None:
             # Handle cases where NoData might be NaN itself or needs careful float comparison
             if np.issubdtype(training_data_df[cols_to_check_for_nodata].values.dtype, np.floating) and np.isnan(nodata_val_from_stack_profile):
                  # isnull() check below will catch this
                  pass
             else:
                  nodata_mask = training_data_df[cols_to_check_for_nodata] == nodata_val_from_stack_profile
        nan_mask = training_data_df[cols_to_check_for_nodata].isnull()
        # Combine masks: a row is dropped if ANY feature in it is NoData OR NaN
        rows_to_drop_mask = (nodata_mask | nan_mask).any(axis=1)

        training_data_df = training_data_df[~rows_to_drop_mask].copy() # Keep only valid rows
        num_dropped_rows = initial_row_count - len(training_data_df)
        print(f"Removed {num_dropped_rows} rows containing NoData ({nodata_val_from_stack_profile}) or NaN values in feature columns.")

        print(f"Final training data shape: {training_data_df.shape}")
        if training_data_df.empty: sys.exit("CRITICAL ERROR: No valid training data remaining after NoData/NaN removal. Cannot train models.")
        print("Sample of final training data (first 5 rows):\n", training_data_df.head())

        # Map string class labels to integers for scikit-learn model compatibility
        # Sort unique classes to ensure consistent mapping across runs
        unique_class_labels_sorted = sorted(training_data_df['Class'].astype(str).unique())
        class_to_int_map = {label: i for i, label in enumerate(unique_class_labels_sorted)}
        int_to_class_map = {i: label for label, i in class_to_int_map.items()}
        print("\nClass Label Mapping (String -> Integer):\n", class_to_int_map)

        y_target_int = training_data_df['Class'].astype(str).map(class_to_int_map) # Integer target variable
        X_features = training_data_df[actual_band_names_from_stack_for_sampling]  # Feature matrix

        print("\nClass Counts in Final Training Data (Integer Representation):\n", y_target_int.value_counts().sort_index())

    except rasterio.RasterioIOError: sys.exit(f"CRITICAL ERROR opening stacked raster {stacked_features_tif} for sampling.")
    except IndexError as e_idx: sys.exit(f"CRITICAL ERROR during sampling (IndexError), possibly coordinates outside raster bounds? Details: {e_idx}")
    except Exception as e_sample: sys.exit(f"CRITICAL ERROR: An unexpected error occurred during training data extraction: {e_sample}")


    # --- 9. Supervised Classification: Random Forest ---
    print("\n--- 9. Random Forest Classification ---")
    rf_successful = False
    # These variables will hold data needed by K-Means if RF runs successfully
    data_for_full_raster_prediction = None
    valid_pixel_mask_for_full_raster = None
    # Profile for saving classification maps (derived from final_stack_profile)
    output_classification_profile = None
    full_raster_shape = None

    if X_features is None or y_target_int is None: # Should have exited earlier if this is the case
        sys.exit("CRITICAL ERROR: Training features (X) or target (y) not available for Random Forest.")
    try:
        # Split data into training and testing sets, stratifying by integer class labels
        X_train, X_test, y_train_int, y_test_int = train_test_split(
            X_features, y_target_int, test_size=test_size_rf, random_state=random_state, stratify=y_target_int
        )
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        # Adjust CV folds if the smallest class count in the training set is less than requested folds
        # This is crucial for stratified cross-validation to work.
        min_class_count_in_train = y_train_int.value_counts().min()
        cv_folds_for_gridsearch = rf_cv_folds
        if min_class_count_in_train > 0 and min_class_count_in_train < cv_folds_for_gridsearch:
            print(f"WARNING: Smallest class in training set ({min_class_count_in_train}) is less than CV folds ({cv_folds_for_gridsearch}).")
            cv_folds_for_gridsearch = min_class_count_in_train
            print(f"         Adjusting CV folds for GridSearchCV to {cv_folds_for_gridsearch}.")
        elif min_class_count_in_train <= 1: # Stratified CV needs at least 2 samples per class for any split
            sys.exit(f"CRITICAL ERROR: Smallest class count in training set ({min_class_count_in_train}) is <= 1. Cannot perform stratified CV.")

        # Define parameter grid for GridSearchCV (hyperparameter tuning)
        # These ranges are common starting points for Random Forest.
        param_grid_rf = {
            'n_estimators': [50, 100, 200, 300],  # Number of trees
            'max_depth': [None, 10, 20],         # Max depth of trees (None = full growth)
            'min_samples_split': [2, 5],         # Min samples to split an internal node
            'min_samples_leaf': [1, 3],          # Min samples at a leaf node
            'class_weight': ['balanced']         # Adjusts weights for imbalanced classes
        }
        rf_classifier = RandomForestClassifier(random_state=random_state)
        print(f"Performing GridSearchCV for Random Forest (CV={cv_folds_for_gridsearch}, using {rf_n_jobs} jobs)...")
        # GridSearchCV will find the best combination of parameters from param_grid_rf
        grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf,
                                      cv=cv_folds_for_gridsearch, n_jobs=rf_n_jobs, verbose=1, scoring='accuracy')
        grid_search_rf.fit(X_train, y_train_int) # Train using integer labels

        print(f"\nGridSearchCV Best Parameters Found: {grid_search_rf.best_params_}")
        best_rf_model = grid_search_rf.best_estimator_ # The optimally tuned Random Forest model

        # Evaluate the best model on the held-out test set
        print("\nEvaluating best RF model on the test set:")
        y_pred_rf_test_int = best_rf_model.predict(X_test) # Predict integer labels for test set

        accuracy_rf_test = accuracy_score(y_test_int, y_pred_rf_test_int) * 100
        kappa_rf_test = cohen_kappa_score(y_test_int, y_pred_rf_test_int)
        cm_rf_test = confusion_matrix(y_test_int, y_pred_rf_test_int)
        # Use original string class names (via int_to_class_map) for a more readable classification report
        report_target_names_str = [int_to_class_map[i] for i in sorted(y_target_int.unique())]
        classification_rep_rf = classification_report(y_test_int, y_pred_rf_test_int,
                                                      target_names=report_target_names_str, zero_division=0)

        print(f"  Overall Accuracy on Test Set: {accuracy_rf_test:.2f}%")
        print(f"  Kappa Coefficient on Test Set: {kappa_rf_test:.3f}")
        print("  Confusion Matrix (Rows: True Int Label, Cols: Pred Int Label):\n", cm_rf_test)
        print("\n  Classification Report (using original class names):\n", classification_rep_rf)

        # Feature Importance from the best (tuned) Random Forest model
        importances_rf = best_rf_model.feature_importances_ * 100 # As percentages
        # Ensure feature names match the columns used for training (X_features.columns)
        feature_importance_df = pd.DataFrame({
            'Feature': X_features.columns, # Use columns from the training DataFrame X_features
            'Importance (%)': importances_rf
        }).sort_values(by='Importance (%)', ascending=False)
        print("\nFeature Importances from Tuned RF Model:\n", feature_importance_df)
        try: # Plot feature importances
            fig_imp, ax_imp = plt.subplots(figsize=(10, max(6, len(X_features.columns) * 0.5))) # Adjust height for many features
            feature_importance_df.plot(kind='barh', x='Feature', y='Importance (%)', ax=ax_imp, legend=False,
                                       color='skyblue', edgecolor='black')
            ax_imp.invert_yaxis() # Highest importance at the top
            ax_imp.set_title('Random Forest Feature Importance')
            ax_imp.set_xlabel('Importance (%)')
            ax_imp.set_ylabel('Feature')
            plt.tight_layout(); plt.show()
        except Exception as plot_e: print(f"WARNING: Failed to display feature importance plot: {plot_e}")


        # Predict classes for the ENTIRE raster using the best trained RF model
        print("\nPredicting classes for the entire raster using the best RF model...")
        with rasterio.open(stacked_features_tif) as src_predict:
            # Read all bands into a 3D numpy array (bands, rows, cols)
            full_raster_data_array = src_predict.read()
            # Get profile from the source stack TIF; this will be the basis for output classification rasters
            output_classification_profile = src_predict.profile.copy() # Ensure it's a copy
            full_raster_shape = (src_predict.height, src_predict.width) # (rows, cols)
            nodata_val_for_prediction = src_predict.nodata if src_predict.nodata is not None else final_stack_nodata_val

            # Create a robust mask for NoData/NaN across ALL bands of the full raster
            valid_pixel_mask_for_full_raster = np.ones(full_raster_shape, dtype=bool) # Assume all valid initially
            for i in range(src_predict.count): # Iterate through each band
                band_data_i = full_raster_data_array[i]
                current_band_mask = np.ones(full_raster_shape, dtype=bool)
                if nodata_val_for_prediction is not None:
                    if np.issubdtype(band_data_i.dtype, np.floating) and np.isnan(nodata_val_for_prediction):
                        current_band_mask = ~np.isnan(band_data_i) # Valid if not NaN
                    else: # Valid if not equal to NoData AND not NaN
                        current_band_mask = (band_data_i != nodata_val_for_prediction) & (~np.isnan(band_data_i))
                else: # No specific NoData defined, only check for NaN
                    current_band_mask = ~np.isnan(band_data_i)
                # Update overall mask: a pixel is valid only if valid in ALL bands processed so far
                valid_pixel_mask_for_full_raster &= current_band_mask

            num_valid_pixels_for_pred = valid_pixel_mask_for_full_raster.sum()
            num_features_in_raster = src_predict.count
            print(f"Number of valid pixels in full raster for prediction: {num_valid_pixels_for_pred}")
            if num_features_in_raster != len(actual_band_names_from_stack_for_sampling):
                 sys.exit(f"FATAL ERROR: Prediction raster band count ({num_features_in_raster}) doesn't match "
                          f"expected features ({len(actual_band_names_from_stack_for_sampling)}).")

            # Prepare data for scikit-learn: shape (n_samples, n_features)
            # Extract data ONLY for valid pixels and transpose to get features as columns
            data_for_full_raster_prediction = np.vstack(
                [full_raster_data_array[i][valid_pixel_mask_for_full_raster] for i in range(num_features_in_raster)]
            ).T

            if data_for_full_raster_prediction.size == 0:
                sys.exit("CRITICAL ERROR: No valid pixels found in the raster for RF prediction (all masked).")

        # Perform prediction using the tuned best_rf_model
        print(f"Predicting on {data_for_full_raster_prediction.shape[0]} valid pixels with {data_for_full_raster_prediction.shape[1]} features...")
        predicted_labels_rf_full_int = best_rf_model.predict(data_for_full_raster_prediction) # Predicts integer labels

        # Create the output classification map array
        # Initialize with a NoData value. Choose dtype suitable for integer labels + NoData (e.g., int16).
        classification_map_rf = np.full(full_raster_shape, -99, dtype=np.int16) # Use a NoData marker
        # Fill in the predicted integer labels ONLY for the valid pixels
        classification_map_rf[valid_pixel_mask_for_full_raster] = predicted_labels_rf_full_int

        print(f"Saving RF classification map (integer labels) to: {rf_classified_file}")
        # Update the profile for single-band integer output
        rf_output_profile = output_classification_profile.copy()
        rf_output_profile.update(dtype=rasterio.int16, count=1, nodata=-99) # Set output dtype, band count, nodata
        save_raster_rio(rf_classified_file, classification_map_rf, rf_output_profile, nodata_value=-99)

        # Plot RF classification map
        num_rf_classes = len(class_to_int_map) # Number of unique classes from ground truth
        # Use a discrete colormap (e.g., tab10, tab20) with enough colors for the classes
        cmap_rf_plot = ListedColormap(plt.cm.get_cmap('tab10', num_rf_classes).colors[:num_rf_classes])
        plot_raster(rf_classified_file, "Random Forest Classification Map", cmap=cmap_rf_plot, label='Class ID (Integer)')
        rf_successful = True
    except NotFittedError: sys.exit("CRITICAL ERROR: RandomForest model/GridSearchCV was not fitted before prediction.")
    except ValueError as e_rf_val: sys.exit(f"CRITICAL ERROR during RF training or prediction (ValueError): {e_rf_val}. Check data.")
    except MemoryError: sys.exit("CRITICAL ERROR: Memory Error during RF prediction. Input raster might be too large or too many features.")
    except Exception as e_rf_other: sys.exit(f"CRITICAL ERROR: An unexpected error occurred during Random Forest stage: {e_rf_other}")


    # --- 10. Unsupervised Classification: K-Means ---
    print("\n--- 10. K-Means Clustering ---")
    kmeans_successful = False
    if rf_successful: # Proceed with K-Means only if RF part (which prepares data) was successful
        try:
            # Ensure data_for_full_raster_prediction and valid_pixel_mask_for_full_raster are available from RF step
            if data_for_full_raster_prediction is None or valid_pixel_mask_for_full_raster is None \
               or output_classification_profile is None or full_raster_shape is None:
                 print("WARNING: Required data from RF step not available for K-Means. Skipping K-Means.")
                 raise RuntimeError("Missing prerequisite data for K-Means from RF stage.")

            # Scaling is CRUCIAL for distance-based algorithms like K-Means.
            print("Scaling data for K-Means clustering...")
            scaler_kmeans = StandardScaler()
            # Work on a copy of the data to avoid modifying data_for_full_raster_prediction if it's used elsewhere
            data_for_kmeans_scaled = data_for_full_raster_prediction.copy()

            # Check for constant features (zero variance) before scaling, as StandardScaler warns/errors.
            # These features don't contribute to clustering and can be ignored by scaling or removed.
            feature_stds = np.std(data_for_kmeans_scaled, axis=0)
            constant_feature_indices = np.where(feature_stds == 0)[0]
            non_constant_feature_indices = np.where(feature_stds != 0)[0]

            if constant_feature_indices.size > 0:
                constant_feat_names = [actual_band_names_from_stack_for_sampling[i] for i in constant_feature_indices]
                print(f"WARNING: Constant features detected for K-Means: {constant_feat_names}. These will not be scaled.")
                if non_constant_feature_indices.size > 0: # Scale only non-constant features
                    print(f"  Scaling {non_constant_feature_indices.size} non-constant features...")
                    data_for_kmeans_scaled[:, non_constant_feature_indices] = scaler_kmeans.fit_transform(
                        data_for_kmeans_scaled[:, non_constant_feature_indices])
                else: print("  All features are constant. K-Means might produce trivial clustering or errors.")
            elif non_constant_feature_indices.size > 0: # All features are non-constant and can be scaled
                 print(f"  Scaling all {non_constant_feature_indices.size} features for K-Means...")
                 data_for_kmeans_scaled = scaler_kmeans.fit_transform(data_for_kmeans_scaled)
            else: # Should not happen if data_for_full_raster_prediction has data
                 print("WARNING: No data available or no non-constant features for K-Means scaling.")
                 raise RuntimeError("No valid non-constant data for K-Means.")

            # Initialize and fit K-Means model
            print(f"Fitting K-Means model (k={n_kmeans_clusters}, n_init='auto' or 10)...")
            # n_init='auto' (in newer scikit-learn) or n_init=10 runs K-Means multiple times with different centroids
            # and selects the one with the lowest Sum of Squared Errors (SSE), improving robustness.
            kmeans_model = KMeans(n_clusters=n_kmeans_clusters, random_state=random_state, n_init='auto' if 'auto' in KMeans().get_params() else 10)
            kmeans_model.fit(data_for_kmeans_scaled)

            # Get cluster labels for the valid pixels (labels are 0-based integers)
            kmeans_predicted_labels_0based = kmeans_model.labels_
            # Convert K-Means 0-based labels to 1-based for map representation (optional, but common)
            kmeans_predicted_labels_1based = kmeans_predicted_labels_0based + 1

            # Create the K-Means classification map array
            classification_map_kmeans = np.full(full_raster_shape, -99, dtype=np.int16) # Use same NoData as RF map
            classification_map_kmeans[valid_pixel_mask_for_full_raster] = kmeans_predicted_labels_1based

            print(f"Saving K-Means classification map (1-based cluster IDs) to: {kmeans_classified_file}")
            # Use the same output profile structure as the RF map for consistency
            kmeans_output_profile = output_classification_profile.copy()
            kmeans_output_profile.update(dtype=rasterio.int16, count=1, nodata=-99)
            save_raster_rio(kmeans_classified_file, classification_map_kmeans, kmeans_output_profile, nodata_value=-99)

            # Plot K-Means classification map
            cmap_kmeans_plot = ListedColormap(plt.cm.get_cmap('tab10', n_kmeans_clusters).colors[:n_kmeans_clusters])
            plot_raster(kmeans_classified_file, f"K-Means Clustering (k={n_kmeans_clusters})",
                        cmap=cmap_kmeans_plot, label='Cluster ID (1-based)')
            kmeans_successful = True
        except RuntimeError as e_km_runtime: # Catch explicit runtime errors raised within this block
            print(f"Skipping K-Means due to a runtime issue: {e_km_runtime}")
        except MemoryError: print(f"WARNING: Memory Error during K-Means processing. Skipping K-Means.")
        except Exception as e_km_other: print(f"WARNING: An unexpected error occurred during K-Means: {e_km_other}. Skipping K-Means.")
    elif not rf_successful:
        print("\nSkipping K-Means clustering because the Random Forest stage (which prepares data for prediction) did not complete successfully.")


    # --- 11. Cleanup Intermediate Files ---
    # This list includes VRTs created for single bands and the full multi-band aligned backscatter TIF (if created and if cleanup is True).
    # The individual terrain GeoTIFFs and the final stacked_features_tif are generally kept as key outputs.
    files_for_cleanup = [
        bathy_band1_vrt,
        backscatter_band1_vrt,      # Only relevant if backscatter was processed
        stacked_features_vrt,       # VRT for the full stack
        backscatter_aligned_full_file # The full aligned backscatter TIF (not the VRT of its band 1)
    ]
    if cleanup_intermediate:
        print("\n--- 11. Cleaning Up Intermediate Files ---")
        for file_to_remove in files_for_cleanup:
            # Check if the path variable is not None (e.g., backscatter_aligned_full_file might be None)
            # and if the file actually exists on disk.
            if file_to_remove and os.path.exists(file_to_remove):
                try:
                    os.remove(file_to_remove)
                    print(f"  Removed: {file_to_remove}")
                except OSError as e_cleanup:
                    print(f"  Error removing {file_to_remove}: {e_cleanup}")
            # else:
            #     print(f"  Skipping removal (file not created, path invalid, or already removed): {file_to_remove}")
    else:
        print("\n--- 11. Skipping Cleanup of Intermediate Files ---")
        print("Intermediate files (VRTs, full aligned backscatter TIF if created) were kept as per 'cleanup_intermediate = False'.")
        print("Key outputs like terrain GeoTIFFs, final stacked TIF, and classification maps are always kept.")

    # --- End Script ---
    script_end_time = time.time()
    print(f"\n--- Script Finished in {script_end_time - script_start_time:.2f} seconds ---")