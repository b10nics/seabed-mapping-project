# -*- coding: utf-8 -*-
"""
Seabed Classification Script: Terrain Analysis, Random Forest (Tuned) & K-Means

Aligns backscatter to bathymetry grid (if file exists), calculates terrain features,
keeps generated GeoTIFFs, and performs classification using available features:
Depth (Band 1), Backscatter (Band 1), Slope, Aspect, TRI, TPI, Roughness.
Generates separate classification maps for Random Forest and K-Means.

*** NOTE: For Backscatter to be included, the input file specified by
    `backscatter_file` MUST exist in the `data_dir` directory. ***
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd

# --- Set PROJ_LIB Environment Variable ---
# Attempt to find PROJ data directory relative to the Python executable
# This is often necessary for GDAL/Rasterio to find their CRS database.
proj_lib_path = os.path.join(sys.prefix, 'share', 'proj')
if not os.path.exists(proj_lib_path):
    # Fallback for conda environments
    proj_lib_path = os.path.join(os.path.dirname(sys.executable), '..', 'share', 'proj')

if os.path.exists(proj_lib_path):
    os.environ['PROJ_LIB'] = proj_lib_path
    print(f"Set PROJ_LIB to: {proj_lib_path}")
else:
    # Check common alternative paths if needed (adjust based on system)
    alt_paths = ['/usr/share/proj', '/usr/local/share/proj']
    for path in alt_paths:
        if os.path.exists(path):
            os.environ['PROJ_LIB'] = path
            print(f"Set PROJ_LIB to alternative path: {path}")
            break
    else:
        print(f"Warning: Default PROJ_LIB path not found ({proj_lib_path}) and no alternatives worked. PROJ errors may occur.")
        print("Ensure PROJ data files are installed and PROJ_LIB environment variable is set correctly.")


# --- Import GDAL/Rasterio AFTER setting PROJ_LIB ---
try:
    from osgeo import gdal, gdal_array, osr # Import osr for CRS handling
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.plot import show as rio_show
except ImportError as e:
    print(f"Error importing GDAL/Rasterio: {e}")
    print("Ensure GDAL and Rasterio are installed correctly.")
    sys.exit(1)

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap # For discrete colormaps
# Removed BoundaryNorm and Patch as they were specific to the comparison map legend

# --- Enable GDAL Exceptions ---
gdal.UseExceptions() # Make GDAL raise Python exceptions on errors

# --- Scikit-learn ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# --- Configuration ---
# !!! ENSURE THIS DIRECTORY IS CORRECT !!!
data_dir = r'/home/jose/Desktop/' # Use raw string for paths
if not os.path.exists(data_dir): os.makedirs(data_dir)
os.chdir(data_dir) # Change working directory
print(f"Running locally. Working directory set to: {data_dir}")

# --- Input Files ---
# !!! ENSURE THESE FILES EXIST IN data_dir !!!
bathy_file = 'bathy_cube_10_filled_5x5.tiff'         # <<< Reference grid source
backscatter_file = 'back_10_filled_5x5.tiff'         # <<< File to be aligned and included
ground_truth_csv = 'ground_truth_samples_removed.csv' # <<< Ground truth points

# --- Output Directory ---
output_dir = 'Outputs_SeabedClassification_NoCompare_With_BS_v1' # Updated folder name
if not os.path.exists(output_dir): os.makedirs(output_dir)

# --- Output Filenames ---
# Intermediate Terrain Files (kept by default)
terrain_output_files = {
    'Slope': os.path.join(output_dir, 'slope.tif'),
    'Aspect': os.path.join(output_dir, 'aspect.tif'),
    'TRI': os.path.join(output_dir, 'tri.tif'),
    'TPI': os.path.join(output_dir, 'tpi.tif'),
    'Roughness': os.path.join(output_dir, 'roughness.tif'),
    'Hillshade': os.path.join(output_dir, 'hillshade.tif') # Optional visualization aid
}
# Intermediate Processing Files (can be cleaned up)
backscatter_aligned_full_file = os.path.join(output_dir, 'backscatter_aligned_to_bathy.tif')
backscatter_band1_vrt = os.path.join(output_dir, 'backscatter_aligned_b1.vrt')
bathy_band1_vrt = os.path.join(output_dir, 'bathy_b1.vrt')
stacked_features_vrt = os.path.join(output_dir, 'stacked_features_for_classification.vrt')
stacked_features_tif = os.path.join(output_dir, 'stacked_features_for_classification.tif') # Final input for models
# Final Classification Outputs
rf_classified_file = os.path.join(output_dir, 'classification_rf.tif')
kmeans_classified_file = os.path.join(output_dir, 'classification_kmeans.tif')
# Removed comparison_map_file

# --- Parameters ---
n_kmeans_clusters = 6 # Number of K-Means clusters - CHECK IF THIS MATCHES # of UNIQUE RF CLASSES
rf_cv_folds = 5
rf_n_jobs = -1 # Use all available CPU cores for RF tuning (-1) or set to 1 if issues arise
test_size_rf = 0.3 # Proportion of ground truth data for testing RF
random_state = 42 # For reproducibility
cleanup_intermediate = False # Keep intermediate files like VRTs, aligned BS by default

# --- Feature List ---
# Define the features intended for use at the start
INITIAL_CLASSIFICATION_FEATURES = ['Depth', 'Backscatter', 'Slope', 'Aspect', 'TRI', 'TPI', 'Roughness']

# --- Helper Functions ---
def save_raster_rio(filename, data_array, profile, nodata_value=None):
    """Saves a numpy array as a GeoTIFF using Rasterio."""
    print(f"Saving raster using Rasterio: {filename}")
    # Update the profile with specifics for this output array
    profile.update(
        dtype=data_array.dtype,
        count=1, # Outputting single band rasters
        nodata=nodata_value
    )
    try:
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(data_array, 1)
        print(f"Successfully saved: {filename}")
    except Exception as e:
        print(f"ERROR saving raster {filename} using Rasterio: {e}")


def plot_raster(raster_path, title, cmap='viridis', label='Value', ax=None, figsize=(8, 8), vmin=None, vmax=None, **kwargs):
    """Enhanced plotting function with error handling."""
    try:
        with rasterio.open(raster_path) as src:
            if ax is None: fig, ax = plt.subplots(figsize=figsize); show_plot = True
            else: fig = ax.figure; show_plot = False

            formatter = FormatStrFormatter('%.0f');
            try: # Avoid errors if axis has no major formatter
                ax.xaxis.set_major_formatter(formatter); ax.yaxis.set_major_formatter(formatter)
            except AttributeError: pass
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Handle potential single-color map issue if vmin=vmax
            if vmin is not None and vmax is not None and vmin == vmax:
                print(f"Warning plotting {raster_path}: vmin equals vmax ({vmin}). Adjusting slightly for display.")
                vmax = vmin + 1e-6 # Add small epsilon

            img = rio_show(src, ax=ax, cmap=cmap, title=title, vmin=vmin, vmax=vmax, **kwargs);
            im = img.get_images()[0]
            # Add colorbar only if image data range is not zero
            data = src.read(1)
            nodata = src.nodata
            valid_data_mask = np.ones(data.shape, dtype=bool)
            if nodata is not None: valid_data_mask = data != nodata
            # Explicitly handle NaN nodata if rasterio doesn't automatically
            if nodata is not None and np.isnan(nodata):
                 valid_data_mask = ~np.isnan(data)
            elif nodata is None: # If NoData is not set, still check for NaNs
                 valid_data_mask = ~np.isnan(data)

            # Check if there's valid data and it has a range > 0
            if valid_data_mask.any() and data[valid_data_mask].ptp() > 0:
                fig.colorbar(im, ax=ax, label=label);
            else:
                 print(f"Skipping colorbar for {raster_path} (valid data range is zero or empty).")

            ax.set_xlabel("Easting"); ax.set_ylabel("Northing")
            ax.ticklabel_format(style='plain', axis='both', useOffset=False) # Prevent scientific notation

            if show_plot: plt.tight_layout(); plt.show()
            return fig, ax
    except rasterio.RasterioIOError: print(f"Error: Could not open raster file {raster_path}"); return None, None
    except ValueError as ve: # Catch specific errors like single-color map
        print(f"Warning during plotting {raster_path}: {ve}. Trying without vmin/vmax.")
        try: # Retry without vmin/vmax
            with rasterio.open(raster_path) as src_retry:
                if ax is None: fig, ax = plt.subplots(figsize=figsize); show_plot = True
                else: fig = ax.figure; show_plot = False
                img = rio_show(src_retry, ax=ax, cmap=cmap, title=title, **kwargs)
                im = img.get_images()[0]
                # Add colorbar logic again for retry
                data_retry = src_retry.read(1)
                nodata_retry = src_retry.nodata
                valid_data_mask_retry = np.ones(data_retry.shape, dtype=bool)
                if nodata_retry is not None: valid_data_mask_retry = data_retry != nodata_retry
                if nodata_retry is not None and np.isnan(nodata_retry): valid_data_mask_retry = ~np.isnan(data_retry)
                elif nodata_retry is None: valid_data_mask_retry = ~np.isnan(data_retry)

                if valid_data_mask_retry.any() and data_retry[valid_data_mask_retry].ptp() > 0:
                    fig.colorbar(im, ax=ax, label=label);
                else:
                    print(f"Skipping colorbar on retry for {raster_path} (valid data range is zero or empty).")

                ax.set_xlabel("Easting"); ax.set_ylabel("Northing")
                ax.ticklabel_format(style='plain', axis='both', useOffset=False)
                if show_plot: plt.tight_layout(); plt.show()
                return fig, ax
        except Exception as e_retry:
             print(f"Retry plotting failed for {raster_path}: {e_retry}")
             return None, None
    except Exception as e: print(f"An error occurred during plotting {raster_path}: {e}"); return None, None

# Helper function to remove feature if it fails
def remove_feature(fname, feature_list):
    """Removes a feature name from a list if present and prints a message."""
    if fname in feature_list:
        feature_list.remove(fname)
        print(f"-> Feature '{fname}' removed from processing list.")
    return feature_list # Return modified list

# --- Main Script ---
if __name__ == "__main__":
    start_time = time.time()
    print(f"--- Starting Seabed Classification Script ---")

    # --- 1. Check Inputs ---
    print("\n--- 1. Checking Input Files ---")
    bathy_path = os.path.join(data_dir, bathy_file)
    backscatter_path = os.path.join(data_dir, backscatter_file)
    gt_csv_path = os.path.join(data_dir, ground_truth_csv)

    # Make a mutable copy of the initial features list to modify during processing
    current_classification_features = INITIAL_CLASSIFICATION_FEATURES[:]

    if not os.path.exists(bathy_path): sys.exit(f"Error: Bathymetry file not found: {bathy_path}")
    # --- Backscatter Check ---
    if not os.path.exists(backscatter_path):
        print(f"Warning: Backscatter file not found: {backscatter_path}. Removing 'Backscatter' from feature list.")
        backscatter_path = None # Set path to None if file missing
        current_classification_features = remove_feature('Backscatter', current_classification_features)
    else:
        print(f"Found backscatter file: {backscatter_path}") # Confirm if found
        # Optionally, add a check for file size > 0
        if os.path.getsize(backscatter_path) == 0:
            print(f"Warning: Backscatter file found but is empty: {backscatter_path}. Removing 'Backscatter'.")
            backscatter_path = None
            current_classification_features = remove_feature('Backscatter', current_classification_features)
    # --- End Backscatter Check ---
    if not os.path.exists(gt_csv_path): sys.exit(f"Error: Ground truth CSV not found: {gt_csv_path}")

    print("Input files check complete.")
    print(f"Attempting classification with features: {current_classification_features}") # Will show if BS was removed dynamically

    # --- 2. Define Reference Grid from Bathymetry & Get NoData ---
    print("\n--- 2. Defining Reference Grid ---")
    # Initialize reference variables
    ref_profile = None; bathy_nodata = None; ref_gt_affine = None; ref_proj_wkt = None; ref_cols = None; ref_rows = None; ref_bounds = None
    try:
        # Use Rasterio to get the essential metadata profile
        with rasterio.open(bathy_path) as src:
            ref_profile = src.profile # Store rasterio profile dictionary
            bathy_nodata = src.nodata # Get NoData value
            ref_gt_affine = src.transform # Get Affine transform object
            ref_proj_wkt = src.crs.to_wkt() if src.crs else None # Get CRS WKT
            if not ref_proj_wkt: print("Warning: Reference bathymetry has no CRS defined in its metadata.")
            ref_cols = src.width
            ref_rows = src.height
            ref_bounds = src.bounds # Get BoundingBox

            print(f"Reference NoData: {bathy_nodata}")
            print(f"Reference Grid: {ref_cols}x{ref_rows}, Res=({ref_profile['transform'].a:.2f},{ref_profile['transform'].e:.2f})")
            print(f"Reference Bounds (L,B,R,T): {ref_bounds}")
            print(f"Reference Projection (WKT): {str(ref_proj_wkt)[:80]}...") # Use str() for safety

            # Check if essential info is present
            if not ref_gt_affine: print("Warning: Reference bathymetry has no GeoTransform defined.")

    except Exception as e: sys.exit(f"Error reading reference grid info from {bathy_path}: {e}")
    # Ensure we have a profile to proceed
    if ref_profile is None: sys.exit("FATAL: Could not obtain reference profile from bathymetry.")

    # --- 3. Perform Terrain Analysis ---
    print("\n--- 3. Performing Terrain Analysis ---")
    terrain_feature_files_generated = {} # Store paths of successfully generated files
    # GDAL DEMProcessing requires the input dataset path
    # It implicitly handles georeferencing based on the input dataset
    potential_terrain_features = ['Slope', 'Aspect', 'TRI', 'TPI', 'Roughness', 'Hillshade']
    for feature_name in potential_terrain_features:
        # Check if feature is needed for classification OR if it's the optional Hillshade
        is_needed_for_classification = feature_name in current_classification_features
        is_hillshade = feature_name == 'Hillshade'
        if not (is_needed_for_classification or is_hillshade):
             continue # Skip if not in the current feature list and not Hillshade

        # Use the predefined dictionary for output paths
        output_file = terrain_output_files.get(feature_name)
        if not output_file:
            print(f"Error: Output path for {feature_name} not defined in `terrain_output_files` dict. Skipping.")
            if is_needed_for_classification: remove_feature(feature_name, current_classification_features)
            continue

        processing_mode = feature_name.lower() if feature_name != 'Hillshade' else 'hillshade'
        options_dict = {'computeEdges': True}
        if feature_name == 'Slope': options_dict['alg'] = 'ZevenbergenThorne'
        elif feature_name == 'Aspect': options_dict['zeroForFlat'] = True
        elif feature_name == 'Hillshade': options_dict['zFactor'] = 2

        try:
            print(f"Calculating {feature_name}...")
            gdal_options = gdal.DEMProcessingOptions(**options_dict)
            ds_out = gdal.DEMProcessing(output_file, bathy_path, processing_mode, options=gdal_options) # Use path directly
            if ds_out is None:
                 print(f"Warning: gdal.DEMProcessing for {feature_name} returned None.")
                 if is_needed_for_classification: remove_feature(feature_name, current_classification_features)
                 continue
            ds_out = None # Close the output dataset explicitly by dereferencing

            if os.path.exists(output_file):
                terrain_feature_files_generated[feature_name] = output_file # Store path if successful
                print(f"Saved: {output_file}")
            else:
                print(f"Warning: {feature_name} output file not found after GDAL processing: {output_file}")
                if is_needed_for_classification: remove_feature(feature_name, current_classification_features)
        except Exception as e:
            print(f"Error during {feature_name} analysis: {e}. Skipping.")
            if is_needed_for_classification: remove_feature(feature_name, current_classification_features)

    print("Terrain analysis stage complete.")


    # --- 4. Align Backscatter to Reference Grid ---
    print("\n--- 4. Aligning Backscatter ---")
    aligned_backscatter_full_path = None
    # Check if backscatter was requested AND the file path is still valid (not set to None earlier)
    if 'Backscatter' in current_classification_features and backscatter_path:
        print(f"Aligning '{backscatter_file}' to bathymetry grid ({ref_cols}x{ref_rows})...")
        # Use reference CRS WKT and transform (Affine) obtained in Step 2
        if ref_proj_wkt is None or ref_gt_affine is None or ref_bounds is None:
            print("Warning: Cannot align backscatter because reference bathymetry lacks CRS, Transform or Bounds. Skipping Backscatter.")
            remove_feature('Backscatter', current_classification_features)
        else:
            try:
                # Use bathy NoData if available, otherwise a common default
                target_nodata = bathy_nodata if bathy_nodata is not None else -9999.0

                warp_options = gdal.WarpOptions(
                    format='GTiff',
                    width=ref_cols,           # Force output width
                    height=ref_rows,          # Force output height
                    outputBounds=(ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top), # Use bounds from rasterio
                    dstSRS=ref_proj_wkt,     # Target CRS WKT
                    resampleAlg='bilinear',
                    dstNodata=target_nodata,
                    creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES']
                )
                ds_warp = gdal.Warp(backscatter_aligned_full_file, backscatter_path, options=warp_options)
                if ds_warp is None:
                    print(f"Warning: GDAL Warp returned None for backscatter.")
                    remove_feature('Backscatter', current_classification_features)
                else:
                     ds_warp = None # Close output dataset
                     if os.path.exists(backscatter_aligned_full_file):
                        aligned_backscatter_full_path = backscatter_aligned_full_file # Store path if successful
                        print(f"Aligned backscatter saved: {aligned_backscatter_full_path}")
                     else:
                        print(f"Warning: GDAL Warp output file not found after processing: {backscatter_aligned_full_file}")
                        remove_feature('Backscatter', current_classification_features)

            except Exception as e:
                print(f"Error during backscatter alignment: {e}. Skipping.")
                remove_feature('Backscatter', current_classification_features)

    elif 'Backscatter' in current_classification_features:
        # This case handles if 'Backscatter' was in the list but backscatter_path was None (file missing/empty)
        print("Skipping backscatter alignment (input unavailable or removed).")
        # Feature might have already been removed in Step 1, but do it again just in case
        remove_feature('Backscatter', current_classification_features)
    else:
        print("Backscatter not requested in INITIAL_CLASSIFICATION_FEATURES or already removed.")


    # --- 5. Prepare Final List of Features for Stacking ---
    print("\n--- 5. Preparing Final Feature List ---")
    feature_list_for_stacking = [] # List of file paths or VRTs to stack
    band_names_for_stacking = [] # Corresponding names for bands in the stack

    # 5a. Add Bathymetry (using VRT for band 1)
    if 'Depth' in current_classification_features:
        try:
            print(f"Creating VRT for Bathy B1: {bathy_band1_vrt}")
            bathy_vrt_options = gdal.BuildVRTOptions(bandList=[1]) # Specify band 1
            vrt_ds = gdal.BuildVRT(bathy_band1_vrt, bathy_path, options=bathy_vrt_options)
            if vrt_ds is not None:
                vrt_ds = None # Close VRT dataset handle
                if os.path.exists(bathy_band1_vrt):
                    feature_list_for_stacking.append(bathy_band1_vrt)
                    band_names_for_stacking.append('Depth')
                    print(f" -> Added Depth from: {bathy_band1_vrt}")
                else:
                    print(f"Error: Bathy VRT file not found after creation: {bathy_band1_vrt}")
                    remove_feature('Depth', current_classification_features)
            else:
                print(f"Error: Failed VRT creation for Bathy (gdal.BuildVRT returned None).")
                remove_feature('Depth', current_classification_features)
        except Exception as e:
            print(f"Error creating Bathy VRT: {e}.")
            remove_feature('Depth', current_classification_features)

    # 5b. Add Aligned Backscatter (using VRT for band 1)
    # Check if still in the list AND the aligned file was successfully created
    if 'Backscatter' in current_classification_features and aligned_backscatter_full_path:
        try:
            print(f"Creating VRT for Aligned Backscatter B1: {backscatter_band1_vrt}")
            bs_vrt_options = gdal.BuildVRTOptions(bandList=[1]) # Specify band 1
            vrt_bs_ds = gdal.BuildVRT(backscatter_band1_vrt, aligned_backscatter_full_path, options=bs_vrt_options)
            if vrt_bs_ds is not None:
                 vrt_bs_ds = None # Close VRT
                 if os.path.exists(backscatter_band1_vrt):
                     feature_list_for_stacking.append(backscatter_band1_vrt)
                     band_names_for_stacking.append('Backscatter')
                     print(f" -> Added Backscatter from: {backscatter_band1_vrt}")
                 else:
                     print(f"Error: Backscatter VRT file not found after creation: {backscatter_band1_vrt}")
                     remove_feature('Backscatter', current_classification_features)
            else:
                print(f"Error: Failed VRT creation for Backscatter (gdal.BuildVRT returned None).")
                remove_feature('Backscatter', current_classification_features)
        except Exception as e:
            print(f"Error creating Backscatter VRT: {e}.")
            remove_feature('Backscatter', current_classification_features)

    # 5c. Add Terrain Derivatives (Directly using the TIF files generated in Step 3)
    terrain_features_to_stack = ['Slope', 'Aspect', 'TRI', 'TPI', 'Roughness']
    for feature_name in terrain_features_to_stack:
        if feature_name in current_classification_features:
            # Check if the feature was successfully generated in Step 3
            feature_path = terrain_feature_files_generated.get(feature_name)
            if feature_path and os.path.exists(feature_path):
                 feature_list_for_stacking.append(feature_path)
                 band_names_for_stacking.append(feature_name)
                 print(f" -> Added feature: {feature_name} from {feature_path}")
            else:
                 print(f"Warning: Skipping {feature_name} (file missing or not generated successfully in Step 3).")
                 remove_feature(feature_name, current_classification_features) # Remove if file not found

    # --- 5d. Final Check and Report ---
    print(f"\nFinal features prepared for stacking ({len(feature_list_for_stacking)}): {band_names_for_stacking}")
    if not feature_list_for_stacking:
        sys.exit("Error: No features successfully prepared for stacking. Cannot proceed.")

    # --- !! Explicit Check for Backscatter Inclusion !! ---
    backscatter_was_requested = 'Backscatter' in INITIAL_CLASSIFICATION_FEATURES
    backscatter_is_included = 'Backscatter' in band_names_for_stacking

    if backscatter_was_requested and not backscatter_is_included:
        print("\n" + "="*60)
        print("IMPORTANT WARNING: 'Backscatter' was requested but failed to be included.")
        print("Check previous log messages for errors related to:")
        print("  - Input file missing/empty (Step 1).")
        print("  - Alignment (Step 4).")
        print("  - VRT creation (Step 5b).")
        print("Proceeding without Backscatter.")
        print("="*60)
        # Allow script to continue without backscatter if requested but failed
    elif backscatter_is_included:
        print(">>> Backscatter IS INCLUDED in the final feature stack.")
    elif not backscatter_was_requested:
         print(">>> Backscatter was NOT REQUESTED and is not included.")
    # --- !! End Explicit Check !! ---


    # --- 6. Stack Prepared Features into Final Multi-band TIF ---
    print("\n--- 6. Stacking Final Features ---")
    final_stack_profile = None # Initialize profile variable for the final stack
    try:
        print(f"Building final VRT from {len(feature_list_for_stacking)} sources: {stacked_features_vrt}")
        # Use separate=True to ensure each input becomes a separate band in the VRT
        vrt_options = gdal.BuildVRTOptions(separate=True)
        vrt_ds = gdal.BuildVRT(stacked_features_vrt, feature_list_for_stacking, options=vrt_options)
        if vrt_ds is None: sys.exit("Error: Failed to build final VRT (gdal.BuildVRT returned None).")
        expected_bands = len(feature_list_for_stacking)
        if vrt_ds.RasterCount != expected_bands:
             actual_bands = vrt_ds.RasterCount; vrt_ds = None # Close before exiting
             sys.exit(f"Error: Final VRT band count mismatch ({actual_bands} vs {expected_bands}). Check input files and VRT creation logs.")
        vrt_ds = None # Close VRT dataset handle

        print(f"Translating final VRT to TIF: {stacked_features_tif}")
        # Determine NoData value for the final stack (use bathy's if available, else default)
        final_stack_nodata = bathy_nodata if bathy_nodata is not None else -9999.0
        translate_options = gdal.TranslateOptions(format='GTiff', noData=final_stack_nodata, creationOptions=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])
        stack_ds = gdal.Translate(stacked_features_tif, stacked_features_vrt, options=translate_options)
        if stack_ds is None: sys.exit("Error: Failed to translate final VRT to TIF (gdal.Translate returned None).")

        # Double check band count in final TIF
        if stack_ds.RasterCount != len(band_names_for_stacking):
             actual_bands = stack_ds.RasterCount; stack_ds = None # Close before exiting
             sys.exit(f"Error: Final TIF band count mismatch ({actual_bands} vs {len(band_names_for_stacking)}). Check VRT translation logs.")

        # Set band descriptions in the final TIF for clarity
        print(f"Setting band descriptions for: {', '.join(band_names_for_stacking)}")
        for i, band_name in enumerate(band_names_for_stacking): # Use final list
            try:
                band = stack_ds.GetRasterBand(i + 1)
                band.SetDescription(band_name)
                band.FlushCache() # Ensure description is written
            except Exception as e: print(f"Warning: Failed set band {i+1} ('{band_name}') description: {e}")
        print(f"Final stacked TIF created successfully.")
        stack_ds = None # Close the final TIF dataset

        # Read the profile of the successfully created stack_tif for later use (saving classifications)
        with rasterio.open(stacked_features_tif) as src:
            final_stack_profile = src.profile
            print(f"Successfully read profile from final stack: {stacked_features_tif}")

    except Exception as e: sys.exit(f"Error during final stacking (VRT build or TIF translate): {e}")
    # Ensure we have the profile for saving classification outputs
    if final_stack_profile is None:
         sys.exit("FATAL: Could not obtain profile from the final stacked features TIF after creation.")


    # --- 7. Load Ground Truth and Project Coordinates ---
    print("\n--- 7. Loading Ground Truth Data ---")
    raster_crs = None # Initialize raster_crs
    gdf = None # Initialize GeoDataFrame
    try:
        gt_df = pd.read_csv(gt_csv_path, encoding='latin-1') # Try common encoding for potential special chars
        print(f"Loaded {len(gt_df)} ground truth points.")
        required_cols = ['Longitude', 'Latitude', 'Class']
        if not all(col in gt_df.columns for col in required_cols): sys.exit(f"CSV must contain columns: {required_cols}")

        # Create GeoDataFrame assuming input coordinates are WGS84 (EPSG:4326)
        gdf = gpd.GeoDataFrame(gt_df, geometry=gpd.points_from_xy(gt_df.Longitude, gt_df.Latitude), crs='EPSG:4326')

        # Get target CRS directly from the final stack profile obtained in Step 6
        target_crs_from_profile = final_stack_profile.get('crs')
        if target_crs_from_profile is None:
             print("Warning: Final stack profile has no CRS information. Cannot reproject ground truth accurately.")
        else:
             # Use the CRS object directly if it's already a CRS object (common with Rasterio profiles)
             if isinstance(target_crs_from_profile, rasterio.crs.CRS):
                 raster_crs = target_crs_from_profile
                 print(f"Using Raster CRS from Stack Profile: {raster_crs.to_string()}")
             else:
                 # Attempt to create CRS object from WKT (or other string format), handling potential errors
                 try:
                     # Try creating from whatever string format is in the profile
                     raster_crs = rasterio.crs.CRS.from_string(str(target_crs_from_profile))
                     print(f"Created CRS object from Stack Profile string: {raster_crs.to_string()}")
                 except Exception as e_crs:
                     print(f"Warning: Error creating CRS object from final stack profile CRS string ('{target_crs_from_profile}'): {e_crs}")
                     print("Attempting reprojection using the raw string (might be less robust).")
                     # Fallback: Use the raw string directly in to_crs, GeoPandas might handle common formats (like WKT or EPSG codes)
                     raster_crs = str(target_crs_from_profile)

        # Reproject Ground Truth if raster CRS was successfully determined and differs from input
        if raster_crs:
            print(f"Ground Truth Input CRS: {gdf.crs.to_string()}")
            # Reproject only if CRSs are different
            try:
                # GeoPandas needs a CRS representation it understands (CRS object, WKT string, EPSG code etc.)
                target_crs_for_reproj = raster_crs
                if gdf.crs != target_crs_for_reproj:
                    print(f"Projecting ground truth points to target CRS: {target_crs_for_reproj}...")
                    gdf = gdf.to_crs(target_crs_for_reproj)
                    print("Projection complete.")
                else:
                    print("Ground truth CRS already matches raster CRS. No reprojection needed.")
            except Exception as reproj_e:
                print(f"ERROR during ground truth reprojection: {reproj_e}")
                print("Cannot proceed without correctly projected ground truth points.")
                sys.exit(1)
        else:
            print("Proceeding without ground truth reprojection due to missing/invalid raster CRS.")
            print("Ensure ground truth coordinates ALREADY match the raster projection: {ref_proj_wkt}")


        # Extract projected coordinates for sampling
        gdf['Easting'] = gdf.geometry.x
        gdf['Northing'] = gdf.geometry.y
        print("Ground truth data (head) after potential reprojection:")
        print(gdf.head())

    except FileNotFoundError: sys.exit(f"Error: Ground truth file not found at {gt_csv_path}")
    except Exception as e: sys.exit(f"Error loading or processing ground truth data: {e}")


    # --- 8. Extract Training Data from Stacked Raster ---
    print("\n--- 8. Extracting Training Data ---")
    coords = [(x, y) for x, y in zip(gdf.Easting, gdf.Northing)]
    class_to_int = {}; int_to_class = {}
    # Define variables to ensure they exist in case of exceptions
    X = None; y_int = None; training_data = None
    actual_band_names = band_names_for_stacking # Use the final list derived in step 5/6

    try:
        with rasterio.open(stacked_features_tif) as src:
             # Verify band count matches expected features
             if len(actual_band_names) != src.count:
                 sys.exit(f"FATAL Error: Sampling band name count ({len(actual_band_names)}) != raster band count ({src.count}) in {stacked_features_tif}")

             print(f"Sampling {src.count} raster bands at {len(coords)} locations: {', '.join(actual_band_names)}")
             # rasterio.sample returns a generator; convert to list of numpy arrays
             sampled_values_list = list(src.sample(coords))
             # Stack the arrays vertically to create a 2D numpy array (n_points, n_features)
             sampled_values_np = np.vstack(sampled_values_list)
             # Create DataFrame with correct band names
             sampled_df = pd.DataFrame(sampled_values_np, columns=actual_band_names)

        # Combine sampled features with coordinates and original class label from GeoDataFrame
        # Reset index on both DataFrames to ensure alignment
        training_data = pd.concat([
            gdf[['Easting', 'Northing', 'Class']].reset_index(drop=True),
            sampled_df.reset_index(drop=True)
        ], axis=1)

        print(f"Training data shape before NoData/NaN removal: {training_data.shape}")
        # Use the NoData value obtained from the final stack's profile
        nodata_val_to_check = final_stack_profile.get('nodata')
        cols_to_check = actual_band_names # Check all feature columns for NoData/NaN
        initial_rows = len(training_data)

        # Create masks for NoData and NaN separately for clarity, then combine
        is_nodata = pd.DataFrame(False, index=training_data.index, columns=cols_to_check)
        # Check against specific NoData value only if it's defined
        if nodata_val_to_check is not None:
             # Handle potential floating point comparisons carefully if nodata is float
             if np.issubdtype(training_data[cols_to_check].values.dtype, np.floating) and np.isnan(nodata_val_to_check):
                  # If NoData is NaN, isnull() check below handles it
                  pass
             else:
                  is_nodata = training_data[cols_to_check] == nodata_val_to_check

        # Check for NaN values in feature columns
        is_nan = training_data[cols_to_check].isnull()

        # Combine masks: drop row if ANY feature is NoData OR NaN
        rows_to_drop = (is_nodata | is_nan).any(axis=1)

        training_data = training_data[~rows_to_drop].copy() # Keep rows where the combined mask is False
        dropped_rows = initial_rows - len(training_data)
        print(f"Removed {dropped_rows} rows containing NoData ({nodata_val_to_check}) or NaN values.")

        print(f"Final training data shape: {training_data.shape}")
        print("Sample of final training data (first 5 rows):"); print(training_data.head())
        if training_data.empty: sys.exit("Error: No valid training data remaining after NoData/NaN removal.")

        # Map string class labels to integers for scikit-learn compatibility
        # Ensure consistent sorting, handle numeric classes by converting to string first
        unique_classes = sorted(training_data['Class'].astype(str).unique())
        class_to_int = {label: i for i, label in enumerate(unique_classes)}
        int_to_class = {i: label for label, i in class_to_int.items()}
        print("\nClass Label Mapping (String -> Integer):"); print(class_to_int)
        # Create the integer target variable 'y_int'
        y_int = training_data['Class'].astype(str).map(class_to_int)

        # Define features (X) using the actual band names from the stack
        X = training_data[actual_band_names]
        print("\nClass Counts in Training Data (Integer Representation):"); print(y_int.value_counts().sort_index())

    except rasterio.RasterioIOError: sys.exit(f"Error opening stacked raster {stacked_features_tif} for sampling.")
    except IndexError as e: sys.exit(f"Error during sampling, possibly due to coordinates outside raster bounds? Details: {e}")
    except Exception as e: sys.exit(f"An unexpected error occurred during training data extraction: {e}")


    # --- 9. Supervised Classification: Random Forest ---
    print("\n--- 9. Random Forest Classification ---")
    rf_success = False # Flag to track successful completion
    classification_rf_map = None # Initialize map variable
    data_for_prediction = None # Initialize variable for prediction data (used by K-Means too)
    valid_mask = None # Initialize variable for valid mask (used by K-Means too)
    profile_for_output = None # Initialize profile for saving classification maps
    raster_shape = None # Initialize raster shape

    # Check if X and y_int were successfully created in the previous step
    if X is None or y_int is None:
        sys.exit("Error: Training data (X or y_int) not available for Random Forest.")
    try:
        # Split data into training and testing sets using integer labels for stratification
        X_train, X_test, y_train_int, y_test_int = train_test_split(
            X, y_int, test_size=test_size_rf, random_state=random_state, stratify=y_int
        )
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        # Adjust CV folds if the smallest class count in the training set is smaller than requested folds
        min_class_count_train = y_train_int.value_counts().min()
        current_cv_folds = rf_cv_folds # Use a temporary variable for CV folds for this run
        if min_class_count_train > 0 and min_class_count_train < current_cv_folds:
            print(f"Warning: Smallest class in training set ({min_class_count_train}) < CV folds ({current_cv_folds}). Adjusting CV folds to {min_class_count_train}.")
            current_cv_folds = min_class_count_train
        elif min_class_count_train <= 1:
            # If a class has only 1 sample in the training set, stratified CV will fail.
            sys.exit(f"Error: Smallest class size in training set ({min_class_count_train}) is <= 1, cannot perform stratified CV.")

        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [50, 100, 200, 300], # Number of trees
            'max_depth': [None, 10, 20],       # Max depth (None = grow fully)
            'min_samples_split': [2, 5],      # Min samples required to split an internal node
            'min_samples_leaf': [1, 3],        # Min samples required to be at a leaf node
            'class_weight': ['balanced']      # Adjust weights inversely proportional to class frequencies
        }

        rf = RandomForestClassifier(random_state=random_state)
        print(f"Performing GridSearchCV for Random Forest (CV={current_cv_folds})...")
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=current_cv_folds, n_jobs=rf_n_jobs, verbose=1, scoring='accuracy')
        grid_search.fit(X_train, y_train_int) # Train using integer labels

        print(f"\nGridSearchCV Best Parameters Found: {grid_search.best_params_}")
        best_rf = grid_search.best_estimator_ # Get the best model identified by the search

        # Evaluate the best model on the held-out test set
        print("\nEvaluating best RF model on the test set:")
        y_pred_rf_int = best_rf.predict(X_test) # Predict integer labels

        # Calculate metrics using integer labels
        accuracy_rf = accuracy_score(y_test_int, y_pred_rf_int) * 100
        kappa_rf = cohen_kappa_score(y_test_int, y_pred_rf_int)
        cm_rf = confusion_matrix(y_test_int, y_pred_rf_int)

        # Use original string labels (via int_to_class map) for classification report clarity
        report_target_names = [int_to_class[i] for i in sorted(y_int.unique())] # Get string names in integer order
        report_rf = classification_report(y_test_int, y_pred_rf_int, target_names=report_target_names, zero_division=0)

        print(f"Overall Accuracy: {accuracy_rf:.2f}%")
        print(f"Kappa Coefficient: {kappa_rf:.3f}")
        print("Confusion Matrix (Rows: True Int Label, Cols: Pred Int Label):\n", cm_rf)
        print("\nClassification Report (using original class names):\n", report_rf)

        # Feature Importance from the best model
        importances = best_rf.feature_importances_ * 100
        # Ensure feature names match the columns used for training (X.columns)
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns, # Use columns from the training DataFrame X
            'Importance (%)': importances
        }).sort_values(by='Importance (%)', ascending=False)
        print("\nFeature Importances:"); print(feature_importance_df)

        # Plot Feature Importance
        try:
            fig_imp, ax_imp = plt.subplots(figsize=(10, 6));
            feature_importance_df.plot(kind='bar', x='Feature', y='Importance (%)', ax=ax_imp, legend=False)
            ax_imp.set_title('Random Forest Feature Importance'); ax_imp.set_ylabel('Importance (%)');
            ax_imp.tick_params(axis='x', rotation=45); plt.tight_layout(); plt.show()
        except Exception as plot_e:
            print(f"Warning: Failed to display feature importance plot: {plot_e}")


        # Predict classes for the entire raster using the best trained RF model
        print("\nPredicting classes for the entire raster using the best RF model...")
        # Reload raster data and mask within this step to ensure consistency
        with rasterio.open(stacked_features_tif) as src:
            # Read all bands into a 3D numpy array (bands, rows, cols)
            raster_data = src.read()
            # Get the profile (metadata) from the source stacked file for output
            profile_for_output = src.profile # Use this profile as basis for classification outputs
            raster_shape = (src.height, src.width) # Store shape (rows, cols)
            # Use NoData value from stack profile, fall back to value used during stack creation
            nodata_val_to_check = src.nodata if src.nodata is not None else final_stack_nodata

            # Create a robust mask for NoData/NaN across all bands
            # Initialize mask assuming all pixels are valid
            valid_mask = np.ones(raster_shape, dtype=bool)
            for i in range(src.count):
                band_data = raster_data[i]
                band_mask = np.ones(raster_shape, dtype=bool) # Mask for current band
                # Handle potential float NoData comparison issues
                if nodata_val_to_check is not None:
                    if np.issubdtype(band_data.dtype, np.floating) and np.isnan(nodata_val_to_check):
                        band_mask = ~np.isnan(band_data) # Valid if not NaN
                    else:
                        # Valid if not equal to NoData AND not NaN
                        band_mask = (band_data != nodata_val_to_check) & (~np.isnan(band_data))
                else: # No specific NoData value, only check for NaN
                    band_mask = ~np.isnan(band_data)

                # Update the overall valid_mask: a pixel is valid only if valid in ALL bands
                valid_mask &= band_mask

            n_samples = valid_mask.sum(); n_features = src.count
            print(f"Number of valid pixels for prediction: {n_samples}")
            if n_features != len(actual_band_names):
                 sys.exit(f"FATAL: Prediction raster band count ({n_features}) doesn't match expected features ({len(actual_band_names)}).")

            # Prepare data in (n_samples, n_features) shape required by scikit-learn
            # Extract data only for valid pixels and transpose
            data_for_prediction = np.vstack([raster_data[i][valid_mask] for i in range(n_features)]).T

            if data_for_prediction.size == 0: sys.exit("Error: No valid pixels found in the raster for prediction.")

        # Predict using the trained best RF model
        print(f"Predicting on {data_for_prediction.shape[0]} samples with {data_for_prediction.shape[1]} features...")
        predicted_labels_int = best_rf.predict(data_for_prediction) # Predicts integer labels

        # Create the output classification map array
        # Initialize with a NoData value (e.g., -99). Choose dtype suitable for integer labels + NoData.
        classification_rf_map = np.full(raster_shape, -99, dtype=np.int16)
        # Fill in the predicted integer labels only for the valid pixels
        classification_rf_map[valid_mask] = predicted_labels_int

        # --- Save the RF classification raster using Rasterio ---
        print(f"Saving RF classification map (integer labels): {rf_classified_file}")
        # Update the profile read from the source stack for single-band integer output
        rf_profile_out = profile_for_output.copy()
        rf_profile_out.update(dtype=rasterio.int16, count=1, nodata=-99) # Set output dtype, band count, nodata
        # Write the numpy array using the updated profile via the helper function
        save_raster_rio(rf_classified_file, classification_rf_map, rf_profile_out, nodata_value=-99)
        # --------------------------------------------------------

        # Plot RF classification map
        num_classes_rf = len(class_to_int) # Number of unique classes from ground truth
        # Use a discrete colormap with enough colors for the classes
        cmap_rf = ListedColormap(plt.cm.tab10.colors[:num_classes_rf])
        plot_raster(rf_classified_file, "Random Forest Classification", cmap=cmap_rf, label='Class ID (Integer)')

        rf_success = True # Mark RF step as successful

    except NotFittedError: sys.exit("Error: GridSearchCV or RandomForest model was not fitted before prediction.")
    except ValueError as e: sys.exit(f"Error during RF training or prediction: {e}. Check data shapes/types and feature consistency.")
    except MemoryError: sys.exit("Memory Error during RF prediction. Input raster might be too large.")
    except Exception as e: sys.exit(f"An unexpected error occurred during Random Forest stage: {e}")


    # --- 10. Unsupervised Classification: K-Means ---
    print("\n--- 10. K-Means Clustering ---")
    kmeans_success = False # Flag to track successful completion
    classification_kmeans_map = None # Initialize variable
    try:
        # Ensure data_for_prediction and valid_mask are available from RF step
        if data_for_prediction is None or valid_mask is None or profile_for_output is None or raster_shape is None:
             print("Error: Required data/metadata (prediction data, valid mask, profile, shape) from RF step not available for K-Means.")
             # Don't exit, just skip K-Means
             raise RuntimeError("Missing prerequisite data for K-Means.")

        # Scaling is crucial for distance-based algorithms like K-Means
        print("Scaling data for K-Means...")
        scaler = StandardScaler()
        # Check for constant features before scaling (StandardScaler gives warning/error if variance is zero)
        scaled_data_for_kmeans = data_for_prediction.copy() # Work on a copy
        stds = np.std(scaled_data_for_kmeans, axis=0)
        constant_feature_indices = np.where(stds == 0)[0]
        non_constant_feature_indices = np.where(stds != 0)[0]

        if constant_feature_indices.size > 0:
            constant_feature_names = [actual_band_names[i] for i in constant_feature_indices]
            print(f"Warning: Constant features detected: {constant_feature_names}. These features will not be scaled.")
            # Only scale the non-constant features if any exist
            if non_constant_feature_indices.size > 0:
                print(f"Scaling {non_constant_feature_indices.size} non-constant features...")
                scaled_data_for_kmeans[:, non_constant_feature_indices] = scaler.fit_transform(scaled_data_for_kmeans[:, non_constant_feature_indices])
            else:
                print("Warning: All features are constant. K-Means might produce trivial clustering.")
        elif non_constant_feature_indices.size > 0: # Only scale if there are non-constant features
             print(f"Scaling all {non_constant_feature_indices.size} features...")
             scaled_data_for_kmeans = scaler.fit_transform(scaled_data_for_kmeans)
        else: # Handle case where data_for_prediction might be empty (should have exited earlier)
             print("Warning: No data available for scaling (data_for_prediction is empty).")
             raise RuntimeError("Empty data cannot be used for K-Means.")

        # Initialize and fit K-Means model
        print(f"Fitting K-Means model (k={n_kmeans_clusters})...")
        # Use n_init=10 (or 'auto') to run multiple initializations and choose the best
        kmeans = KMeans(n_clusters=n_kmeans_clusters, random_state=random_state, n_init=10)
        kmeans.fit(scaled_data_for_kmeans)

        # Get cluster labels for the valid pixels (labels are 0-based integers)
        kmeans_labels = kmeans.labels_

        # K-Means labels start from 0. Add 1 to make them 1-based for typical map representation (optional, but common)
        kmeans_labels_1based = kmeans_labels + 1

        # Create the K-Means classification map array
        # Use the same shape and NoData value as the RF map for consistency
        classification_kmeans_map = np.full(raster_shape, -99, dtype=np.int16)
        # Assign the 1-based cluster labels to the valid pixels
        classification_kmeans_map[valid_mask] = kmeans_labels_1based

        # --- Save the K-Means classification raster using Rasterio ---
        print(f"Saving K-Means classification map (1-based cluster IDs): {kmeans_classified_file}")
        # Use the same output profile structure as the RF map
        kmeans_profile_out = profile_for_output.copy() # Reuse profile from stack
        kmeans_profile_out.update(dtype=rasterio.int16, count=1, nodata=-99) # Ensure correct dtype/nodata
        # Save using the helper function
        save_raster_rio(kmeans_classified_file, classification_kmeans_map, kmeans_profile_out, nodata_value=-99)
        # ----------------------------------------------------------

        # Plot K-Means classification map
        # Use a discrete colormap with enough colors for the k clusters
        cmap_kmeans = ListedColormap(plt.cm.tab10.colors[:n_kmeans_clusters])
        plot_raster(kmeans_classified_file, f"K-Means Clustering (k={n_kmeans_clusters})", cmap=cmap_kmeans, label='Cluster ID (1-based)')

        kmeans_success = True # Mark K-Means step as successful

    except RuntimeError as e: # Catch explicit runtime errors raised above
        print(f"Skipping K-Means due to runtime error: {e}")
    except MemoryError:
        print(f"Memory Error during K-Means processing. Skipping K-Means.")
    except Exception as e:
        print(f"Warning: An unexpected error occurred during K-Means clustering: {e}. Skipping K-Means.")


    # --- 11. Cleanup Intermediate Files --- (Renumbered from 12)
    # List includes VRTs and the multi-band aligned backscatter (if created).
    # Single-band VRTs (bathy_b1, backscatter_b1) are derived from these.
    files_to_remove = [
        bathy_band1_vrt,
        backscatter_band1_vrt,
        stacked_features_vrt,
        backscatter_aligned_full_file # The full multi-band aligned file (if created)
    ]
    if cleanup_intermediate:
        print("\n--- 11. Cleaning Up Intermediate Files ---")
        for f in files_to_remove:
            # Check if the path variable exists (e.g., backscatter VRT might not if BS failed) and if the file exists
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                    print(f"Removed: {f}")
                except OSError as e:
                    print(f"Error removing {f}: {e}")
            # else:
            #     print(f"Skipping removal (file not created or path invalid): {f}")
    else:
        print("\n--- 11. Skipping Cleanup ---")
        print("Intermediate files (VRTs, full aligned BS TIF) were kept.")
        print("Feature GeoTIFFs (Slope, Aspect, etc.) and the final stacked TIF are also kept.")


    # --- End Script ---
    end_time = time.time()
    print(f"\n--- Script Finished in {end_time - start_time:.2f} seconds ---")