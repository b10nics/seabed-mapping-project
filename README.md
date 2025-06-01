# Seabed Classification using Integrated Acoustic Data and Machine Learning

**Authors (of original research concept):** Esraa E. Abouelmaaty, Obed Omane Okyere, José Manuel Echevarría Rubio
**Date (of original research concept):** April 8th, 2025
**Script Author/Maintainer:** [b10nics]

## Project Overview

This project implements a seabed classification workflow using integrated multibeam echosounder (MBES) data and machine learning algorithms. It is based on the research paper titled "Seabed Classification in Northwest Fernandina Island using Integrated Acoustic Data and Machine Learning: Comparing Random Forest Performance at 5m vs. 10m Resolution."

The primary goal is to classify the seabed into distinct categories based on bathymetry, backscatter, and derived terrain features. This script automates the following processes:
1.  **Terrain Analysis:** Calculation of morphological features (Slope, Aspect, TRI, TPI, Roughness, Hillshade) from a bathymetry GeoTIFF.
2.  **Data Alignment:** Optional alignment of a backscatter GeoTIFF to the bathymetry grid.
3.  **Feature Engineering:** Stacking of bathymetry, (aligned) backscatter, and terrain features into a multi-band GeoTIFF.
4.  **Ground Truth Integration:** Extraction of feature values at ground truth point locations (from a CSV file), including reprojection to match the raster CRS.
5.  **Supervised Classification:** Training and application of a Random Forest classifier, including hyperparameter tuning via GridSearchCV.
6.  **Unsupervised Classification:** Application of K-Means clustering for an exploratory perspective.
7.  **Output Generation:** Creation of GeoTIFF classification maps for both Random Forest and K-Means results.

## Recent Script Updates
*   **Path Handling:** The script now uses absolute paths for input and output directories, derived from the script's own location. This resolves previous "No such file or directory" errors by ensuring file operations target the correct locations, irrespective of the initial working directory or `os.chdir()` calls (which has been removed).
*   **PROJ_LIB:** Added `/opt/conda/share/proj` to the search paths for the PROJ library, enhancing compatibility in Conda environments.
*   **Plotting (Ongoing):** The plotting function for rasters has been simplified. Note: There are known issues with `FormatStrFormatter` for discrete classification map colorbars which may cause plotting errors (these errors do not affect the generation of the GeoTIFF classification maps themselves).

## Summary of Latest Run Results (as of YYYY-MM-DD)

The script successfully processed the example dataset (`bathy_cube_10_filled_5x5.tiff`, `back_10_filled_5x5.tiff`, `ground_truth_samples_removed.csv`):

*   **Input Data:**
    *   Bathymetry: 3299x1907 pixels, 10m resolution, UTM zone 15S.
    *   Backscatter: Successfully aligned to the bathymetry grid.
    *   Ground Truth: 292 points loaded and reprojected from EPSG:4326 to EPSG:32715.
*   **Feature Generation:**
    *   All terrain derivatives (Slope, Aspect, TRI, TPI, Roughness, Hillshade) were successfully generated.
    *   The final stacked raster for classification contained 7 features: Depth, Backscatter, Slope, Aspect, TRI, TPI, and Roughness.
*   **Training Data:**
    *   Feature values were extracted for all 292 ground truth points; no points were dropped due to NoData values.
    *   Classes were mapped to integers (e.g., 'Biogenic mat': 0, 'Lava flows': 4).
*   **Random Forest Classification:**
    *   The model was trained on 204 samples and tested on 88 samples.
    *   **Best Parameters (GridSearchCV):** `{'class_weight': 'balanced', 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}`
    *   **Test Set Performance:**
        *   Overall Accuracy: 77.27%
        *   Kappa Coefficient: 0.727
    *   **Feature Importances (Top 3):**
        1.  Depth: ~29.22%
        2.  TPI: ~14.32%
        3.  Backscatter: ~12.46%
    *   The full classification map (`classification_rf.tif`) was generated for 485,932 valid pixels.
*   **K-Means Clustering:**
    *   Data was scaled, and K-Means (k=7) was successfully applied.
    *   The K-Means classification map (`classification_kmeans.tif`) was generated.
*   **Execution Time:** Approximately 17-19 seconds.
*   **Known Issues (from log):**
    *   Plotting of the final classification maps encountered an error: "This method only works with the ScalarFormatter." This is a Matplotlib issue with the current colorbar formatter for discrete integer maps and does not affect the GeoTIFF output.
    *   A UserWarning from scikit-learn (`X does not have valid feature names, but RandomForestClassifier was fitted with feature names`) was observed during prediction. This is generally benign if the order and number of features are consistent, but ideally, prediction data should also be a DataFrame with matching column names.

## Input Data Requirements

The script expects the following input files to be placed in the `input_data` directory (relative to the script's location):

1.  **Bathymetry File (`bathy_file_name`):**
    *   Format: GeoTIFF (`.tiff`, `.tif`)
    *   Example: `bathy_cube_10_filled_5x5.tiff`
2.  **Backscatter File (`backscatter_file_name`):** (Optional)
    *   Format: GeoTIFF (`.tiff`, `.tif`)
    *   Example: `back_10_filled_5x5.tiff`
3.  **Ground Truth CSV File (`ground_truth_csv_name`):**
    *   Format: CSV (`.csv`)
    *   Required Columns: `Longitude`, `Latitude`, `Class`
    *   Example: `ground_truth_samples_removed.csv`

## Output Files

The script generates output files in the `Outputs_SeabedClassification` directory (relative to the script's location):
*   **Terrain Feature Rasters:** `slope.tif`, `aspect.tif`, `tri.tif`, `tpi.tif`, `roughness.tif`, `hillshade.tif`.
*   **Processed Backscatter:** `backscatter_aligned_to_bathy.tif`.
*   **Stacked Features:** `stacked_features_for_classification.tif`.
*   **Classification Maps:** `classification_rf.tif`, `classification_kmeans.tif`.
*   Intermediate VRT files may also be present.

## Software and Libraries

Python 3 with the following major libraries: GDAL/OGR, Rasterio, GeoPandas, Pandas, NumPy, Scikit-learn, Matplotlib.
(See `requirements.txt` for a more detailed list).

## Setup and Installation

1.  **Clone the repository.**
2.  **Install GDAL:** System-wide or via Conda is recommended (e.g., `sudo apt install gdal-bin libgdal-dev python3-gdal` or `conda install -c conda-forge gdal`).
3.  **Create a Python Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
4.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure `requirements.txt` reflects the necessary packages).*
5.  **PROJ_LIB Environment Variable:** The script attempts to set this. If CRS errors persist, ensure it points to your PROJ data directory (e.g., `/usr/share/proj`, `/opt/conda/envs/your_env/share/proj`).

## How to Run

1.  **Prepare Data:** Place input files in the `input_data` sub-directory.
2.  **Configure Script:**
    *   Open `Seabed_classification_mod.py`.
    *   Verify `data_dir_relative` and `output_dir_relative` if your project structure differs.
    *   Adjust `bathy_file_name`, `backscatter_file_name`, `ground_truth_csv_name` if needed.
    *   Review other parameters like `n_kmeans_clusters`, `rf_cv_folds`, etc.
3.  **Execute:**
    ```bash
    python Seabed_classification_mod.py
    ```
4.  **Check Outputs:** In the `Outputs_SeabedClassification` sub-directory.

## Creating `requirements.txt`

A basic `requirements.txt` would be:
```numpy
pandas
geopandas
# GDAL (python3-gdal) usually installed via system/conda
rasterio
matplotlib
scikit-learn