# Import the os module for file and directory operations
import os
# Import numpy for numerical operations and array handling
import numpy as np
# Import rasterio for working with geospatial raster data (GeoTIFF files)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import rasterio
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import ndimage
from scipy.special import expit
import pandas as pd

BASE_DIR = '.'
OUTPUT_DIR = 'flood_risk_outputs'

# Check if the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Modified function to handle data leakage and geographic splitting
# Modified function to handle ocean masking and use weighted features
def calculate_flood_risk(elevation, flow_direction, hand, upstream_area):
    """
    Calculate flood risk based on weighted features.
    Handles ocean masking and geographic splitting.
    """
    # Define feature weights (Derived from linreg.py)
    weights = {
        'elevation': 0.5819,
        'hand': 0.3723,
        'upstream_area': 0.0304,
        'flow_convergence': 0.0154,
        'river_width': 0.0000  # This feature is not used but kept in weights dict for reference
    }
    
    land_mask = elevation > 0
    land_indices = np.where(land_mask)

    # Four features: elevation, hand, upstream_area, flow_convergence
    X = np.zeros((len(land_indices[0]), 4))

    # Process elevation: lower = higher risk
    elevation_feature = -elevation[land_indices]
    X[:, 0] = elevation_feature

    # Process HAND: replace negatives with 0
    hand_feature = np.where(hand > 0, hand, 0)[land_indices]
    X[:, 1] = hand_feature

    # Process upstream area (ensure only positive values)
    upstream_feature = np.where(upstream_area > 0, upstream_area, 0)[land_indices]
    X[:, 2] = upstream_feature
    
    # Calculate flow convergence by counting unique flow directions in 3x3 neighborhood
    flow_conv = ndimage.generic_filter(flow_direction, lambda x: len(np.unique(x)), size=3)
    flow_feature = flow_conv[land_indices]
    X[:, 3] = flow_feature

    coords = np.array(land_indices).T

    def geographic_split(X, coords, test_size=0.2, random_state=42):
        center_row = np.mean(coords[:, 0])
        center_col = np.mean(coords[:, 1])
        distances = np.sqrt((coords[:, 0] - center_row)**2 + (coords[:, 1] - center_col)**2)
        sorted_indices = np.argsort(distances)
        n_train = int((1 - test_size) * len(X))
        train_indices = np.array([])
        test_indices = np.array([])
        n_bands = 5
        samples_per_band = len(sorted_indices) // n_bands

        for i in range(n_bands):
            band_start = i * samples_per_band
            band_end = (i + 1) * samples_per_band if i < n_bands - 1 else len(sorted_indices)
            band_indices = sorted_indices[band_start:band_end]
            if i % 2 == 0:
                train_indices = np.append(train_indices, band_indices)
            else:
                test_indices = np.append(test_indices, band_indices)

        if len(train_indices) > n_train:
            n_to_move = len(train_indices) - n_train
            indices_to_move = np.random.choice(train_indices, size=int(n_to_move), replace=False)
            train_indices = np.setdiff1d(train_indices, indices_to_move)
            test_indices = np.append(test_indices, indices_to_move)
        elif len(train_indices) < n_train:
            n_to_move = n_train - len(train_indices)
            indices_to_move = np.random.choice(test_indices, size=int(n_to_move), replace=False)
            test_indices = np.setdiff1d(test_indices, indices_to_move)
            train_indices = np.append(train_indices, indices_to_move)

        return train_indices.astype(int), test_indices.astype(int)

    train_indices, test_indices = geographic_split(X, coords, test_size=0.2, random_state=42)
    X_train, X_test = X[train_indices], X[test_indices]

    scalers = [MinMaxScaler() for _ in range(X.shape[1])]
    for i in range(X.shape[1]):
        X_train[:, i] = scalers[i].fit_transform(X_train[:, i].reshape(-1, 1)).ravel()
        X_test[:, i] = scalers[i].transform(X_test[:, i].reshape(-1, 1)).ravel()

    X[train_indices] = X_train
    X[test_indices] = X_test

    # Apply feature weights instead of simple average
    feature_weights = np.array([
        weights['elevation'],
        weights['hand'],
        weights['upstream_area'],
        weights['flow_convergence']
    ])
    
    # Compute weighted flood risk
    risk_values = np.dot(X, feature_weights)

    # Apply Z-score standardization
    mean = np.mean(risk_values)
    std = np.std(risk_values)
    z_scores = (risk_values - mean) / std

    # Apply sigmoid to spread values into [0, 1]
    risk_values = expit(z_scores)  # equivalent to 1 / (1 + exp(-z))

    # Create final flood risk array
    flood_risk = np.full_like(elevation, np.nan, dtype=float)
    flood_risk[land_indices] = risk_values

    return flood_risk


def visualize_flood_risk(risk_array, output_path, transform, crs):
    """Visualize the flood risk map and save to file with ocean areas colored black."""
    # Create a figure and get the current axes
    fig = plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Create masked array for visualization
    masked_data = np.ma.masked_invalid(risk_array)
    
    # Define distinct colors with dark purple at the highest values
    colors = [
        '#000033',  # Very Dark Navy (0.00-0.05)
        '#000099',  # Dark Navy (0.05-0.10)
        '#0000FF',  # Blue (0.10-0.15)
        '#0099FF',  # Sky Blue (0.15-0.20)
        '#00FFFF',  # Cyan (0.20-0.25)
        '#00FF99',  # Spring Green (0.25-0.30)
        '#00CC00',  # Green (0.30-0.35)
        '#33FF33',  # Bright Green (0.35-0.40)
        '#99FF33',  # Yellow-Green (0.40-0.45)
        '#CCFF00',  # Lime (0.45-0.50)
        '#FFFF00',  # Yellow (0.50-0.55)
        '#FFCC00',  # Gold (0.55-0.60)
        '#FF9900',  # Orange (0.60-0.65)
        '#FF6600',  # Dark Orange (0.65-0.70)
        '#FF0000',  # Red (0.70-0.75)
        '#CC0000',  # Dark Red (0.75-0.80)
        '#990000',  # Very Dark Red (0.80-0.85)
        '#660066',  # Dark Purple (0.85-0.90)
        '#9900CC',  # Purple (0.90-0.95)
        '#FF00FF',  # Magenta (0.95-1.00)
    ]
    
    # Create custom colormap with 20 distinct bands (for 0.05 increments)
    cmap = LinearSegmentedColormap.from_list('high_contrast', colors, N=20)
    cmap.set_bad('black')  # Set NaN values to black (ocean)
    
    # Create boundaries for distinct color bands with 0.05 increments
    bounds = np.linspace(0, 1, 21)  # 21 boundaries for 20 distinct ranges
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Display the risk array with the custom colormap
    img = ax.imshow(masked_data, cmap=cmap, norm=norm)
    
    # Add a colorbar with tick marks at the boundaries
    cbar = plt.colorbar(img, ax=ax, ticks=bounds)
    cbar.set_label('Flood Hazard Index (0-1)', fontsize=12)
    
    # Format the tick labels to show ranges with 0.05 increments
    tick_labels = [f"{bounds[i]:.2f}-{bounds[i+1]:.2f}" for i in range(len(bounds)-1)]
    # Add an empty string at the beginning for proper alignment
    cbar.set_ticklabels([''] + tick_labels)
    
    # Add a title to the plot
    plt.title('Flood Hazard Map of Japan', fontsize=18, fontweight='bold')
    # Turn off the axis labels and ticks
    plt.axis('off')
    
    # Save the visualization as a PNG file
    plt.savefig(os.path.join(os.path.dirname(output_path), 'flood_hazard_visualization.png'), dpi=300, bbox_inches='tight')
    
    # Save the risk data as a GeoTIFF file for GIS applications
    with rasterio.open(output_path, 'w',
                      driver='GTiff',          # Use GeoTIFF format
                      height=risk_array.shape[0],  # Set height from input array
                      width=risk_array.shape[1],   # Set width from input array
                      count=1,                 # One band/channel
                      dtype=rasterio.float32,  # Use float32 to support NaN values
                      crs=crs,                 # Set coordinate reference system
                      transform=transform,     # Set geospatial transform
                      nodata=np.nan) as dst:   # Set NaN as the nodata value
        # Write the risk array to the first band
        dst.write(risk_array.astype(rasterio.float32), 1)
    
    # Close the plot to free memory
    plt.close()
    # Return the path to the created GeoTIFF file
    return output_path

def main():
    """Simplified workflow that uses existing final files with improved ocean masking."""
    # Print initial message
    print("Starting Flood Risk Assessment using pre-processed files...")
    
    # Print message about loading datasets
    print("Loading prepared datasets...")
    # Open the flow direction file and read its data
    with rasterio.open(os.path.join('flow_direction_final.tif')) as src:
        # Read the first band of data
        flow_direction = src.read(1)
        # Get metadata from the file
        meta = src.meta
        # Get the geospatial transform
        transform = src.transform
        # Get the coordinate reference system
        crs = src.crs
    
    # Open the elevation file and read its data
    with rasterio.open(os.path.join('elevation_final.tif')) as src:
        # Read the first band of data
        elevation = src.read(1)
    
    # Open the HAND (Height Above Nearest Drainage) file and read its data
    with rasterio.open(os.path.join('hand_final.tif')) as src:
        # Read the first band of data
        hand = src.read(1)
    
    # Open the upstream area file and read its data
    with rasterio.open(os.path.join('upstream_area_final.tif')) as src:
        # Read the first band of data
        upstream_area = src.read(1)
    
    print("Calculating flood risk with ocean masking, proper scaling, and geographic splitting...")
    flood_risk = calculate_flood_risk(elevation, flow_direction, hand, upstream_area)
    
    print("Visualizing results")
    risk_geotiff = visualize_flood_risk(flood_risk, os.path.join(OUTPUT_DIR, 'flood_risk.tif'), transform, crs)
    
    print(f"\nFlood risk assessment complete! Results saved to {OUTPUT_DIR}/")

# Check if this script is being run directly (not imported)
if __name__ == "__main__":
    main()
