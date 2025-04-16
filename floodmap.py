# Import the os module for file and directory operations
import os
# Import numpy for numerical operations and array handling
import numpy as np
# Import rasterio for working with geospatial raster data (GeoTIFF files)
import rasterio
# Import matplotlib.pyplot for creating visualizations
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

BASE_DIR = '.'
OUTPUT_DIR = 'flood_risk_outputs'

# Check if the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Modify the calculate_flood_risk function to handle ocean masking
def calculate_flood_risk(elevation, flow_direction, hand, upstream_area, river_width):
    """
    Calculate flood risk based on multiple factors with ML-derived weights and advanced processing.
    Now with proper handling of ocean areas (masked as NaN or no_data values).
    """
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    from scipy import ndimage
    
    # Create a land mask - any pixel with elevation <= 0 is considered ocean
    # You may need to adjust this threshold based on your elevation data
    land_mask = elevation > 0
    
    # Create a new MinMaxScaler instance for normalizing values
    scaler = MinMaxScaler()
    
    # ELEVATION: Implement thresholds with ocean masking
    elevation_norm = np.where(land_mask, elevation, 0)  # Handle non-land areas
    # Invert the values (lower elevation = higher risk value)
    elevation_norm = np.where(land_mask, -elevation_norm, 0)

    # Normalize elevation values to 0-1 range (land areas only)
    elevation_norm_reshaped = elevation_norm.reshape(-1, 1)
    land_indices = np.where(land_mask.reshape(-1))[0]
    if len(land_indices) > 0:  # Only normalize if there are land pixels
        scaler = MinMaxScaler()
        elevation_norm_reshaped[land_indices] = scaler.fit_transform(elevation_norm_reshaped[land_indices])
    elevation_norm = elevation_norm_reshaped.reshape(elevation.shape)

    # Replace negative HAND values with 0 (HAND should be non-negative) and apply land mask
    hand_norm = np.where((hand > 0) & land_mask, hand, 0)
    # Normalize the transformed HAND values (land areas only)
    hand_norm_reshaped = hand_norm.reshape(-1, 1)
    land_indices = np.where(land_mask.reshape(-1))[0]
    # Only proceed with normalization if land pixels exist in the dataset
    if len(land_indices) > 0:  # Only normalize if there are land pixels
        hand_norm_reshaped[land_indices] = scaler.fit_transform(hand_norm_reshaped[land_indices])
    hand_norm = hand_norm_reshaped.reshape(hand.shape)
    
    # Apply land mask and ensure only positive upstream area values are considered
    upstream_norm = np.where((upstream_area > 0) & land_mask, upstream_area, 0)  # Keep only positive upstream area values on land, set everything else to 0
    # Normalize the transformed upstream area values (land areas only)
    upstream_norm_reshaped = upstream_norm.reshape(-1, 1)  # Reshape the 2D array to a column vector for the scaler function
    # Only proceed with normalization if land pixels exist in the dataset
    if len(land_indices) > 0:  # Check if there are any land pixels to normalize
        upstream_norm_reshaped[land_indices] = scaler.fit_transform(upstream_norm_reshaped[land_indices])  # Scale land values to 0-1 range
    # Reshape back to original dimensions
    upstream_norm = upstream_norm_reshaped.reshape(upstream_area.shape)  # Convert column vector back to original 2D array shape
    
    # Calculate flow convergence by counting unique flow directions in 3x3 neighborhood
    flow_direction = ndimage.generic_filter(flow_direction, lambda x: len(np.unique(x)), size=3)
    # Normalize the flow convergence values (land areas only)
    flow_direction_norm_reshaped = flow_direction.reshape(-1, 1)
    # Only proceed with normalization if land pixels exist in the dataset
    if len(land_indices) > 0:  # Only normalize if there are land pixels
        flow_direction_norm_reshaped[land_indices] = scaler.fit_transform(flow_direction_norm_reshaped[land_indices])
    flow_direction_norm = flow_direction_norm_reshaped.reshape(flow_direction.shape)
    

    river_width = np.where(land_mask, river_width, 0)
    # Normalize the river influence values (land areas only)
    river_width_norm_reshaped = river_width.reshape(-1, 1)
    # Only proceed with normalization if land pixels exist in the dataset
    if len(land_indices) > 0:  # Only normalize if there are land pixels
        river_width_norm_reshaped[land_indices] = scaler.fit_transform(river_width_norm_reshaped[land_indices])
    river_width_norm = river_width_norm_reshaped.reshape(river_width.shape)
    
    # Calculate final risk score using ML-derived weights for each factor
    flood_risk = np.zeros_like(elevation)
    # Only calculate risk for land areas
    flood_risk = np.where(land_mask, 
        0.364475 * elevation_norm +             
        0.302217 * hand_norm +                
        0.161996 * upstream_norm +             
        0.165351 * flow_direction_norm +    
        0.005961 * river_width_norm,      # River width has smaller impact
        np.nan)  # Use NaN for ocean areas

    flood_risk = np.where(~np.isnan(flood_risk), np.power(flood_risk, 0.1), np.nan)
    
    # Normalize the final risk scores to 0-1 range (land areas only)
    flood_risk_reshaped = flood_risk.reshape(-1, 1)
    valid_indices = np.where(~np.isnan(flood_risk_reshaped.reshape(-1)))[0]
    if len(valid_indices) > 0:  # Only normalize if there are valid pixels
        flood_risk_reshaped[valid_indices] = scaler.fit_transform(flood_risk_reshaped[valid_indices])
    flood_risk = flood_risk_reshaped.reshape(flood_risk.shape)
    
    # Return the calculated flood risk array
    return flood_risk

def visualize_flood_risk(risk_array, output_path, transform, crs):
    """Visualize the flood risk map and save to file with ocean areas colored black."""
    # Create a figure and get the current axes
    fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
    
    # Define a set of distinct colors for small ranges
    colors = [
        '#000080',  # Navy (0.0-0.1)
        '#0000FF',  # Blue (0.1-0.2)
        '#00FFFF',  # Cyan (0.2-0.3)
        '#008000',  # Green (0.3-0.4)
        '#ADFF2F',  # GreenYellow (0.4-0.5)
        '#FFFF00',  # Yellow (0.5-0.6)
        '#FFA500',  # Orange (0.6-0.7)
        '#FF0000',  # Red (0.7-0.8)
        '#800000',  # Maroon (0.8-0.9)
        '#FF00FF',  # Magenta (0.9-1.0)
    ]
    
    # Create custom colormap with 10 distinct bands
    cmap = LinearSegmentedColormap.from_list('high_contrast', colors, N=10)
    cmap.set_bad('black')  # Set NaN values to black
    
    # Create boundaries for distinct color bands
    bounds = np.linspace(0, 1, 11)  # 11 boundaries for 10 distinct ranges
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Display the risk array with the custom colormap
    img = ax.imshow(risk_array, cmap=cmap, norm=norm)
    
    # Add a colorbar with tick marks at the boundaries
    cbar = plt.colorbar(img, ax=ax, ticks=bounds)
    cbar.set_label('Flood Hazard Index (0-1)')
    
    # Format the tick labels to show ranges
    tick_labels = [f"{bounds[i]:.1f}-{bounds[i+1]:.1f}" for i in range(len(bounds)-1)]
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
    
    # Open the river width file and read its data
    with rasterio.open(os.path.join('river_width_final.tif')) as src:
        # Read the first band of data
        river_width = src.read(1)
    
    print("Calculating flood risk with ocean masking...")
    flood_risk = calculate_flood_risk(elevation, flow_direction, hand, upstream_area, river_width)
    
    print("Visualizing results")
    risk_geotiff = visualize_flood_risk(flood_risk, os.path.join(OUTPUT_DIR, 'flood_risk.tif'), transform, crs)
    
    print(f"\nFlood risk assessment complete! Results saved to {OUTPUT_DIR}/")

# Check if this script is being run directly (not imported)
if __name__ == "__main__":
    main()
