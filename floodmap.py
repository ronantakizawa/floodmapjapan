# Import the os module for file and directory operations
import os
# Import numpy for numerical operations and array handling
import numpy as np
# Import rasterio for working with geospatial raster data (GeoTIFF files)
import rasterio
# Import matplotlib.pyplot for creating visualizations
import matplotlib.pyplot as plt

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
    elevation_safe = 100  # "Safe" elevation threshold in meters
    elevation_high_risk = 10  # High risk threshold in meters
    
    # Create a more nuanced risk curve for elevation (only for land areas)
    elevation_norm = np.zeros_like(elevation)
    # High risk for very low elevations (below high_risk threshold)
    elevation_norm = np.where((elevation <= elevation_high_risk) & land_mask, 1.0, elevation_norm)
    # Graduated risk for middle elevations
    middle_mask = (elevation > elevation_high_risk) & (elevation < elevation_safe) & land_mask
    elevation_norm = np.where(
        middle_mask,
        1 - ((elevation - elevation_high_risk) / (elevation_safe - elevation_high_risk)),
        elevation_norm
    )
    
    # Apply square root transformation to reduce extremes (land areas only)
    elevation_norm = np.where(land_mask, elevation_norm ** 0.5, elevation_norm)
    
    # Replace negative HAND values with 0 (HAND should be non-negative) and apply land mask
    hand_norm = np.where((hand > 0) & land_mask, hand, 0)
    # Apply exponential decay transformation based on ML analysis
    hand_norm = np.where(land_mask, np.exp(-0.15 * hand_norm), 0)
    # Normalize the transformed HAND values (land areas only)
    hand_norm_reshaped = hand_norm.reshape(-1, 1)
    land_indices = np.where(land_mask.reshape(-1))[0]
    if len(land_indices) > 0:  # Only normalize if there are land pixels
        hand_norm_reshaped[land_indices] = scaler.fit_transform(hand_norm_reshaped[land_indices])
    hand_norm = hand_norm_reshaped.reshape(hand.shape)
    # Apply square root transformation
    hand_norm = np.where(land_mask, hand_norm ** 0.5, hand_norm)
    
    # Apply log transformation to upstream area (addresses skewed distribution)
    upstream_norm = np.where((upstream_area > 0) & land_mask, np.log1p(upstream_area), 0)
    # Normalize the transformed upstream area values (land areas only)
    upstream_norm_reshaped = upstream_norm.reshape(-1, 1)
    if len(land_indices) > 0:  # Only normalize if there are land pixels
        upstream_norm_reshaped[land_indices] = scaler.fit_transform(upstream_norm_reshaped[land_indices])
    upstream_norm = upstream_norm_reshaped.reshape(upstream_area.shape)
    # Apply square root transformation
    upstream_norm = np.where(land_mask, upstream_norm ** 0.5, upstream_norm)
    
    # Calculate flow convergence by counting unique flow directions in 3x3 neighborhood
    flow_convergence = ndimage.generic_filter(flow_direction, lambda x: len(np.unique(x)), size=3)
    # Normalize the flow convergence values (land areas only)
    flow_convergence_norm_reshaped = flow_convergence.reshape(-1, 1)
    if len(land_indices) > 0:  # Only normalize if there are land pixels
        flow_convergence_norm_reshaped[land_indices] = scaler.fit_transform(flow_convergence_norm_reshaped[land_indices])
    flow_convergence_norm = flow_convergence_norm_reshaped.reshape(flow_direction.shape)
    # Apply square root transformation
    flow_convergence_norm = np.where(land_mask, flow_convergence_norm ** 0.5, flow_convergence_norm)
    
    # Apply Gaussian filter to river width to model the influence on nearby areas
    river_influence = ndimage.gaussian_filter(np.where(land_mask, river_width, 0), sigma=7)
    # Normalize the river influence values (land areas only)
    river_influence_norm_reshaped = river_influence.reshape(-1, 1)
    if len(land_indices) > 0:  # Only normalize if there are land pixels
        river_influence_norm_reshaped[land_indices] = scaler.fit_transform(river_influence_norm_reshaped[land_indices])
    river_influence_norm = river_influence_norm_reshaped.reshape(river_width.shape)
    # Apply square root transformation
    river_influence_norm = np.where(land_mask, river_influence_norm ** 0.5, river_influence_norm)
    
    # Calculate final risk score using ML-derived weights for each factor
    flood_risk = np.zeros_like(elevation)
    # Only calculate risk for land areas
    flood_risk = np.where(land_mask, 
        0.509 * elevation_norm +             
        0 * hand_norm +                
        0.491 * upstream_norm +             
        0 * flow_convergence_norm +    
        0 * river_influence_norm,      # River width has smaller impact
        np.nan)  # Use NaN for ocean areas
    
    # Apply logarithmic scaling to compress the range of values (land areas only)
    flood_risk = np.where(~np.isnan(flood_risk), np.log1p(flood_risk) / np.log1p(1), np.nan)
    
    # Normalize the final risk scores to 0-1 range (land areas only)
    flood_risk_reshaped = flood_risk.reshape(-1, 1)
    valid_indices = np.where(~np.isnan(flood_risk_reshaped.reshape(-1)))[0]
    if len(valid_indices) > 0:  # Only normalize if there are valid pixels
        flood_risk_reshaped[valid_indices] = scaler.fit_transform(flood_risk_reshaped[valid_indices])
    flood_risk = flood_risk_reshaped.reshape(flood_risk.shape)
    
    # Return the calculated flood risk array
    return flood_risk

def visualize_flood_risk(risk_array, output_path, transform, crs):
    """Visualize the flood risk map and save to file with ocean areas colored white."""
    # Create a figure and get the current axes
    fig, ax = plt.figure(figsize=(12, 10)), plt.gca()
    
    # Create a custom colormap that has white for NaN values
    cmap = plt.cm.RdYlBu_r.copy()
    cmap.set_bad('black')  # Set NaN values to Black
    
    # Display the risk array as an image with specific color mapping
    img = ax.imshow(risk_array, cmap=cmap, vmin=0, vmax=1)
    
    # Add a colorbar to the plot to show the risk scale
    cbar = plt.colorbar(img, ax=ax)
    # Add a label to the colorbar
    cbar.set_label('Flood Risk Index (0-1)')
    
    # Add a title to the plot
    plt.title('Flood Risk Assessment for Japan', fontsize=16)
    # Turn off the axis labels and ticks
    plt.axis('off')
    
    # Save the visualization as a PNG file
    plt.savefig(os.path.join(OUTPUT_DIR, 'flood_risk_visualization.png'), dpi=300, bbox_inches='tight')
    
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
