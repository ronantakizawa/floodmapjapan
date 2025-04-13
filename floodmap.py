# Import the os module for file and directory operations
import os
# Import numpy for numerical operations and array handling
import numpy as np
# Import rasterio for working with geospatial raster data (GeoTIFF files)
import rasterio
# Import matplotlib.pyplot for creating visualizations
import matplotlib.pyplot as plt
# Import folium for creating interactive maps
import folium
# Import HeatMap plugin from folium for creating heatmap visualizations
from folium.plugins import HeatMap
# Import rioxarray for working with raster data as xarray objects
import rioxarray as rxr
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = '.'
JAPAN_DEM = 'japan_dem_wgs84.tif'
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

def create_improved_interactive_map(risk_geotiff, output_html):
    """Create an enhanced interactive web map with the flood risk data."""
    # Open the risk GeoTIFF file using rioxarray
    risk_data = rxr.open_rasterio(risk_geotiff)
    
    # Extract coordinate data
    lats = risk_data.y.values
    lons = risk_data.x.values
    
    # Create a base map centered on Japan
    m = folium.Map(location=[36.2048, 138.2529], zoom_start=5, 
                  tiles='CartoDB dark_matter')  # Dark background as in your matplotlib visual
    
    # Add alternative basemap options
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(m)
    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)
    
    # Define sample rate to reduce data points for performance
    sample_rate = max(1, int(len(lats) / 500))  # Adaptive sampling
    
    # Initialize an empty list to store heatmap data points
    heat_data = []
    risk_array = risk_data.values[0]
    
    # Process data for heatmap - same thresholds as your original visualization
    for i in range(0, len(lats), sample_rate):
        for j in range(0, len(lons), sample_rate):
            risk_value = risk_array[i, j]
            if not np.isnan(risk_value) and risk_value > 0.25:
                heat_data.append([float(lats[i]), float(lons[j]), float(risk_value)])
    
    # Define the color gradient matching your matplotlib visualization
    # Using the same RdYlBu_r color scheme: blue -> yellow -> orange -> red
    gradient_dict = {
        "0.25": 'blue',      # Low risk
        "0.5": 'lime',       # Medium-low risk
        "0.65": 'yellow',    # Medium risk
        "0.8": 'orange',     # Medium-high risk
        "1.0": 'red'         # High risk
    }
    
    # Add the heatmap layer to the map
    HeatMap(heat_data,
            radius=12,           # Slightly larger radius for better visibility
            blur=10,             # Add blur for smoother appearance
            max_zoom=13,         # Maximum zoom level for the heatmap
            gradient=gradient_dict,  # Color gradient matching matplotlib
            min_opacity=0.6,     # Higher minimum opacity for better visibility
            name='Flood Risk').add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright').add_to(m)
    
    # Add scale bar
    folium.plugins.MeasureControl(position='bottomleft').add_to(m)
    
    # Add fullscreen option
    folium.plugins.Fullscreen().add_to(m)
    
    # Note: Removed the Search plugin that was causing the error
    
    # Improved HTML for title, description and legend
    title_html = '''
         <div style="position: fixed; 
                    top: 10px; left: 50px; width: 280px; 
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color: rgba(0, 0, 0, 0.7);
                    color: white; padding: 10px;
                    border-radius: 5px;">
         <h4 style="margin-top: 0; text-align: center;">Japan Flood Risk Assessment</h4>
         <p style="margin-bottom: 5px;">Risk Level Legend:</p>
         <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: red; margin-right: 5px;"></div>
            <span>High Risk (>0.8)</span>
         </div>
         <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: orange; margin-right: 5px;"></div>
            <span>Medium-High Risk (0.65-0.8)</span>
         </div>
         <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: yellow; margin-right: 5px;"></div>
            <span>Medium Risk (0.5-0.65)</span>
         </div>
         <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="width: 20px; height: 20px; background-color: lime; margin-right: 5px;"></div>
            <span>Medium-Low Risk (0.25-0.5)</span>
         </div>
         <div style="display: flex; align-items: center;">
            <div style="width: 20px; height: 20px; background-color: blue; margin-right: 5px;"></div>
            <span>Low Risk (<0.25)</span>
         </div>
         </div>
         '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add custom CSS to ensure the background is black as in your matplotlib
    custom_css = '''
    <style>
        body {
            background-color: black;
            color: white;
        }
    </style>
    '''
    m.get_root().html.add_child(folium.Element(custom_css))
    
    # Save the map
    try:
        m.save(output_html)
        print(f"Successfully saved enhanced interactive map to {output_html}")
    except Exception as e:
        print(f"Error saving interactive map: {e}")
        # Create a simpler fallback map if needed
        fallback_map = folium.Map(location=[36.2048, 138.2529], zoom_start=5, 
                               tiles='CartoDB dark_matter')
        fallback_map.save(output_html)
        print(f"Saved fallback map instead")
        
    return output_html

def calculate_at_risk_populations(risk_geotiff, population_data=None):
    """
    Calculate at-risk populations based on flood risk and population density.
    Excludes ocean areas from calculations.
    """
    # Open the risk GeoTIFF file
    with rasterio.open(risk_geotiff) as src:
        # Read the first band of the raster data
        risk_data = src.read(1)
        # Calculate the area of each pixel in square degrees
        pixel_area = abs(src.transform[0] * src.transform[4])
        
        # Convert square degrees to approximate square kilometers at Japan's latitude
        pixel_area_sqkm = pixel_area * 111 * 111 * np.cos(np.radians(36))  # 111 km per degree at equator, adjusted for latitude
        
        # Count pixels in high risk category (risk > 0.75), excluding NaN values
        high_risk_pixels = np.sum((risk_data > 0.75) & ~np.isnan(risk_data))
        # Count pixels in medium risk category (0.5 < risk <= 0.75), excluding NaN values
        medium_risk_pixels = np.sum((risk_data > 0.5) & (risk_data <= 0.75) & ~np.isnan(risk_data))
        # Count pixels in low risk category (0.25 < risk <= 0.5), excluding NaN values
        low_risk_pixels = np.sum((risk_data > 0.25) & (risk_data <= 0.5) & ~np.isnan(risk_data))
        
        # Calculate total area for high risk category
        high_risk_area = high_risk_pixels * pixel_area_sqkm
        # Calculate total area for medium risk category
        medium_risk_area = medium_risk_pixels * pixel_area_sqkm
        # Calculate total area for low risk category
        low_risk_area = low_risk_pixels * pixel_area_sqkm
        
        # Print results for high risk area
        print(f"High risk area (>0.75): {high_risk_area:.2f} sq km")
        # Print results for medium risk area
        print(f"Medium risk area (0.5-0.75): {medium_risk_area:.2f} sq km")
        # Print results for low risk area
        print(f"Low risk area (0.25-0.5): {low_risk_area:.2f} sq km")
        
        # Note: In a real application, population data would be used to estimate affected people

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
    
    print("Creating interactive map with ocean areas excluded...")
    interactive_map = create_improved_interactive_map(risk_geotiff, os.path.join(OUTPUT_DIR, 'flood_risk_map_japan_only.html'))
    
    print("Estimating at-risk areas (land only)...")
    calculate_at_risk_populations(risk_geotiff)
    
    print(f"\nFlood risk assessment complete! Results saved to {OUTPUT_DIR}/")
    print(f"Interactive map available at: {interactive_map}")

# Check if this script is being run directly (not imported)
if __name__ == "__main__":
    main()