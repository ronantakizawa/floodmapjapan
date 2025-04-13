# floodmapjapan
A Smart Flood Hazard Map of Japan built with GIS and Machine Learning.
![floodmapjapan](https://github.com/user-attachments/assets/897e6ecd-ccdc-49ef-ba02-805b66073939)


## Features

Data-driven flood risk assessment using Random Forest Regressor
Processing of multiple geographical features (elevation, drainage, river width, etc.)
Proper handling of ocean masking for focused land-based analysis
Interactive and static map visualizations
Comparison with official flood hazard maps from Japanese government sources

## Data Sources
The project uses two primary datasets:

Hydromap of Japan (GeoTIFF files from the University of Tokyo):
Water directional flow
Elevation data
Upstream drainage area (flow accumulation)
River channel width
HAND (Height Above Nearest Drainage)


Historical flood records (CSV from National Research Institute for Earth Science and Disaster Prevention):

Infrastructure damage
Transportation impacts
Water/flood impacts

## Machine Learning Approach
The project employs a Random Forest Regressor model to derive optimal weights for various flood risk factors:
Derived weights for flood risk factors:
elevation: 0.896
hand: 0.044
upstream: 0.048
flow_conv: 0.002
river: 0.009

These weights are used in the calculate_flood_risk function to create a normalized risk score (0-1) for each location in Japan. 

## Technical Implementation
The main algorithm processes GeoTIFF files containing geographical data and applies the ML-derived weights to calculate flood risk scores. Ocean areas are properly masked to focus the analysis on land areas.
Key components:

Raster data processing with rasterio and rioxarray
Normalization and spatial transformations
Interactive map creation with folium
Advanced visualization techniques

## Results
The final flood risk map identifies several high-risk zones in Japan:

Tokyo and Kanto Plain: Densely populated, highly urbanized areas with extensive low-lying floodplains
Osaka/Kyoto/Kobe region: Coastal lowlands and historical flood plains
Hokkaido: Large river basins with flat terrain and cold-climate runoff characteristics

## Future Work
Find larger and more detailed datasets to further refine flood risk heuristics
Implement additional variables such as soil permeability, urbanization levels, and climate change projections
Develop dynamic risk assessment models accounting for seasonal variations and extreme weather events
