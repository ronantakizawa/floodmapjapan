# floodmapjapan
**A Smart Flood Hazard Map of Japan built with GIS and Random Forest Regression** 
![floodhazardmapjapan](https://github.com/user-attachments/assets/cdcc3ec3-9d6e-489b-a61a-0314a4acaf42)



## Features

Data-driven flood risk assessment using Random Forest Regressor
Processing of multiple geographical features (elevation, drainage, river width, etc.)
Proper handling of ocean masking for focused land-based analysis
Interactive and static map visualizations
Comparison with official flood hazard maps from Japanese government sources

## Data Sources
The project uses two primary datasets:

#1 A hydromap of Japan (University of Tokyo) with features such as:

* **Water Directional Flow:** The direction in which surface water is expected to move based on terrain and hydrological modeling (Measured in angular degrees from 0–360°).
* **Elevation:** The height of the land surface above sea level (m).
* **Upstream Drainage Area (Flow Accumulation Area):** The total land area that drains into a specific point on the landscape, indicating how much water can potentially flow through it (km²),
* **River Channel Width:** The estimated horizontal width of a river at a given location, which affects its capacity to carry floodwaters (m).
* **HAND (Height Above Nearest Drainage):** A terrain metric that represents how high a given point is above the nearest stream or drainage line, helping identify flood-prone lowlands (m).

#2 GIS data from the Japanese National Research Institute for Earth Science and Disaster Prevention, covering floods in Japan from 1961 to 2008, with information such as:
* Infrastructure damage
* Transportation Damage
* Water / Flood Impacts (Flood area, flood depth)

## Machine Learning Approach
The project employs a Random Forest Regressor model to derive optimal weights for various flood risk factors:
Derived weights for flood risk factors:

* elevation: 0.189
* hand: 0.011
* upstream: 0.760
* flow_conv: 0.000
* river_width: 0.039

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
