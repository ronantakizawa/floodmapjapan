# floodmapjapan
**A Smart Flood Hazard Map of Japan built with GIS and Random Forest Regression** 

Leave a star if you enjoyed it! ⭐️

Full Detailed Analysis: https://medium.com/@ronantech/building-a-smart-flood-map-of-japan-using-gis-and-machine-learning-3303efff9957

![floodhazardmapjapan](https://github.com/user-attachments/assets/034ae90e-4b4d-481d-aef6-65e66fd96105)




## Features

Data-driven flood risk assessment using Random Forest Regressor
Processing of multiple geographical features (elevation, drainage, river width, etc.)
Proper handling of ocean masking for focused land-based analysis
Interactive and static map visualizations
Comparison with official flood hazard maps from Japanese government sources

## Datasets
The project uses two primary datasets:

#1 A hydromap of Japan (University of Tokyo) with features such as:

* **Water Directional Flow:** The direction in which surface water is expected to move based on terrain and hydrological modeling (Measured in angular degrees from 0–360°).
* **Elevation:** The height of the land surface above sea level (m).
* **Upstream Drainage Area (Flow Accumulation Area):** The total land area that drains into a specific point on the landscape, indicating how much water can potentially flow through it (km²),
* **River Channel Width:** The estimated horizontal width of a river at a given location, which affects its capacity to carry floodwaters (m).
* **HAND (Height Above Nearest Drainage):** A terrain metric that represents how high a given point is above the nearest stream or drainage line, helping identify flood-prone lowlands (m).

#2 GIS data from the Japanese National Research Institute for Earth Science and Disaster Prevention, covering floods in Japan from 1961 to 2008, with information such as:
* Flood date, time, location, area


## Technical Implementation
The main algorithm processes GeoTIFF files containing geographical data and applies the ML-derived weights to calculate flood risk scores. Ocean areas are properly masked to focus the analysis on land areas.

## Machine Learning Approach
The project employs a Random Forest Regressor model to derive optimal weights for various flood risk factors. Weights are derived by getting the elevation, HAND, upstream drainage area, river direction, and river width values from flood locations in Japan from 1961-2008, and running a Random Forest Regressor model with flood area as the target variable to see which variables contribute to large flood areas the most. 

Here are the weights after running the model:

* elevation: 0.364475
* hand: 0.302217
* upstream: 0.161996
* flow_conv:0.165351
* river_width: 0.005961

These weights are used in the calculate_flood_risk function to create a normalized risk score (0-1) for each location in Japan. 

![feature_importance](https://github.com/user-attachments/assets/c3125fd6-7019-4bd8-ab10-daf947bc9e14)

## Results
The final flood risk map identifies several high-risk zones in Japan:

Tokyo and Kanto Plain: Densely populated, highly urbanized areas with extensive low-lying floodplains
Osaka/Kyoto/Kobe region: Coastal lowlands and historical flood plains
Hokkaido: Large river basins with flat terrain and cold-climate runoff characteristics

## Future Work
Find larger and more detailed datasets to further refine flood risk heuristics
Implement additional variables such as soil permeability, urbanization levels, and climate change projections
Develop dynamic risk assessment models accounting for seasonal variations and extreme weather events
