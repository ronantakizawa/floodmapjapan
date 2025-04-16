import pandas as pd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set paths
csv_path = 'japan_flood_data_with_coords.csv'
elevation_tif = 'elevation_final.tif'
hand_tif = 'hand_final.tif'
flow_direction_tif = 'flow_direction_final.tif'
river_width_tif = 'river_width_final.tif'
upstream_tif = 'upstream_area_final.tif'  # Added upstream area raster

# Read the CSV file with flooding data
flood_data = pd.read_csv(csv_path, low_memory=False)


# Function to extract raster values at point locations with better error handling
def extract_raster_values(raster_path, points_df, buffer_size=3):
    """
    Extract values from a raster at the locations of points in a DataFrame.
    Uses a small buffer around each point to handle coordinate precision issues.
    """
    values = []
    
    with rasterio.open(raster_path) as src:
        # Print raster info
        print(f"\nRaster: {raster_path}")
        print(f"CRS: {src.crs}")
        print(f"Bounds: {src.bounds}")
        print(f"Shape: {src.shape}")
        
        # Create transformer from WGS84 to raster CRS if they're different
        if src.crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)
        
        for idx, row in points_df.iterrows():
            try:
                if src.crs != CRS.from_epsg(4326):
                    # Transform coordinates if necessary
                    x, y = transformer.transform(row['longitude'], row['latitude'])
                else:
                    x, y = row['longitude'], row['latitude']
                
                # Convert geographic coordinates to pixel coordinates
                py, px = src.index(x, y)
                
                # Take the mean of a small window around the point to account for precision issues
                window = src.read(1, window=((py-buffer_size, py+buffer_size+1), 
                                           (px-buffer_size, px+buffer_size+1)))
                
                # Calculate the mean, ignoring NoData values
                nodata = src.nodata if src.nodata is not None else np.nan
                valid_values = window[window != nodata]
                
                if len(valid_values) > 0:
                    values.append(float(np.mean(valid_values)))
                else:
                    values.append(np.nan)
                    
            except Exception as e:
                if idx % 1000 == 0:
                    print(f"Error at index {idx}: {e}")
                values.append(np.nan)
                
    return values

# Extract values from TIF files with progress indicators
print("\nExtracting values from rasters...")
for i, (name, path) in enumerate([('elevation', elevation_tif), ('hand', hand_tif), 
                                ('flow_direction', flow_direction_tif), ('river_width', river_width_tif),
                                ('upstream_area', upstream_tif)]):  # Added upstream area
    print(f"Processing {name} ({i+1}/5)...")  # Updated count
    flood_data[name] = extract_raster_values(path, flood_data)
    valid_count = flood_data[name].notna().sum()
    print(f"  - Valid values extracted: {valid_count} ({valid_count/len(flood_data)*100:.1f}%)")

# Clean the data - remove rows with missing values
clean_data = flood_data.dropna(subset=['elevation', 'hand', 'flow_direction', 'river_width', 'upstream_area', 'flood_area_km2'])
print(f"\nNumber of samples after cleaning: {len(clean_data)} ({len(clean_data)/len(flood_data)*100:.1f}% of original)")

# Log transform the target if it's skewed
if clean_data['flood_area_km2'].skew() > 1:
    print("Target variable is skewed, applying log transformation...")
    clean_data['flood_area_km2_log'] = np.log1p(clean_data['flood_area_km2'])
    target = 'flood_area_km2_log'
    print(f"Skewness before: {clean_data['flood_area_km2'].skew():.2f}, after: {clean_data['flood_area_km2_log'].skew():.2f}")
else:
    target = 'flood_area_km2'

# Log transform the upstream area if it's skewed
if clean_data['upstream_area'].skew() > 1:
    print("Upstream area is skewed, applying log transformation...")
    clean_data['upstream_area_log'] = np.log1p(clean_data['upstream_area'])
    upstream_feature = 'upstream_area_log'
    print(f"Upstream area skewness before: {clean_data['upstream_area'].skew():.2f}, after: {clean_data['upstream_area_log'].skew():.2f}")
else:
    upstream_feature = 'upstream_area'

# Define features including upstream area
features = ['elevation', 'hand', 'flow_direction', 'river_width', upstream_feature]

# Examine feature relationships
print("\nFeature correlation with target:")
correlations = clean_data[features + [target]].corr()[target].sort_values(ascending=False)
print(correlations)

# Prepare data for modeling
X = clean_data[features]
y = clean_data[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)

# Create and train Random Forest with hyperparameter tuning
print("\nTraining Random Forest model with cross-validation...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                          cv=5, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# Make predictions
y_pred = best_rf.predict(X_test)

# Convert back from log scale if necessary
if target == 'flood_area_km2_log':
    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)
else:
    y_test_original = y_test
    y_pred_original = y_pred


# Feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Flood Area Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

