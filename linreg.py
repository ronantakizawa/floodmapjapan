import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import rasterio
from rasterio.transform import rowcol

def extract_features_at_flood_points(flood_df, elevation, flow_direction, hand, upstream_area, river_width, transform):
    """Extract feature values at each flood location."""
    # Calculate flow convergence
    from scipy import ndimage
    flow_conv = ndimage.generic_filter(flow_direction, lambda x: len(np.unique(x)), size=3)
    
    # Create dataframe for valid points
    valid_data = []
    
    for _, row in flood_df.iterrows():
        try:
            # Get pixel coordinates for this lat/lon
            row_idx, col_idx = rowcol(transform, row['longitude'], row['latitude'])
            
            # Check if coordinates are within bounds and on land
            if (0 <= row_idx < elevation.shape[0] and 
                0 <= col_idx < elevation.shape[1] and
                elevation[row_idx, col_idx] > 0):  # Land mask
                
                # Extract feature values
                elev = elevation[row_idx, col_idx]
                hand_val = max(0, hand[row_idx, col_idx])  # Replace negative with 0
                upstream = upstream_area[row_idx, col_idx]
                flow_convergence = flow_conv[row_idx, col_idx]
                # Extract river width, handle possible NaN values
                river_w = river_width[row_idx, col_idx]
                if np.isnan(river_w):
                    river_w = 0  # Set NaN river width to 0 (no river)
                flood_area = row['flood_area_km2']
                
                valid_data.append({
                    'elevation': -elev,  # Negate elevation (lower = higher risk)
                    'hand': hand_val,
                    'upstream_area': upstream,
                    'flow_convergence': flow_convergence,
                    'river_width': river_w,
                    'flood_area_km2': flood_area
                })
        except Exception:
            continue
    
    return pd.DataFrame(valid_data)

def train_models_and_get_weights(feature_df):
    """Train linear regression model and output weights with a visualization."""
    # Prepare features and target
    X = feature_df[['elevation', 'hand', 'upstream_area', 'flow_convergence', 'river_width']]
    y = feature_df['flood_area_km2']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y, test_size=0.2, random_state=42
    )
    
    # Initialize model
    model = LinearRegression()
    
    print("\nMODEL WEIGHTS AND PERFORMANCE METRICS:")
    print("--------------------------------------")
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled_df, y, cv=5, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {np.mean(cv_scores):.4f}")
    
    # Test set performance
    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Test set R²: {test_r2:.4f}")
    print(f"Test set RMSE: {test_rmse:.4f}")
    
    # Extract coefficients
    coefs = model.coef_
    
    # Get normalized weights (absolute values, sum to 1)
    abs_coefs = np.abs(coefs)
    normalized_weights = abs_coefs / np.sum(abs_coefs)
    
    # Display weights
    print(f"\nFeature coefficients:")
    for feature, weight, norm_weight in zip(X.columns, coefs, normalized_weights):
        print(f"  {feature}: raw={weight:.4f}, normalized={norm_weight:.4f}")
    
    # Print feature importance sorted from most to least important
    sorted_indices = np.argsort(-normalized_weights)
    print("\nFeature Importance Ranking:")
    print("--------------------------")
    for i, idx in enumerate(sorted_indices):
        feature = X.columns[idx]
        weight = normalized_weights[idx]
        print(f"{i+1}. {feature}: {weight:.4f}")
    
    # Create and save a bar chart of the weights
    plt.figure(figsize=(12, 8))
    
    # Sort features by importance for the chart
    sorted_features = [X.columns[i] for i in sorted_indices]
    sorted_weights = normalized_weights[sorted_indices]
    
    # Create bar chart
    bars = plt.bar(sorted_features, sorted_weights, color='steelblue')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontsize=12)
    
    # Customize chart
    plt.title('Feature Importance Weights for Flood Hazard Model', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Normalized Weight', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.ylim(0, max(normalized_weights) * 1.15)  # Add 15% padding at the top
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('feature_weights_chart.png', dpi=300, bbox_inches='tight')
    print("\nFeature importance chart saved as 'feature_weights_chart.png'")
    
    return sorted_features, sorted_weights

def main():
    """Load data, extract features, and determine weights."""
    print("Loading data...")
    
    # Load flood data
    flood_df = pd.read_csv('japan_flood_data_with_coords.csv', encoding='utf-8')
    flood_df = flood_df.dropna(subset=['latitude', 'longitude', 'flood_area_km2'])
    
    # Load raster data
    with rasterio.open('flow_direction_final.tif') as src:
        flow_direction = src.read(1)
        transform = src.transform
    
    with rasterio.open('elevation_final.tif') as src:
        elevation = src.read(1)
    
    with rasterio.open('hand_final.tif') as src:
        hand = src.read(1)
    
    with rasterio.open('upstream_area_final.tif') as src:
        upstream_area = src.read(1)
    
    with rasterio.open('river_width_final.tif') as src:
        river_width = src.read(1)
    
    print("Extracting features at flood locations...")
    feature_df = extract_features_at_flood_points(
        flood_df, elevation, flow_direction, hand, upstream_area, river_width, transform
    )
    
    print(f"Found {len(feature_df)} valid locations with features and flood area data")
    
    # Check for any NaN values in the features
    print("\nFeature statistics:")
    print("-----------------")
    print(feature_df.describe())
    
    # Check for NaN values
    nan_counts = feature_df.isna().sum()
    print("\nNaN count per feature:")
    print(nan_counts)
    
    # Remove any remaining NaN values
    feature_df = feature_df.dropna()
    print(f"After dropping NaNs: {len(feature_df)} locations remain")
    
    # Train model and get weights with visualization
    sorted_features, sorted_weights = train_models_and_get_weights(feature_df)
    
    # Now update the flood hazard model with these weights
    print("\nTo use these weights in your flood hazard model, update the following in calculate_flood_risk():")
    print("```python")
    print("# Instead of equal weights:")
    print("# risk_values = X.mean(axis=1)")
    print("")
    print("# Use the learned weights:")
    print("feature_weights = {")
    for feature, weight in zip(sorted_features, sorted_weights):
        print(f"    '{feature}': {weight:.4f},")
    print("}")
    print("# Apply weights in order matching your feature matrix")
    print("weights = np.array([feature_weights[feature] for feature in ['elevation', 'hand', 'upstream_area', 'flow_convergence', 'river_width']])")
    print("risk_values = np.sum(X * weights, axis=1)")
    print("```")

if __name__ == "__main__":
    main()