import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation


grib_file = 'data.grib'  # Replace with your GRIB file path
ds = xr.open_dataset(grib_file, engine='cfgrib')


print(ds)  


u = ds['u']  
v = ds['v']  


if 'time' in ds.dims:
    times = ds['time']
else:
    raise ValueError("No time dimension found in the dataset.")

fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
plt.tight_layout()
ax.coastlines()
ax.gridlines()
robust_marker = None
center_marker = None

# Function to calculate vorticity
def calculate_vorticity(u, v, dx, dy):
    """
    Compute the relative vorticity from wind components u and v.
    """
    dudx = (u.shift(longitude=-1) - u) / dx
    dvdy = (v.shift(latitude=-1) - v) / dy
    return dvdy - dudx  
def calculate_robustness(u, v, high_vorticity_idx, lats, lons, threshold):
    """
    Compute robustness for high-vorticity points using a merge tree at one level.
    
    Parameters:
        u (xr.DataArray): U-component of the wind field.
        v (xr.DataArray): V-component of the wind field.
        high_vorticity_idx (tuple): Indices of high-vorticity regions.
        lats (np.array): Latitude values of the grid.
        lons (np.array): Longitude values of the grid.
        threshold (float): Robustness threshold for filtering points.
        
    Returns:
        filtered_points (list): List of (lat, lon) for points with robustness > threshold.
    """
    # Calculate wind speed magnitude as the scalar field
    magnitude = np.sqrt(u**2 + v**2)
    
    # Smooth the scalar field to remove small-scale noise
    magnitude_smoothed = gaussian_filter(magnitude, sigma=1.0)

    # Build a spatial tree for efficient neighbor searches
    lat_lon_grid = np.array([(lat, lon) for lat in lats for lon in lons])
    kd_tree = KDTree(lat_lon_grid)
    
    filtered_points = []
    
    # Iterate through high-vorticity points
    for idx in zip(*high_vorticity_idx):
        lat_center = lats[idx[0]]
        lon_center = lons[idx[1]]
        
        # Query the nearest neighbors within a fixed radius
        neighbors_idx = kd_tree.query_ball_point([lat_center, lon_center], r=0.5)
        
        # Compute robustness as the difference between peak and lowest neighbor
        values = magnitude_smoothed.flatten()[neighbors_idx]
        peak = magnitude_smoothed[idx[0], idx[1]]
        robustness = peak - np.min(values)
        
        # Filter based on robustness threshold
        if robustness > threshold:
            filtered_points.append((lat_center, lon_center))
    
    return filtered_points

def update(frame):
    global center_marker
    global robust_marker
    time_idx = frame
    time = times[time_idx]

    # Extract and flip the data along the latitude axis
    u_time = u.isel(time=time_idx).isel(latitude=slice(None, None, -1))
    v_time = v.isel(time=time_idx).isel(latitude=slice(None, None, -1))

    # Calculate wind magnitude
    magnitude = np.sqrt(u_time**2 + v_time**2)

    vorticityThreshold = 0.0001

    dx = np.gradient(ds['longitude'].values)[0] * 111000  # Convert degrees to meters
    dy = np.gradient(ds['latitude'].values)[0] * 111000  # Convert degrees to meters

    high_vorticity_idx = np.where(calculate_vorticity(u_time, v_time, dx,dy) > vorticityThreshold)

    # Extract coordinates of low-wind regions
    high_vorticity_lats = ds['latitude'].values[::-1][high_vorticity_idx[0]]
    high_vorticity_lons = ds['longitude'].values[high_vorticity_idx[1]]

    # Calculate robustness for high-vorticity points
    filtered_points = calculate_robustness(
        u_time.values, v_time.values,
        high_vorticity_idx, 
        ds['latitude'].values[::-1], ds['longitude'].values, 
        threshold=6  # Adjust robustness threshold as needed
    )

    # Normalize vectors for RGB encoding
    u_normalized = u_time / magnitude
    v_normalized = v_time / magnitude
    R = (u_normalized + 1) / 2 * magnitude / np.max(magnitude)  # Map from [-1, 1] to [0, 1]
    G = (v_normalized + 1) / 2 * magnitude / np.max(magnitude) # Map from [-1, 1] to [0, 1]
    B = magnitude / np.max(magnitude) *magnitude / np.max(magnitude) # Normalize magnitude for brightness

    # Create RGB image
    rgb_image = np.dstack((R, G, B))

    # Plot the normal map
    ax.clear()
    ax.coastlines()
    ax.gridlines()
    ax.imshow(
        rgb_image, 
        origin='lower', 
        extent=[ds['longitude'].min(), ds['longitude'].max(), 
                ds['latitude'].min(), ds['latitude'].max()],
        transform=ccrs.PlateCarree()
    )

    # Mark high_vorticity points
    if center_marker:
        center_marker[0].remove()
    center_marker = ax.plot(
        high_vorticity_lons, high_vorticity_lats, 'bo', markersize=4, label='Cyclone Center'
    )
    # Mark topologically_robust points
    if filtered_points:
        lat_filtered, lon_filtered = zip(*filtered_points)
        ax.plot(lon_filtered, lat_filtered, 'ro', markersize=4, label='Robust Points')
    
    ax.legend(loc='upper right')
    ax.set_title(f'Normal Map and Low Wind Regions at {time.values}')


ani = FuncAnimation(fig, update, frames=len(times), repeat=True)


plt.show()
