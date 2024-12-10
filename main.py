import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation

# 1. Load GRIB data using xarray with cfgrib engine
grib_file = '77df4cfa0cc7db1727b16c3ef2cb92e5.grib'  # Replace with your GRIB file path
ds = xr.open_dataset(grib_file, engine='cfgrib')

# 2. Inspect the available variables in the dataset
print(ds)  # This will print the dataset structure so you can identify U and V components

# 3. Extract U and V components (Adjust the variable names as necessary based on your dataset)
u = ds['u']  # Replace with the actual name in your dataset
v = ds['v']  # Replace with the actual name in your dataset

# 4. Check if a time dimension exists
if 'time' in ds.dims:
    times = ds['time']
    print("Time dimension found:", times)
else:
    raise ValueError("No time dimension found in the dataset.")

# 5. Create a figure for the animation
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines()
ax.gridlines()
quiver = None
center_marker = None

# Function to calculate vorticity
def calculate_vorticity(u, v, dx, dy):
    """
    Compute the relative vorticity from wind components u and v.
    """
    dudx = (u.shift(longitude=-1) - u) / dx
    dvdy = (v.shift(latitude=-1) - v) / dy
    return dvdy - dudx

# Function to update the frame
def update(frame):
    global quiver, center_marker
    time_idx = frame
    time = times[time_idx]

    # Select data for the current time step
    u_time = u.isel(time=time_idx)
    v_time = v.isel(time=time_idx)

    # Compute vorticity
    dx = np.gradient(ds['longitude'].values)[0] * 111000  # Approximate meters per degree longitude
    dy = np.gradient(ds['latitude'].values)[0] * 111000  # Approximate meters per degree latitude
    vorticity = calculate_vorticity(u_time, v_time, dx, dy)

    # Locate the cyclone center (maximum vorticity)
    max_vort_idx = np.unravel_index(np.argmax(vorticity.values), vorticity.shape)
    cyclone_lat = ds['latitude'].values[max_vort_idx[0]]
    cyclone_lon = ds['longitude'].values[max_vort_idx[1]]

    # Compute magnitude and normalize for vector field
    magnitude = np.sqrt(u_time**2 + v_time**2)
    magnitude_threshold = 10
    mask = magnitude >= magnitude_threshold
    u_normalized = u_time * mask / magnitude
    v_normalized = v_time * mask / magnitude

    # Clear previous quiver and marker
    if quiver:
        quiver.remove()
    if center_marker:
        center_marker[0].remove()  # Access the first (and only) marker in the list

    # Plot the vector field
    quiver = ax.quiver(ds['longitude'][::5], ds['latitude'][::5], 
                       u_normalized[::5, ::5], v_normalized[::5, ::5], 
                       magnitude[::5, ::5] * mask[::5, ::5], scale=50, width=0.002, 
                       cmap='viridis', pivot='middle')

    # Mark the cyclone center
    center_marker = ax.plot(cyclone_lon, cyclone_lat, 'ro', markersize=8, label='Cyclone Center')
    ax.legend(loc='upper right')
    ax.set_title(f'Vector Field and Cyclone Center at {time.values}')

# Create the animation
ani = FuncAnimation(fig, update, frames=len(times), repeat=True)

# Show the animation
plt.show()
