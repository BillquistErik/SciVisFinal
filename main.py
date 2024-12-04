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

# Function to update the frame
def update(frame):
    global quiver
    time_idx = frame
    time = times[time_idx]

    # Select data for the current time step
    u_time = u.isel(time=time_idx)
    v_time = v.isel(time=time_idx)

    # Compute magnitude and normalize
    magnitude = np.sqrt(u_time**2 + v_time**2)
    magnitude_threshold = 10
    mask = magnitude >= magnitude_threshold
    u_normalized = u_time * mask / magnitude
    v_normalized = v_time * mask / magnitude

    # Clear previous quiver
    if quiver:
        quiver.remove()

    # Plot the vector field
    quiver = ax.quiver(ds['longitude'][::5], ds['latitude'][::5], 
                       u_normalized[::5, ::5], v_normalized[::5, ::5], 
                       magnitude[::5, ::5] * mask[::5, ::5], scale=50, width=0.002, 
                       cmap='viridis', pivot='middle')
    ax.set_title(f'Vector Field at {time.values}')

# Create the animation
ani = FuncAnimation(fig, update, frames=len(times), repeat=True)

# Show the animation
plt.show()
