import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs

# 1. Load GRIB data using xarray with cfgrib engine
grib_file = '5f5a242007810c1007258e05fa0ca38d.grib'  # Replace with your GRIB file path
ds = xr.open_dataset(grib_file, engine='cfgrib')

# 2. Inspect the available variables in the dataset
print(ds)  # This will print the dataset structure so you can identify U and V components

# 3. Extract U and V components (Adjust the variable names as necessary based on your dataset)
u = ds['u']  # Replace with the actual name in your dataset
v = ds['v']  # Replace with the actual name in your dataset

# 4. Optional: Check the dimensions of u and v
print("U shape:", u.shape)
print("V shape:", v.shape)

# 5. Compute the magnitude of the vector field
magnitude = np.sqrt(u**2 + v**2)

# 6. Define a threshold for magnitude
magnitude_threshold = 10  # Change this value to adjust the threshold

# 7. Filter the vectors based on magnitude
mask = magnitude >= magnitude_threshold

# 8. Normalize the vectors (scale the vectors according to their magnitudes)
u_normalized = u * mask / magnitude  # Normalize u components based on magnitude
v_normalized = v * mask / magnitude  # Normalize v components based on magnitude

# 9. Plot the vector field using a quiver plot
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

# Apply the mask to filter vectors and plot only the ones above the threshold
quiver = ax.quiver(ds['longitude'][::5], ds['latitude'][::5], 
                  u_normalized[::5, ::5], v_normalized[::5, ::5], 
                  magnitude[::5, ::5] * mask[::5, ::5], scale=50, width=0.002, 
                  cmap='viridis', pivot='middle')

# 10. Add coastlines, gridlines, and a colorbar
ax.coastlines()
ax.gridlines()
fig.colorbar(quiver, ax=ax, orientation='vertical', label='Magnitude')

# 11. Add a title
ax.set_title('Vector Field of U and V Components (Magnitude Thresholded)')

# 12. Show the plot
plt.show()
