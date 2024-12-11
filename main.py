import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation


grib_file = 'data950pa.grib'  # Replace with your GRIB file path
ds = xr.open_dataset(grib_file, engine='cfgrib')


print(ds)  


u = ds['u']  
v = ds['v']  


if 'time' in ds.dims:
    times = ds['time']
else:
    raise ValueError("No time dimension found in the dataset.")

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

def compute_jacobian(u, v, dx, dy):
    """
    Compute the Jacobian matrix at each grid point for the 2D velocity field.
    """
    dudx = (u.shift(longitude=-1) - u) / dx
    dudy = (u.shift(latitude=-1) - u) / dy
    dvdx = (v.shift(longitude=-1) - v) / dx
    dvdy = (v.shift(latitude=-1) - v) / dy

    # Construct the Jacobian matrix
    J11 = dudx
    J12 = dudy
    J21 = dvdx
    J22 = dvdy

    return J11, J12, J21, J22

def is_center(J11, J12, J21, J22):
    """
    Check if the Jacobian matrix corresponds to a center (purely imaginary eigenvalues).
    The Jacobian matrix is represented as:
    [[J11, J12],
     [J21, J22]]
    """

    # Construct the Jacobian matrix
    J = np.array([[J11, J12], [J21, J22]])

    # Compute the eigenvalues of the Jacobian matrix
    eigvals = la.eigvals(J)

    # Check if both eigenvalues are purely imaginary (i.e., the real part is near zero)
    return np.isclose(np.real(eigvals[0]), 0) and np.isclose(np.real(eigvals[1]), 0) and \
           np.imag(eigvals[0]) != 0 and np.imag(eigvals[1]) != 0

def update(frame):
    global center_marker
    time_idx = frame
    time = times[time_idx]

    # Extract and flip the data along the latitude axis
    u_time = u.isel(time=time_idx).isel(latitude=slice(None, None, -1))
    v_time = v.isel(time=time_idx).isel(latitude=slice(None, None, -1))

    # Calculate wind magnitude
    magnitude = np.sqrt(u_time**2 + v_time**2)

    vorticityThreshold = 0.0002

    dx = np.gradient(ds['longitude'].values)[0] * 111000  # Convert degrees to meters
    dy = np.gradient(ds['latitude'].values)[0] * 111000  # Convert degrees to meters

    high_vorticity_idx = np.where(calculate_vorticity(u_time, v_time, dx,dy) > vorticityThreshold)

    # Extract coordinates of low-wind regions
    low_wind_lats = ds['latitude'].values[::-1][high_vorticity_idx[0]]
    low_wind_lons = ds['longitude'].values[high_vorticity_idx[1]]

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

    # Mark low-wind-speed points
    if center_marker:
        center_marker[0].remove()
    center_marker = ax.plot(
        low_wind_lons, low_wind_lats, 'bo', markersize=4, label='Low Wind Speed'
    )
    ax.legend(loc='upper right')
    ax.set_title(f'Normal Map and Low Wind Regions at {time.values}')


ani = FuncAnimation(fig, update, frames=len(times), repeat=True)


plt.show()
