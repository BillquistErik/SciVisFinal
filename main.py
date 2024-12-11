import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import cartopy.crs as ccrs
from matplotlib.animation import FuncAnimation


grib_file = 'data2.grib'  # Replace with your GRIB file path
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
    Returns the Jacobian matrix as a 2x2 matrix.
    """
    # Compute differences with appropriate shifts
    dudx = (u.shift(longitude=-1) - u) / dx
    dudy = (u.shift(latitude=-1) - u) / dy
    dvdx = (v.shift(longitude=-1) - v) / dx
    dvdy = (v.shift(latitude=-1) - v) / dy

    # Check for NaNs or Infs in the calculated derivatives
    if np.any(np.isnan([dudx, dudy, dvdx, dvdy])) or np.any(np.isinf([dudx, dudy, dvdx, dvdy])):
        return None, None, None, None  # Invalid Jacobian

    # Return Jacobian components
    return dudx, dudy, dvdx, dvdy

def is_center(J11, J12, J21, J22):
    """
    Check if the Jacobian matrix corresponds to a center (purely imaginary eigenvalues).
    """
    if J11 is None or J12 is None or J21 is None or J22 is None:
        return False  # Skip invalid Jacobian matrices

    # Construct the Jacobian matrix from its components
    J = np.array([[J11, J12], [J21, J22]])

    # Check for NaNs or Infs in the Jacobian matrix
    if np.any(np.isnan(J)) or np.any(np.isinf(J)):
        return False  # Skip invalid Jacobian matrices

    # Compute eigenvalues of the Jacobian matrix
    eigvals = la.eigvals(J)

    # Check if eigenvalues are purely imaginary (i.e., real part is close to zero)
    return np.isclose(np.real(eigvals[0]), 0) and np.isclose(np.real(eigvals[1]), 0) and np.imag(eigvals[0]) != 0 and np.imag(eigvals[1]) != 0

def update(frame):
    global center_marker
    time_idx = frame
    time = times[time_idx]

    # Extract and flip the data along the latitude axis
    u_time = u.isel(time=time_idx).isel(latitude=slice(None, None, -1))
    v_time = v.isel(time=time_idx).isel(latitude=slice(None, None, -1))

    # Calculate wind magnitude
    magnitude = np.sqrt(u_time**2 + v_time**2)

    # Identify points where wind speed is near zero
    zero_wind_threshold = 0.5  # Define a small threshold
    low_wind_idx = np.where(magnitude.values < zero_wind_threshold)

    # Extract coordinates of low-wind regions
    low_wind_lats = ds['latitude'].values[::-1][low_wind_idx[0]]
    low_wind_lons = ds['longitude'].values[low_wind_idx[1]]

    # Initialize list for centers
    centers_lats = []
    centers_lons = []

    # Compute Jacobian and check for centers at low-wind locations
    dx = np.gradient(ds['longitude'].values)[0] * 111000  
    dy = np.gradient(ds['latitude'].values)[0] * 111000  

    for lat_idx, lon_idx in zip(low_wind_idx[0], low_wind_idx[1]):
        # Calculate the Jacobian components
        J11, J12, J21, J22 = compute_jacobian(u_time, v_time, dx, dy)

        # Skip if Jacobian is invalid (contains NaN or Inf)
        if J11 is None or J12 is None or J21 is None or J22 is None:
            continue  # Skip invalid points

        # Check if this point is a center
        if is_center(J11[lat_idx, lon_idx], J12[lat_idx, lon_idx], J21[lat_idx, lon_idx], J22[lat_idx, lon_idx]):
            centers_lats.append(ds['latitude'].values[::-1][lat_idx])
            centers_lons.append(ds['longitude'].values[lon_idx])

    # Normalize vectors for RGB encoding
    u_normalized = u_time / magnitude
    v_normalized = v_time / magnitude
    R = (u_normalized + 1) / 2  # Map from [-1, 1] to [0, 1]
    G = (v_normalized + 1) / 2  # Map from [-1, 1] to [0, 1]
    B = magnitude / np.max(magnitude)  # Normalize magnitude for brightness

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

    # Mark low_wind_speed points
    if center_marker:
        center_marker[0].remove()
    center_marker = ax.plot(
        low_wind_lons, low_wind_lats, 'bo', markersize=4, label='Low Wind Speed'
    )
    ax.plot(
        centers_lons, centers_lats, 'go', markersize=8, label='Center'
    )
    ax.legend(loc='upper right')
    ax.set_title(f'Normal Map and Low Wind Regions at {time.values}')


ani = FuncAnimation(fig, update, frames=len(times), repeat=True)


plt.show()