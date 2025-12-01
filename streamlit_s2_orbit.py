"""
Streamlit App: Animated 2D Visualization of Star S2's Orbit Around Sagittarius A*
This app uses orbital parameters to create a moving visualization of S2's orbit.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time
import sys
import os

# Try to import from Project3_Code
def get_s2_orbital_parameters():
    """
    Get published orbital parameters for star S2 (S0-2) around Sagittarius A*.
    These values come from Project3_Code or use published defaults.
    """
    # Default published values (used if Project3_Code can't be loaded)
    default_params = {
        'a': 0.1255,  # Semi-major axis in arcseconds
        'e': 0.88466,  # Eccentricity
        'i': 134.567,  # Inclination in degrees
        'Omega': 226.94,  # Longitude of ascending node in degrees
        'omega': 66.32,  # Argument of periapsis in degrees
        'P': 16.0518,  # Orbital period in years
        'T0': 2018.38,  # Time of pericenter passage (year) - May 2018 pericenter
        'M_bh': 4.154e6,  # Mass of Sgr A* in solar masses
        'distance_gc': 8.178,  # Distance to Galactic Center in kiloparsecs
    }
    
    try:
        # Try to read and execute Project3_Code file
        with open('Project3_Code', 'r', encoding='utf-8') as f:
            code = f.read()
            # Create a namespace for execution with mock imports for astroquery
            namespace = {
                '__builtins__': __builtins__,
                # Mock astroquery modules to prevent import errors
                'astroquery': type('MockAstroquery', (), {}),
            }
            # Suppress import errors during execution
            try:
                exec(code, namespace)
                if 'get_s2_orbital_parameters' in namespace:
                    return namespace['get_s2_orbital_parameters']()
            except (ImportError, ModuleNotFoundError):
                # If astroquery is not available, use defaults
                pass
    except (FileNotFoundError, IOError, Exception):
        # If file can't be read or executed, use defaults
        pass
    
    # Return default values
    return default_params

# Page configuration
st.set_page_config(
    page_title="S2 Orbit Visualization",
    page_icon="⭐",
    layout="wide"
)

st.title("Star S2 (S0-2) Orbit Around Sagittarius A*")
st.markdown("""
This visualization shows the orbit of star S2 around the supermassive black hole Sagittarius A* 
at the center of our Milky Way galaxy. The orbit is highly elliptical with a period of ~16 years.
""")

# Get orbital parameters
try:
    params = get_s2_orbital_parameters()
    st.sidebar.success("✓ Orbital parameters loaded")
except Exception as e:
    st.sidebar.error(f"Error loading parameters: {e}")
    # Use default published values
    params = {
        'a': 0.1255,  # arcseconds
        'e': 0.88466,
        'i': 134.567,  # degrees
        'Omega': 226.94,  # degrees
        'omega': 66.32,  # degrees
        'P': 16.0518,  # years
        'T0': 2002.321,  # year
        'M_bh': 4.154e6,  # solar masses
        'distance_gc': 8.178,  # kiloparsecs
    }
    st.sidebar.warning("Using default published parameters")

# Sidebar controls
st.sidebar.header("Orbital Parameters")
st.sidebar.markdown("**Current Values:**")

# Display parameters (allow editing)
a = st.sidebar.number_input(
    "Semi-major axis a (arcsec)",
    value=params['a'],
    min_value=0.01,
    max_value=1.0,
    step=0.001,
    format="%.4f"
)

e = st.sidebar.number_input(
    "Eccentricity e",
    value=params['e'],
    min_value=0.0,
    max_value=0.99,
    step=0.001,
    format="%.5f"
)

i_deg = st.sidebar.number_input(
    "Inclination i (degrees)",
    value=params['i'],
    min_value=0.0,
    max_value=180.0,
    step=0.1,
    format="%.2f"
)

Omega_deg = st.sidebar.number_input(
    "Longitude of ascending node Ω (degrees)",
    value=params['Omega'],
    min_value=0.0,
    max_value=360.0,
    step=0.1,
    format="%.2f"
)

omega_deg = st.sidebar.number_input(
    "Argument of periapsis ω (degrees)",
    value=params['omega'],
    min_value=0.0,
    max_value=360.0,
    step=0.1,
    format="%.2f"
)

P = st.sidebar.number_input(
    "Orbital period P (years)",
    value=params['P'],
    min_value=1.0,
    max_value=100.0,
    step=0.1,
    format="%.4f"
)

T0 = st.sidebar.number_input(
    "Time of pericenter passage T₀ (year)",
    value=params['T0'],
    min_value=1990.0,
    max_value=2030.0,
    step=0.01,
    format="%.3f"
)

# Animation controls
st.sidebar.header("Animation Controls")
animation_speed = st.sidebar.slider(
    "Animation Speed (years per second)",
    min_value=0.1,
    max_value=10.0,
    value=2.0,
    step=0.1
)

time_range = st.sidebar.slider(
    "Time Range (years from T₀)",
    min_value=-P,
    max_value=P * 2,
    value=(0.0, P),
    step=0.1
)

num_points = st.sidebar.slider(
    "Number of points in orbit",
    min_value=50,
    max_value=1000,
    value=500,
    step=50
)

show_trail = st.sidebar.checkbox("Show orbital trail", value=True)
show_ellipse = st.sidebar.checkbox("Show orbital ellipse", value=True)

# Convert angles to radians
i = np.radians(i_deg)
Omega = np.radians(Omega_deg)
omega = np.radians(omega_deg)

# Constants
G = 6.67430e-11  # m³ kg⁻¹ s⁻²
M_sun = 1.989e30  # kg
pc_to_m = 3.085677581e16  # meters
arcsec_to_rad = np.pi / (180 * 3600)  # radians per arcsecond

# Distance to Galactic Center
distance_gc_m = params['distance_gc'] * 1000 * pc_to_m

# Convert semi-major axis from arcseconds to meters
a_arcsec = a
a_rad = a_arcsec * arcsec_to_rad
a_m = a_rad * distance_gc_m

# Calculate mean motion (n = 2π/P)
P_seconds = P * 365.25 * 24 * 3600
n = 2 * np.pi / P_seconds  # rad/s

# Time array
t_start = time_range[0] * 365.25 * 24 * 3600  # Convert to seconds
t_end = time_range[1] * 365.25 * 24 * 3600
t_array = np.linspace(t_start, t_end, num_points)

# Reference time (T0) in seconds since year 2000
T0_seconds = (T0 - 2000.0) * 365.25 * 24 * 3600

# Calculate orbital positions
def calculate_orbit_positions(times):
    """Calculate x, y positions in sky plane for given times."""
    # Mean anomaly: M = n(t - T0)
    M = n * (times - T0_seconds)
    
    # Solve Kepler's equation: E = M + e*sin(E) using Newton's method
    E = M.copy()  # Initial guess
    for _ in range(20):  # Iterate to solve
        E = M + e * np.sin(E)
    
    # True anomaly: ν = 2*arctan(sqrt((1+e)/(1-e)) * tan(E/2))
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )
    
    # Distance from focus: r = a(1 - e²) / (1 + e*cos(ν))
    r = a_m * (1 - e**2) / (1 + e * np.cos(nu))
    
    # Position in orbital plane (x along major axis, y along minor axis)
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = np.zeros_like(x_orb)
    
    # Rotation matrices to transform from orbital plane to sky plane
    # Rotation 1: Argument of periapsis (ω)
    R_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])
    
    # Rotation 2: Inclination (i)
    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    # Rotation 3: Longitude of ascending node (Ω)
    R_Omega = np.array([
        [np.cos(Omega), -np.sin(Omega), 0],
        [np.sin(Omega), np.cos(Omega), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: R = R_Ω * R_i * R_ω
    R = R_Omega @ R_i @ R_omega
    
    # Transform positions
    pos_orb = np.vstack([x_orb, y_orb, z_orb])
    pos_3d = R @ pos_orb
    
    # Project onto sky plane (x, y coordinates in arcseconds)
    x_sky = pos_3d[0, :] / distance_gc_m / arcsec_to_rad
    y_sky = pos_3d[1, :] / distance_gc_m / arcsec_to_rad
    
    return x_sky, y_sky, times

# Calculate full orbit
x_full, y_full, t_full = calculate_orbit_positions(t_array)

# Current time for animation
current_year = st.sidebar.number_input(
    "Current time (year)",
    value=float(2024.0),
    min_value=1990.0,
    max_value=2050.0,
    step=0.01,
    format="%.2f"
)

# Calculate current position
t_current = (current_year - 2000.0) * 365.25 * 24 * 3600
x_current, y_current, _ = calculate_orbit_positions(np.array([t_current]))

# Main visualization
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Orbital Visualization")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot full orbit
    if show_trail:
        ax.plot(x_full, y_full, 'b-', alpha=0.3, linewidth=1, label='Orbit trail')
    
    # Plot orbital ellipse outline (projected onto sky plane)
    if show_ellipse:
        # The ellipse in the sky plane is the projection of the 3D orbit
        # We'll draw it using the calculated orbit points for accuracy
        # Create a denser set of points for smooth ellipse
        t_ellipse = np.linspace(0, P_seconds, 1000)
        x_ellipse, y_ellipse, _ = calculate_orbit_positions(t_ellipse + T0_seconds)
        ax.plot(x_ellipse, y_ellipse, 'g--', linewidth=1, alpha=0.4, label='Orbital ellipse')
    
    # Plot Sgr A* at center
    ax.plot(0, 0, 'ko', markersize=15, label='Sagittarius A*')
    ax.plot(0, 0, 'yo', markersize=8, label='Sgr A* center')
    
    # Plot current position of S2
    ax.plot(x_current[0], y_current[0], 'ro', markersize=12, label=f'S2 (year {current_year:.2f})')
    ax.plot(x_current[0], y_current[0], 'r*', markersize=20)
    
    # Mark pericenter (closest approach)
    t_peri = T0_seconds
    x_peri, y_peri, _ = calculate_orbit_positions(np.array([t_peri]))
    ax.plot(x_peri[0], y_peri[0], 'g^', markersize=10, label='Pericenter (closest)')
    
    # Mark apocenter (farthest point)
    t_apo = T0_seconds + P_seconds / 2
    x_apo, y_apo, _ = calculate_orbit_positions(np.array([t_apo]))
    ax.plot(x_apo[0], y_apo[0], 'rv', markersize=10, label='Apocenter (farthest)')
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('X position (arcseconds)', fontsize=12)
    ax.set_ylabel('Y position (arcseconds)', fontsize=12)
    ax.set_title(f'S2 Orbit Around Sagittarius A*\n(Current: {current_year:.2f} year)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    
    # Set reasonable axis limits
    max_range = a * (1 + e) * 1.2  # Slightly larger than apocenter
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    
    st.pyplot(fig)

with col2:
    st.header("Orbital Information")
    
    # Calculate some orbital properties
    b_arcsec = a * np.sqrt(1 - e**2)  # Semi-minor axis
    peri_distance = a * (1 - e)  # Pericenter distance
    apo_distance = a * (1 + e)   # Apocenter distance
    
    st.metric("Semi-major axis", f"{a:.4f} arcsec")
    st.metric("Semi-minor axis", f"{b_arcsec:.4f} arcsec")
    st.metric("Eccentricity", f"{e:.5f}")
    st.metric("Orbital period", f"{P:.2f} years")
    st.metric("Pericenter distance", f"{peri_distance:.4f} arcsec")
    st.metric("Apocenter distance", f"{apo_distance:.4f} arcsec")
    
    st.markdown("---")
    st.subheader("Current Position")
    st.metric("X position", f"{x_current[0]:.4f} arcsec")
    st.metric("Y position", f"{y_current[0]:.4f} arcsec")
    
    # Calculate distance from Sgr A*
    r_current = np.sqrt(x_current[0]**2 + y_current[0]**2)
    st.metric("Distance from Sgr A*", f"{r_current:.4f} arcsec")
    
    # Calculate time since pericenter
    if t_current >= T0_seconds:
        time_since_peri = (t_current - T0_seconds) / (365.25 * 24 * 3600)
        st.metric("Time since pericenter", f"{time_since_peri:.2f} years")
    else:
        time_to_peri = (T0_seconds - t_current) / (365.25 * 24 * 3600)
        st.metric("Time to next pericenter", f"{time_to_peri:.2f} years")
    
    st.markdown("---")
    st.subheader("Orbital Elements")
    st.write(f"**Inclination (i):** {i_deg:.2f}°")
    st.write(f"**Ω (longitude of ascending node):** {Omega_deg:.2f}°")
    st.write(f"**ω (argument of periapsis):** {omega_deg:.2f}°")
    st.write(f"**T₀ (pericenter passage):** {T0:.3f} year")

# Additional information
st.markdown("---")
st.header("About Star S2 (S0-2)")
st.markdown("""
Star S2 (also known as S0-2) is one of the closest stars to Sagittarius A*, the supermassive black hole 
at the center of our Milky Way galaxy. Its highly elliptical orbit has been carefully monitored for decades, 
providing crucial evidence for the existence of a supermassive black hole and testing Einstein's theory of 
general relativity.

**Key Facts:**
- **Orbital Period:** ~16 years
- **Closest Approach:** ~120 AU from Sgr A* (at pericenter)
- **Farthest Distance:** ~2000 AU from Sgr A* (at apocenter)
- **Eccentricity:** ~0.88 (highly elliptical)
- **Last Pericenter Passage:** May 2018 (T₀ ≈ 2018.38)

The orbit of S2 has been used to:
1. Measure the mass of Sgr A* (~4 million solar masses)
2. Test predictions of general relativity (gravitational redshift, orbital precession)
3. Constrain the distance to the Galactic Center
""")

# Footer
st.markdown("---")
st.caption("Data sources: GRAVITY Collaboration, ESO observations, and published orbital element catalogs")

