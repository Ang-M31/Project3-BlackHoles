"""
Streamlit App: Animated 2D Visualization of Star S2's Orbit Around the Black Hole at the center of the Milky Way*
This app uses orbital parameters to create a moving visualization of S2's orbit.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.constants import c, G
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

st.title("Star S2 (S0-2) Orbit Around the Black Hole at the center of the Milky Way")
st.markdown("""
This visualization shows the orbit of star S2 around the supermassive black hole at the center of the Milky Way 
at the center of our Milky Way galaxy. The orbit is highly elliptical with a period of ~16 years.
""")

# Get orbital parameters
try:
    params = get_s2_orbital_parameters()
except Exception as e:
    # Use default published values
    params = {
        'a': 0.1255,  # arcseconds
        'e': 0.88466,
        'i': 134.567,  # degrees
        'Omega': 226.94,  # degrees
        'omega': 66.32,  # degrees
        'P': 16.0518,  # years
        'T0': 2018.38,  # year
        'M_bh': 4.154e6,  # solar masses
        'distance_gc': 8.178,  # kiloparsecs
    }

# Extract orbital parameters (read-only, fixed values)
a = params['a']
e = params['e']
i_deg = params['i']
Omega_deg = params['Omega']
omega_deg = params['omega']
P = params['P']
T0 = params['T0']

# ============================================================================
# SIDEBAR: Animation Controls
# ============================================================================
st.sidebar.header("🎬 Animation Controls")

# Initialize session state for animation
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_time' not in st.session_state:
    st.session_state.current_time = 2024.0
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'show_orbital_params' not in st.session_state:
    st.session_state.show_orbital_params = False

# Animation speed selector
speed_options = {
    "Slow": 0.5,    # 0.5 years per second
    "Medium": 2.0,  # 2 years per second
    "Fast": 5.0     # 5 years per second
}

speed_selected = st.sidebar.radio(
    "Animation Speed",
    options=list(speed_options.keys()),
    index=1,  # Default to Medium
    help=f"Slow: {speed_options['Slow']} years/sec, Medium: {speed_options['Medium']} years/sec, Fast: {speed_options['Fast']} years/sec"
)
animation_speed = speed_options[speed_selected]
st.sidebar.caption(f"**{speed_selected}**: {animation_speed} years/second")

# Play/Pause controls
if st.sidebar.button("▶️ Play" if not st.session_state.is_playing else "⏸️ Pause", 
             use_container_width=True, type="primary"):
    st.session_state.is_playing = not st.session_state.is_playing
    st.session_state.last_update = time.time()

# Time controls
st.sidebar.markdown("---")
# Manual time control - if user moves slider, pause animation
slider_time = st.sidebar.slider(
    "Current Time (year)",
    min_value=float(T0 - P),
    max_value=float(T0 + P * 2),
    value=float(st.session_state.current_time),
    step=0.01,
    format="%.2f",
    key="time_slider"
)
# If slider changed, update time and pause
if abs(slider_time - st.session_state.current_time) > 0.001:
    st.session_state.current_time = slider_time
    st.session_state.is_playing = False

# Reset button
if st.sidebar.button("🔄 Reset to T₀", use_container_width=True):
    st.session_state.current_time = T0
    st.session_state.is_playing = False
    st.rerun()

# Button to show orbital parameters at paused point (only when paused)
st.sidebar.markdown("---")
if not st.session_state.is_playing:
    if st.sidebar.button("📊 Show Orbital Parameters", use_container_width=True):
        st.session_state.show_orbital_params = not st.session_state.show_orbital_params
else:
    st.sidebar.caption("⏸️ Pause animation to view orbital parameters")

# Update time if playing
if st.session_state.is_playing:
    current_time_actual = time.time()
    elapsed = current_time_actual - st.session_state.last_update
    time_delta = elapsed * animation_speed
    st.session_state.current_time += time_delta
    st.session_state.last_update = current_time_actual
    
    # Wrap around if beyond one period
    if st.session_state.current_time > T0 + P:
        st.session_state.current_time = T0
    elif st.session_state.current_time < T0:
        st.session_state.current_time = T0
    
    # Auto-rerun to update animation (small delay to control frame rate)
    time.sleep(0.05)  # ~20 fps
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Display Options")

num_points = st.sidebar.slider(
    "Number of points in orbit",
    min_value=50,
    max_value=1000,
    value=500,
    step=50
)

show_trail = st.sidebar.checkbox("Show orbital trail", value=True)
show_ellipse = st.sidebar.checkbox("Show orbital ellipse", value=True)
show_schwarzschild = st.sidebar.checkbox("Show Schwarzschild radius", value=True)

# Convert angles to radians
i = np.radians(i_deg)
Omega = np.radians(Omega_deg)
omega = np.radians(omega_deg)

# Constants (using scipy.constants like the first app)
M_sun = 1.989e30  # kg
pc_to_m = 3.085677581e16  # meters
arcsec_to_rad = np.pi / (180 * 3600)  # radians per arcsecond

# Distance to Galactic Center
distance_gc_m = params['distance_gc'] * 1000 * pc_to_m

# Calculate Schwarzschild radius for Sgr A* (using same method as streamlit_app.py)
# Mass: 4 million solar masses = 4e6 * 1.989e30 kg
M_bh_kg = params['M_bh'] * M_sun  # Mass of Sgr A* in kg
schwarzschild_radius = (2 * G * M_bh_kg) / (c ** 2)  # Schwarzschild radius in meters
schwarzschild_radius_km = schwarzschild_radius / 1000

# Convert Schwarzschild radius to arcseconds
# Formula: angle (rad) = radius / distance, then convert rad to arcsec
# angle_rad = schwarzschild_radius / distance_gc_m
# angle_arcsec = angle_rad / arcsec_to_rad
r_schwarzschild_arcsec = (schwarzschild_radius / distance_gc_m) / arcsec_to_rad

# Convert semi-major axis from arcseconds to meters
a_arcsec = a
a_rad = a_arcsec * arcsec_to_rad
a_m = a_rad * distance_gc_m

# Calculate mean motion (n = 2π/P)
P_seconds = P * 365.25 * 24 * 3600
n = 2 * np.pi / P_seconds  # rad/s

# Time array for orbit calculation (already defined above, this is just for reference)
# t_array is calculated above after getting orbital parameters

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

# Calculate full orbit for trail
# Time array for full orbit visualization
t_start = (T0 - P - 2000.0) * 365.25 * 24 * 3600
t_end = (T0 + P * 2 - 2000.0) * 365.25 * 24 * 3600
t_array = np.linspace(t_start, t_end, num_points)
x_full, y_full, t_full = calculate_orbit_positions(t_array)

# Current time for animation (from session state)
current_year = st.session_state.current_time
t_current = (current_year - 2000.0) * 365.25 * 24 * 3600

# Calculate full 3D orbit positions function
def calculate_orbit_positions_3d(times):
    """Calculate x, y, z positions in 3D space for given times."""
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
    
    # Rotation matrices to transform from orbital plane to 3D space
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
    
    # Convert to arcseconds for display
    x_3d = pos_3d[0, :] / distance_gc_m / arcsec_to_rad
    y_3d = pos_3d[1, :] / distance_gc_m / arcsec_to_rad
    z_3d = pos_3d[2, :] / distance_gc_m / arcsec_to_rad
    
    return x_3d, y_3d, z_3d, times

# Main visualization
st.header("Orbital Visualization (3D)")

# Create 3D figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Calculate full 3D orbit
x_full_3d, y_full_3d, z_full_3d, _ = calculate_orbit_positions_3d(t_array)

# Plot full orbit trail
if show_trail:
    ax.plot(x_full_3d, y_full_3d, z_full_3d, 'b-', alpha=0.3, linewidth=1, label='Orbit trail')

# Plot orbital ellipse outline
if show_ellipse:
    t_ellipse = np.linspace(0, P_seconds, 1000)
    x_ellipse, y_ellipse, z_ellipse, _ = calculate_orbit_positions_3d(t_ellipse + T0_seconds)
    ax.plot(x_ellipse, y_ellipse, z_ellipse, 'g--', linewidth=1, alpha=0.4, label='Orbital ellipse')

# Plot transparent yellow sphere extending to Schwarzschild radius
# This represents the event horizon region around the black hole center
# The sphere is centered at (0,0,0) with radius = Schwarzschild radius
# Note: The Schwarzschild radius is extremely small (microarcseconds), so it may appear as a point
u_yellow = np.linspace(0, 2 * np.pi, 40)
v_yellow = np.linspace(0, np.pi, 30)
x_yellow = r_schwarzschild_arcsec * np.outer(np.cos(u_yellow), np.sin(v_yellow))
y_yellow = r_schwarzschild_arcsec * np.outer(np.sin(u_yellow), np.sin(v_yellow))
z_yellow = r_schwarzschild_arcsec * np.outer(np.ones_like(u_yellow), np.cos(v_yellow))
# Make it more visible with higher alpha and ensure it's always shown
ax.plot_surface(x_yellow, y_yellow, z_yellow, color='yellow', alpha=0.4, 
               linewidth=0, shade=False, label='Schwarzschild radius (yellow)')

# Plot estimated black hole center at the center of the yellow Schwarzschild radius sphere
# This dot represents the center of the 4 million solar mass black hole
# Positioned at (0,0,0) which is the center of the yellow sphere
ax.scatter([0], [0], [0], c='black', s=20, marker='o', label='Estimated Black Hole Center', 
           edgecolors='black', linewidths=0.5, zorder=100)

# Calculate current 3D position
x_current_3d, y_current_3d, z_current_3d, _ = calculate_orbit_positions_3d(np.array([t_current]))

# Plot current position of S2
ax.scatter([x_current_3d[0]], [y_current_3d[0]], [z_current_3d[0]], 
          c='red', s=200, marker='*', label=f'S2 (year {current_year:.2f})')
ax.scatter([x_current_3d[0]], [y_current_3d[0]], [z_current_3d[0]], 
          c='red', s=100, marker='o', alpha=0.5)

# Mark pericenter (closest approach)
t_peri = T0_seconds
x_peri_3d, y_peri_3d, z_peri_3d, _ = calculate_orbit_positions_3d(np.array([t_peri]))
ax.scatter([x_peri_3d[0]], [y_peri_3d[0]], [z_peri_3d[0]], 
          c='green', s=150, marker='^', label='Pericenter (closest)')

# Mark apocenter (farthest point)
t_apo = T0_seconds + P_seconds / 2
x_apo_3d, y_apo_3d, z_apo_3d, _ = calculate_orbit_positions_3d(np.array([t_apo]))
ax.scatter([x_apo_3d[0]], [y_apo_3d[0]], [z_apo_3d[0]], 
          c='red', s=150, marker='v', label='Apocenter (farthest)')

# Set labels and title
ax.set_xlabel('X position (arcseconds)', fontsize=12)
ax.set_ylabel('Y position (arcseconds)', fontsize=12)
ax.set_zlabel('Z position (arcseconds)', fontsize=12)
ax.set_title(f'S2 Orbit Around the Black Hole at the center of the Milky Way (3D)\n(Current: {current_year:.2f} year)', 
            fontsize=14, fontweight='bold')

# Set reasonable axis limits
max_range = a * (1 + e) * 1.2  # Slightly larger than apocenter
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

# Set viewing angle
ax.view_init(elev=20, azim=45)

ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))

st.pyplot(fig)

# Show orbital parameters in main section if button was clicked (only when paused)
if st.session_state.show_orbital_params and not st.session_state.is_playing:
    st.markdown("---")
    st.header("📊 Orbital Parameters at Current Time")
    
    # Calculate some orbital properties
    b_arcsec = a * np.sqrt(1 - e**2)  # Semi-minor axis
    peri_distance = a * (1 - e)  # Pericenter distance
    apo_distance = a * (1 + e)   # Apocenter distance
    
    col_param1, col_param2 = st.columns(2)
    
    with col_param1:
        st.subheader("Orbital Properties")
        st.metric("Semi-major axis", f"{a:.4f} arcsec")
        st.metric("Semi-minor axis", f"{b_arcsec:.4f} arcsec")
        st.metric("Eccentricity", f"{e:.5f}")
        st.metric("Orbital period", f"{P:.2f} years")
        st.metric("Pericenter distance", f"{peri_distance:.4f} arcsec")
        st.metric("Apocenter distance", f"{apo_distance:.4f} arcsec")
    
    with col_param2:
        st.subheader("Current Position (3D)")
        # Calculate current 3D position for display
        x_curr_3d, y_curr_3d, z_curr_3d, _ = calculate_orbit_positions_3d(np.array([t_current]))
        st.metric("X position", f"{x_curr_3d[0]:.4f} arcsec")
        st.metric("Y position", f"{y_curr_3d[0]:.4f} arcsec")
        st.metric("Z position", f"{z_curr_3d[0]:.4f} arcsec")
        
        # Calculate distance from Sgr A*
        r_current = np.sqrt(x_curr_3d[0]**2 + y_curr_3d[0]**2 + z_curr_3d[0]**2)
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
    col_elem1, col_elem2 = st.columns(2)
    with col_elem1:
        st.write(f"**Inclination (i):** {i_deg:.2f}°")
        st.write(f"**Ω (longitude of ascending node):** {Omega_deg:.2f}°")
    with col_elem2:
        st.write(f"**ω (argument of periapsis):** {omega_deg:.2f}°")
        st.write(f"**T₀ (pericenter passage):** {T0:.3f} year")

# Orbital Information (moved to bottom)
st.markdown("---")
st.header("Orbital Information")

# Calculate some orbital properties
b_arcsec = a * np.sqrt(1 - e**2)  # Semi-minor axis
peri_distance = a * (1 - e)  # Pericenter distance
apo_distance = a * (1 + e)   # Apocenter distance

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.subheader("Orbital Properties")
    st.metric("Semi-major axis", f"{a:.4f} arcsec")
    st.metric("Semi-minor axis", f"{b_arcsec:.4f} arcsec")
    st.metric("Eccentricity", f"{e:.5f}")
    st.metric("Orbital period", f"{P:.2f} years")
    st.metric("Pericenter distance", f"{peri_distance:.4f} arcsec")
    st.metric("Apocenter distance", f"{apo_distance:.4f} arcsec")

with col_info2:
    st.subheader("Orbital Elements")
    st.write(f"**Inclination (i):** {i_deg:.2f}°")
    st.write(f"**Ω (longitude of ascending node):** {Omega_deg:.2f}°")
    st.write(f"**ω (argument of periapsis):** {omega_deg:.2f}°")
    st.write(f"**T₀ (pericenter passage):** {T0:.3f} year")
    st.write(f"**Semi-major axis (a):** {a:.4f} arcsec")
    st.write(f"**Eccentricity (e):** {e:.5f}")

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

