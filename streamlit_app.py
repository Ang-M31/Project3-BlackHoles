import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.constants import c, G

st.set_page_config(
    page_title="Schwarzschild Radius",
    page_icon="🕳️",
    layout="wide"
)

st.title("Understanding the Schwarzschild Radius")
st.markdown("Commonly known as the Event Horizon of a black hole, the Schwarzschild Radius mathematically describes the tipping point when an object becomes so dense, it must become a black hole.")
st.markdown(r"Formula: $r_s = \frac{2GM}{c^2}$")
st.markdown(
    "G = Gravitational constant  \n"
    "M = Mass, in kilograms  \n"
    "c = Speed of Light  \n"
    "Assumes the object is symmetrical with no rotation"
)
# Sidebar for inputs
st.sidebar.header("Black Hole Estimator")

# Mass input
mass_solar = st.sidebar.number_input(
    "Mass (Solar Masses)",
    min_value=0.1,
    max_value=4000000.0,
    value=10.0,
    step=0.1,
    help="Mass of the black hole in solar masses"
)
st.sidebar.markdown(
    "**Examples:**  \n"
    "- Typical star: 1 M☉  \n"
    "- Massive star: 20 M☉  \n"
    "- Stellar black hole: 50 M☉  \n"
    "- Intermediate black hole: 10⁴ M☉  \n"
    "- Black hole S2 Orbits: 4.3×10⁶ M☉"
)

# Calculate properties
solar_mass = 1.989e30  # kg
mass_kg = mass_solar * solar_mass

# Schwarzschild radius
schwarzschild_radius = (2 * G * mass_kg) / (c ** 2)
schwarzschild_radius_km = schwarzschild_radius / 1000
schwarzschild_radius_miles = schwarzschild_radius_km * 0.621371

size_comparisons = [
    (0.1, "about the length of a football field"),
    (1, "roughly the height of the Empire State Building"),
    (5, "similar to the width of Manhattan"),
    (16, "close to the width of the Grand Canyon"),
    (100, "about the distance across Los Angeles"),
    (6371, "comparable to Earth's radius"),
    (696340, "approaching the Sun's radius"),
]

relative_size = "far larger than the Sun's radius"
for threshold, description in size_comparisons:
    if schwarzschild_radius_km <= threshold:
        relative_size = description
        break

# Event horizon area
event_horizon_area = 4 * np.pi * schwarzschild_radius ** 2

# Hawking temperature (simplified)
hawking_temp = (6.17e-8) / mass_solar  # Kelvin

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("Properties")
    st.markdown("*This calculator uses simplified models. Real black holes may have rotation and charge.*")
    
    st.metric(
        "Schwarzschild Radius",
        f"{schwarzschild_radius_km:.2f} km ({schwarzschild_radius_miles:.2f} mi)",
    )
    st.metric("Mass", f"{mass_solar:.2f} M☉")
    st.caption(f"≈ {mass_kg:.2e} kg")
    st.metric("Event Horizon Area", f"{event_horizon_area:.2e} m²")
    st.metric("Hawking Temperature", f"{hawking_temp:.2e} K")
    st.markdown(
        "<small><em>The theoretical temperature of a black hole due to Hawking "
        "radiation. Smaller black holes have higher temperatures and evaporate faster."
        "</em></small>",
        unsafe_allow_html=True,
    )

with col2:
    st.header("Visualization")
    st.markdown(f"*This event horizon radius is {relative_size}, making the entire diameter ≈ {2 * schwarzschild_radius_km:.2f} km.*")
    
    # Create a 3D visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Event horizon sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = schwarzschild_radius_km * np.outer(np.cos(u), np.sin(v))
    y = schwarzschild_radius_km * np.outer(np.sin(u), np.sin(v))
    z = schwarzschild_radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="black", alpha=0.8, linewidth=0, shade=True)
    
    # Accretion disk (thin torus approximation)
    disk_r_outer = schwarzschild_radius_km * 3
    disk_r_inner = schwarzschild_radius_km * 1.2
    disk_u = np.linspace(0, 2 * np.pi, 200)
    disk_v = np.linspace(disk_r_inner, disk_r_outer, 2)
    disk_u, disk_v = np.meshgrid(disk_u, disk_v)
    disk_x = disk_v * np.cos(disk_u)
    disk_y = disk_v * np.sin(disk_u)
    disk_z = np.zeros_like(disk_x)
    ax.plot_surface(
        disk_x, disk_y, disk_z, color="orange", alpha=0.3, linewidth=0, shade=False
    )
    
    limit = disk_r_outer * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("Event Horizon & Accretion Disk")
    ax.view_init(elev=25, azim=35)
    
    st.pyplot(fig)