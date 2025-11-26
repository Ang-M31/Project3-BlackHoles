import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, G

st.set_page_config(
    page_title="Black Hole Calculator",
    page_icon="🕳️",
    layout="wide"
)

st.title("Black Hole Calculator")
st.markdown("Calculate and visualize Schwarzschild radius of a black hole")

# Sidebar for inputs
st.sidebar.header("Black Hole Parameters")

# Mass input
mass_solar = st.sidebar.number_input(
    "Mass (Solar Masses)",
    min_value=0.1,
    max_value=1000000.0,
    value=10.0,
    step=0.1,
    help="Mass of the black hole in solar masses"
)

# Calculate properties
solar_mass = 1.989e30  # kg
mass_kg = mass_solar * solar_mass

# Schwarzschild radius
schwarzschild_radius = (2 * G * mass_kg) / (c ** 2)
schwarzschild_radius_km = schwarzschild_radius / 1000

# Event horizon area
event_horizon_area = 4 * np.pi * schwarzschild_radius ** 2

# Hawking temperature (simplified)
hawking_temp = (6.17e-8) / mass_solar  # Kelvin

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.header("Physical Properties")
    
    st.metric("Schwarzschild Radius", f"{schwarzschild_radius_km:.2f} km")
    st.metric("Mass", f"{mass_solar:.2f} M☉")
    st.metric("Event Horizon Area", f"{event_horizon_area:.2e} m²")
    st.metric("Hawking Temperature", f"{hawking_temp:.2e} K")
    
    # Additional info
    st.info(f"**Mass in kilograms:** {mass_kg:.2e} kg")

with col2:
    st.header("Visualization")
    
    # Create a simple visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the event horizon as a circle
    circle = plt.Circle((0, 0), schwarzschild_radius_km, 
                       color='black', fill=True, alpha=0.8)
    ax.add_patch(circle)
    
    # Draw accretion disk (simplified)
    inner_radius = schwarzschild_radius_km * 1.5
    outer_radius = schwarzschild_radius_km * 3
    disk = plt.Circle((0, 0), outer_radius, 
                     color='orange', fill=False, linewidth=2, alpha=0.6)
    ax.add_patch(disk)
    
    ax.set_xlim(-outer_radius * 1.5, outer_radius * 1.5)
    ax.set_ylim(-outer_radius * 1.5, outer_radius * 1.5)
    ax.set_aspect('equal')
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Distance (km)")
    ax.set_title("Black Hole Event Horizon")
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# Additional information
st.header("📚 Information")
st.markdown("""
### Schwarzschild Radius
The Schwarzschild radius is the radius of the event horizon of a non-rotating black hole.
Objects within this radius cannot escape the black hole's gravitational pull.

### Hawking Temperature
The theoretical temperature of a black hole due to Hawking radiation. 
Smaller black holes have higher temperatures and evaporate faster.

### Event Horizon
The boundary beyond which nothing, not even light, can escape the black hole's gravity.
""")

# Footer
st.markdown("---")
st.markdown("**Note:** This calculator uses simplified models. Real black holes may have rotation and charge.")
