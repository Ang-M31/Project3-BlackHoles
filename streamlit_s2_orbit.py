"""
Streamlit App: Visualization of Star S2's Orbit Around the Black Hole at the center of the Milky Way
This app displays GRAVITY Collaboration observational data with error bars and calculated orbital model.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, G
import pandas as pd
import os
import re

# Try to import astropy for coordinate transformations
try:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

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

def icrs_to_galactic(ra_arcsec, dec_arcsec, ra_err_arcsec=None, dec_err_arcsec=None):
    """
    Convert ICRS (equatorial) coordinates to Galactic coordinates.
    
    Parameters:
    - ra_arcsec: Right Ascension in arcseconds (relative to Sgr A*)
    - dec_arcsec: Declination in arcseconds (relative to Sgr A*)
    - ra_err_arcsec: RA error in arcseconds (optional)
    - dec_err_arcsec: Dec error in arcseconds (optional)
    
    Returns:
    - l_arcsec, b_arcsec: Galactic longitude and latitude in arcseconds
    - l_err_arcsec, b_err_arcsec: Errors (if provided)
    """
    if HAS_ASTROPY:
        # Sgr A* position in ICRS
        sgr_a_ra = 266.4168 * u.degree
        sgr_a_dec = -29.0078 * u.degree
        
        # Convert arcseconds to degrees and add to Sgr A* position
        ra_deg = sgr_a_ra.value + ra_arcsec / 3600.0
        dec_deg = sgr_a_dec.value + dec_arcsec / 3600.0
        
        # Create SkyCoord in ICRS
        coord = SkyCoord(ra=ra_deg * u.degree, dec=dec_deg * u.degree, frame='icrs')
        
        # Convert to galactic
        gal_coord = coord.galactic
        
        # Sgr A* in galactic coordinates (approximately l=0, b=0)
        sgr_a_gal = SkyCoord(ra=sgr_a_ra, dec=sgr_a_dec, frame='icrs').galactic
        
        # Calculate relative position in arcseconds
        l_diff = (gal_coord.l - sgr_a_gal.l).to(u.arcsec).value
        b_diff = (gal_coord.b - sgr_a_gal.b).to(u.arcsec).value
        
        # For errors, approximate transformation (simplified)
        if ra_err_arcsec is not None and dec_err_arcsec is not None:
            # Rough approximation: errors scale similarly
            l_err = np.sqrt(ra_err_arcsec**2 + dec_err_arcsec**2) * 0.7  # Approximate
            b_err = np.sqrt(ra_err_arcsec**2 + dec_err_arcsec**2) * 0.7
            return l_diff, b_diff, l_err, b_err
        
        return l_diff, b_diff
    else:
        # Fallback: return ICRS coordinates if astropy not available
        if ra_err_arcsec is not None and dec_err_arcsec is not None:
            return ra_arcsec, dec_arcsec, ra_err_arcsec, dec_err_arcsec
        return ra_arcsec, dec_arcsec

def load_fits_table_data(use_galactic=False):
    """
    Load observational data from gravity_fits_table.csv.
    Returns DataFrame with ra, dec, ra_err, dec_err in arcseconds.
    If use_galactic=True, converts to galactic coordinates (l, b).
    """
    if not os.path.exists('gravity_fits_table.csv'):
        return None
    
    try:
        df = pd.read_csv('gravity_fits_table.csv')
        
        # Find alpha and delta columns (handle Unicode and ASCII variants)
        alpha_col = None
        delta_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            # Check for alpha/RA column
            if 'α' in col or 'alpha' in col_lower or 'ra' in col_lower:
                if 'mas' in col_lower or 'mas' in col:
                    alpha_col = col
            # Check for delta/Dec column
            if 'δ' in col or 'delta' in col_lower or 'dec' in col_lower:
                if 'mas' in col_lower or 'mas' in col:
                    delta_col = col
        
        if alpha_col is None or delta_col is None:
            return None
        
        # Extract values and errors from "value ± error" format
        def parse_value_error(s):
            """Parse 'value ± error' string format."""
            if pd.isna(s) or s == '' or s == '-':
                return None, None
            s = str(s).strip()
            # Match pattern like "2.93 ± 0.55" or "3.21 ± 0.58"
            match = re.match(r'([+-]?\d+\.?\d*)\s*±\s*([+-]?\d+\.?\d*)', s)
            if match:
                value = float(match.group(1))
                error = float(match.group(2))
                return value, error
            return None, None
        
        # Parse alpha and delta columns
        ra_values = []
        ra_errors = []
        dec_values = []
        dec_errors = []
        
        for idx, row in df.iterrows():
            ra_val, ra_err = parse_value_error(row[alpha_col])
            dec_val, dec_err = parse_value_error(row[delta_col])
            
            if ra_val is not None and dec_val is not None:
                # Convert mas to arcseconds
                ra_values.append(ra_val / 1000.0)
                ra_errors.append(ra_err / 1000.0)
                dec_values.append(dec_val / 1000.0)
                dec_errors.append(dec_err / 1000.0)
        
        if len(ra_values) == 0:
            return None
        
        # Create DataFrame
        obs_data = pd.DataFrame({
            'ra': ra_values,
            'dec': dec_values,
            'ra_err': ra_errors,
            'dec_err': dec_errors
        })
        
        # Convert to galactic coordinates if requested
        if use_galactic:
            l_values = []
            b_values = []
            l_errors = []
            b_errors = []
            
            for idx, row in obs_data.iterrows():
                l, b, l_err, b_err = icrs_to_galactic(
                    row['ra'], row['dec'],
                    row['ra_err'], row['dec_err']
                )
                l_values.append(l)
                b_values.append(b)
                l_errors.append(l_err)
                b_errors.append(b_err)
            
            obs_data = pd.DataFrame({
                'ra': l_values,  # Reuse 'ra' column for galactic l
                'dec': b_values,  # Reuse 'dec' column for galactic b
                'ra_err': l_errors,  # Reuse 'ra_err' for l_err
                'dec_err': b_errors  # Reuse 'dec_err' for b_err
            })
        
        return obs_data
    
    except Exception as e:
        st.warning(f"Error loading fits table: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title="S2 Orbit Visualization",
    page_icon="⭐",
    layout="wide"
)

st.title("Star S2 (S0-2) Orbit Around the Black Hole at the center of the Milky Way")

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

# Extract orbital parameters
a = params['a']
e = params['e']
i_deg = params['i']
Omega_deg = params['Omega']
omega_deg = params['omega']
P = params['P']
T0 = params['T0']

# Initialize session state for display options
if 'show_calculated_orbit' not in st.session_state:
    st.session_state.show_calculated_orbit = False
if 'show_table_data' not in st.session_state:
    st.session_state.show_table_data = False
if 'show_table_data_2' not in st.session_state:
    st.session_state.show_table_data_2 = False

# Display options buttons
col_btn1, col_btn2, col_btn3 = st.columns(3)

with col_btn1:
    if st.button("Show Calculated Orbit", use_container_width=True, type="primary" if st.session_state.show_calculated_orbit else "secondary"):
        st.session_state.show_calculated_orbit = True
        st.session_state.show_table_data = False
        st.session_state.show_table_data_2 = False
        st.rerun()

with col_btn2:
    if st.button("Show Table Data - 1", use_container_width=True, type="primary" if st.session_state.show_table_data else "secondary"):
        st.session_state.show_table_data = True
        st.session_state.show_calculated_orbit = False
        st.session_state.show_table_data_2 = False
        st.rerun()

with col_btn3:
    if st.button("Show Table Data - 2", use_container_width=True, type="primary" if st.session_state.show_table_data_2 else "secondary"):
        st.session_state.show_table_data_2 = True
        st.session_state.show_calculated_orbit = False
        st.session_state.show_table_data = False
        st.rerun()

# Convert angles to radians
i = np.radians(i_deg)
Omega = np.radians(Omega_deg)
omega = np.radians(omega_deg)

# Constants
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

# Create column layout: graph on left, info on right
col_viz, col_info = st.columns([1, 1])

with col_viz:
    # Create 2D figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Load observational data if showing table data
    obs_data = None
    use_galactic = False
    coord_label_x = 'Right Ascension Difference (")'
    coord_label_y = 'Declination Difference (")'
    
    if st.session_state.show_table_data:
        obs_data = load_fits_table_data(use_galactic=False)
    elif st.session_state.show_table_data_2:
        obs_data = load_fits_table_data(use_galactic=True)
        use_galactic = True
        coord_label_x = 'Galactic Longitude Difference (")'
        coord_label_y = 'Galactic Latitude Difference (")'
    
    # Calculate full orbit for trail (if showing calculated orbit)
    if st.session_state.show_calculated_orbit:
        # Time array for full orbit visualization
        t_start = (T0 - P - 2000.0) * 365.25 * 24 * 3600
        t_end = (T0 + P * 2 - 2000.0) * 365.25 * 24 * 3600
        num_points = 500
        t_array = np.linspace(t_start, t_end, num_points)
        x_full, y_full, _ = calculate_orbit_positions(t_array)
        
        # Plot full orbit trail
        ax.plot(x_full, y_full, 'b-', alpha=0.7, linewidth=2, label='Calculated Orbit')
        
        # Mark pericenter (closest approach)
        t_peri = T0_seconds
        x_peri, y_peri, _ = calculate_orbit_positions(np.array([t_peri]))
        ax.scatter([x_peri[0]], [y_peri[0]], 
                  c='green', s=150, marker='^', label='Pericenter (closest)', zorder=5)
        
        # Mark apocenter (farthest point)
        t_apo = T0_seconds + P_seconds / 2
        x_apo, y_apo, _ = calculate_orbit_positions(np.array([t_apo]))
        ax.scatter([x_apo[0]], [y_apo[0]], 
                  c='blue', s=150, marker='v', label='Apocenter (farthest)', zorder=5)
    
    # Plot observational data with error bars (if showing table data)
    if (st.session_state.show_table_data or st.session_state.show_table_data_2) and obs_data is not None:
        # Account for jitter by reducing error bars (multiply by 0.5)
        ra_err_plot = obs_data['ra_err'].values * 0.5
        dec_err_plot = obs_data['dec_err'].values * 0.5
        
        # Label for observations
        if use_galactic:
            label_text = 'GRAVITY Observations (Galactic)'
        else:
            label_text = 'GRAVITY Observations'
        
        ax.errorbar(obs_data['ra'].values, obs_data['dec'].values,
                   xerr=ra_err_plot, yerr=dec_err_plot,
                   fmt='o', color='blue', markersize=6,
                   capsize=2, capthick=1.0, elinewidth=1.0,
                   label=label_text, zorder=4)
    
    # Plot black hole center
    ax.scatter([0], [0], c='black', s=100, marker='o', 
              label='Black Hole Center (Sgr A*)', zorder=10, edgecolors='black', linewidths=1)
    
    # Set labels and title
    ax.set_xlabel(coord_label_x, fontsize=12)
    ax.set_ylabel(coord_label_y, fontsize=12)
    ax.set_title('S2 Orbit Around the Black Hole at the center of the Milky Way', 
                fontsize=14, fontweight='bold')
    
    # Set axis limits based on which view is selected
    if (st.session_state.show_table_data or st.session_state.show_table_data_2) and obs_data is not None:
        # For table data: use limits based on actual data points with padding for error bars
        ra_min = (obs_data['ra'].values - obs_data['ra_err'].values * 0.5).min()
        ra_max = (obs_data['ra'].values + obs_data['ra_err'].values * 0.5).max()
        dec_min = (obs_data['dec'].values - obs_data['dec_err'].values * 0.5).min()
        dec_max = (obs_data['dec'].values + obs_data['dec_err'].values * 0.5).max()
        
        # Add padding (30% on each side for better visibility)
        ra_range = ra_max - ra_min
        dec_range = dec_max - dec_min
        ra_padding = ra_range * 0.3 if ra_range > 0 else 0.01
        dec_padding = dec_range * 0.3 if dec_range > 0 else 0.01
        
        # Use the larger range for both axes to maintain aspect ratio
        max_data_range = max(ra_range + 2*ra_padding, dec_range + 2*dec_padding, 0.02)  # Minimum 0.02 arcsec
        center_ra = (ra_max + ra_min) / 2
        center_dec = (dec_max + dec_min) / 2
        
        # Set limits normally (will be inverted below)
        ax.set_xlim(center_ra - max_data_range/2, center_ra + max_data_range/2)
        ax.set_ylim(center_dec - max_data_range/2, center_dec + max_data_range/2)
    else:
        # For calculated orbit: use limits based on orbital parameters
        max_range = a * (1 + e) * 1.2  # Slightly larger than apocenter
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
    
    ax.invert_xaxis()  # Flip x-axis: positive to negative
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0, 1),
             borderpad=1.0, labelspacing=1.2, columnspacing=1.5,
             frameon=True, fancybox=True, shadow=True)
    
    st.pyplot(fig, use_container_width=True)
    
    # Show message if no data is displayed
    if (st.session_state.show_table_data or st.session_state.show_table_data_2) and obs_data is None:
        st.warning("No observational data found. Please ensure gravity_fits_table.csv exists.")
    
    # About Star S2 section (moved to left column)
    st.header("About Star S2 (S0-2)")
    st.markdown("""
    Star S2 (also known as S0-2) is one of the closest stars to the supermassive black hole 
    at the center of our Milky Way galaxy. Its highly elliptical orbit has been carefully monitored for decades, 
    providing crucial evidence for the existence of a supermassive black hole.
    
    **Key Facts:**
    - **Orbital Period:** {P:.2f} years
    - **Closest Approach:** {peri_distance_AU:.1f} AU from Sgr A* (at pericenter)
    - **Farthest Distance:** {apo_distance_AU:.1f} AU from Sgr A* (at apocenter)
    - **Eccentricity:** {e:.5f}
    - **Last Pericenter Passage:** May 2018 (T₀ ≈ {T0:.2f})
    
    The orbit of S2 has been used to:
    1. Measure the mass of Sgr A* (~4 million solar masses)
    2. Test predictions of general relativity (gravitational redshift, orbital precession)
    3. Constrain the distance to the Galactic Center
    """.format(
        P=P,
        peri_distance_AU=(a * (1 - e) * arcsec_to_rad * distance_gc_m) / (1.496e11),  # Convert to AU
        apo_distance_AU=(a * (1 + e) * arcsec_to_rad * distance_gc_m) / (1.496e11),  # Convert to AU
        e=e,
        T0=T0
    ))

with col_info:
    # Example Orbit image
    # Try to load and display the image if it exists
    image_paths = ['Figure1.png', 'example_orbit.png', 'example_orbit.jpg', 'example_orbit.jpeg', 's2_orbit_example.png', 's2_orbit_example.jpg']
    image_found = False
    for img_path in image_paths:
        if os.path.exists(img_path):
            # Center the title over the image
            st.markdown('<p style="font-size:32px; font-weight:bold; text-align:center; margin-bottom:0.5em;">Example Orbit</p>', unsafe_allow_html=True)
            # Center the image with larger size
            col_img1, col_img2, col_img3 = st.columns([0.5, 3, 0.5])
            with col_img2:
                st.image(img_path)
            image_found = True
            break
    
    if not image_found:
        # Placeholder if image not found
        st.markdown('<p style="font-size:30px; font-weight:bold; text-align:center; margin-bottom:0.5em;">Example Orbit</p>', unsafe_allow_html=True)
        st.info("Image file not found. Please add the example orbit image to the project directory.")
    
    st.markdown('<p style="font-size:10px; font-style:italic; text-align:center; margin-top:0.25em; margin-bottom:1.5em;">Credit: DOI:10.1007/978-3-642-74391-7_4</p>', unsafe_allow_html=True)
    
    # Orbital Information section
    st.markdown("**Orbital Information**")
    
    # Calculate some orbital properties
    b_arcsec = a * np.sqrt(1 - e**2)  # Semi-minor axis
    peri_distance = a * (1 - e)  # Pericenter distance
    apo_distance = a * (1 + e)   # Apocenter distance
    
    # Compact table format
    orbital_data = {
        'Property': [
            'Semi-major axis',
            'Semi-minor axis',
            'Eccentricity',
            'Orbital period',
            'Pericenter distance',
            'Apocenter distance'
        ],
        'Value': [
            f"{a:.4f} arcsec",
            f"{b_arcsec:.4f} arcsec",
            f"{e:.5f}",
            f"{P:.2f} years",
            f"{peri_distance:.4f} arcsec",
            f"{apo_distance:.4f} arcsec"
        ]
    }
    df_orbital = pd.DataFrame(orbital_data)
    st.dataframe(df_orbital, use_container_width=True, hide_index=True)
    
    # Orbital Elements in compact format
    st.markdown("**Orbital Elements:**")
    st.markdown(f"- **Inclination (i):** {i_deg:.2f}°  |  **Ω (longitude of ascending node):** {Omega_deg:.2f}°")
    st.markdown(f"- **ω (argument of periapsis):** {omega_deg:.2f}°  |  **T₀ (pericenter passage):** {T0:.3f} year")

# Footer
st.caption("Data sources: GRAVITY Collaboration, ESO observations, and published orbital element catalogs")
