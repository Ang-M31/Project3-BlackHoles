"""
Streamlit App: Visualization of Star S2's Orbit Around the Black Hole at the center of the Milky Way
This app displays GRAVITY Collaboration observational data with error bars and calculated orbital model.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, G
from scipy.optimize import minimize, curve_fit
import pandas as pd
import os
import re
import urllib.request
from io import StringIO

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

@st.cache_data
def load_cds_data(use_galactic=False):
    """
    Load observational data from CDS archive URL.
    Data source: https://cdsarc.cds.unistra.fr/ftp/J/ApJ/707/L114/table1.dat
    
    Fixed-width format:
    Columns 1-8:   Year (F8.3)
    Columns 10-14: oRA in mas (F5.1)
    Columns 16-19: e_oRA in mas (F4.1)
    Columns 21-25: oDE in mas (F5.1)
    Columns 27-30: e_oDE in mas (F4.1)
    Columns 32-37: Telescope (A6)
    Columns 39-46: Ep.V year (F8.3, optional)
    Columns 48-52: VLSR in km/s (I5, optional)
    Columns 54-56: e_VLSR in km/s (I3, optional)
    Columns 58-61: Tel.V (A4, optional)
    
    Returns DataFrame with ra, dec, ra_err, dec_err in arcseconds.
    If use_galactic=True, converts to galactic coordinates (l, b).
    """
    url = "https://cdsarc.cds.unistra.fr/ftp/J/ApJ/707/L114/table1.dat"
    
    try:
        # Fetch data from URL
        with urllib.request.urlopen(url) as response:
            data_text = response.read().decode('utf-8')
        
        # Parse fixed-width format
        ra_values = []
        ra_errors = []
        dec_values = []
        dec_errors = []
        years = []
        telescopes = []
        
        for line in data_text.strip().split('\n'):
            line = line.rstrip()
            if len(line) < 30:  # Skip lines that are too short
                continue
            
            try:
                # Parse fixed-width columns
                year_str = line[0:8].strip()
                oRA_str = line[9:14].strip()
                e_oRA_str = line[15:19].strip()
                oDE_str = line[20:25].strip()
                e_oDE_str = line[26:30].strip()
                tel_str = line[31:37].strip() if len(line) > 31 else ""
                
                # Skip if essential data is missing
                if not year_str or not oRA_str or not oDE_str:
                    continue
                
                # Convert to float
                year = float(year_str)
                oRA_mas = float(oRA_str)
                oDE_mas = float(oDE_str)
                
                # Parse errors (may be empty or whitespace)
                e_oRA_mas = float(e_oRA_str) if e_oRA_str.strip() else 0.0
                e_oDE_mas = float(e_oDE_str) if e_oDE_str.strip() else 0.0
                
                # Convert mas to arcseconds
                ra_values.append(oRA_mas / 1000.0)
                ra_errors.append(e_oRA_mas / 1000.0)
                dec_values.append(oDE_mas / 1000.0)
                dec_errors.append(e_oDE_mas / 1000.0)
                years.append(year)
                telescopes.append(tel_str)
                
            except (ValueError, IndexError) as e:
                # Skip lines that can't be parsed
                continue
        
        if len(ra_values) == 0:
            return None
        
        # Create DataFrame
        obs_data = pd.DataFrame({
            'year': years,
            'ra': ra_values,
            'dec': dec_values,
            'ra_err': ra_errors,
            'dec_err': dec_errors,
            'telescope': telescopes
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
                'year': years,
                'ra': l_values,  # Reuse 'ra' column for galactic l
                'dec': b_values,  # Reuse 'dec' column for galactic b
                'ra_err': l_errors,  # Reuse 'ra_err' for l_err
                'dec_err': b_errors,  # Reuse 'dec_err' for b_err
                'telescope': telescopes
            })
        
        return obs_data
    
    except Exception as e:
        st.warning(f"Error loading CDS data: {e}")
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
if 'show_table_data_3' not in st.session_state:
    st.session_state.show_table_data_3 = False

# Display options buttons
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

with col_btn1:
    if st.button("Show Calculated Orbit", use_container_width=True, type="primary" if st.session_state.show_calculated_orbit else "secondary"):
        st.session_state.show_calculated_orbit = True
        st.session_state.show_table_data = False
        st.session_state.show_table_data_2 = False
        st.session_state.show_table_data_3 = False
        st.rerun()

with col_btn2:
    if st.button("All Data", use_container_width=True, type="primary" if st.session_state.show_table_data else "secondary"):
        st.session_state.show_table_data = True
        st.session_state.show_calculated_orbit = False
        st.session_state.show_table_data_2 = False
        st.session_state.show_table_data_3 = False
        st.rerun()

with col_btn3:
    if st.button("Initial Data Model", use_container_width=True, type="primary" if st.session_state.show_table_data_2 else "secondary"):
        st.session_state.show_table_data_2 = True
        st.session_state.show_calculated_orbit = False
        st.session_state.show_table_data = False
        st.session_state.show_table_data_3 = False
        st.rerun()

with col_btn4:
    if st.button("Adjusted Model", use_container_width=True, type="primary" if st.session_state.show_table_data_3 else "secondary"):
        st.session_state.show_table_data_3 = True
        st.session_state.show_calculated_orbit = False
        st.session_state.show_table_data = False
        st.session_state.show_table_data_2 = False
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

# Helper: black hole mass from Kepler's third law (S2-only)
def compute_bh_mass_solar(a_arcsec_val, P_year_val, dist_kpc_val):
    a_m_val = a_arcsec_val * arcsec_to_rad * (dist_kpc_val * 1000 * pc_to_m)
    P_sec_val = P_year_val * 365.25 * 24 * 3600
    M_kg = 4 * np.pi**2 * a_m_val**3 / (G * P_sec_val**2)
    return M_kg / M_sun

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
    # Initial guess: E ≈ M for small e, or use M + e*sin(M) for better initial guess
    E = M.copy()
    if e > 0.8:  # For high eccentricity, use better initial guess
        E = M + np.sign(np.sin(M)) * e
    
    # Newton's method: E_new = E_old - f(E)/f'(E)
    # where f(E) = E - e*sin(E) - M = 0
    # and f'(E) = 1 - e*cos(E)
    for _ in range(50):  # More iterations for convergence
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        E = E - f / f_prime
        
        # Check for convergence (optional, but helps for edge cases)
        if np.max(np.abs(f)) < 1e-10:
            break
    
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

def calculate_orbit_from_params(times, a_arcsec, e, i_deg, Omega_deg, omega_deg, T0_year, P_year, distance_gc_kpc, use_damped_solver=False):
    """
    Calculate orbital positions from parameters in EQUATORIAL (ICRS) coordinates.
    This function is designed to work with curve_fit.
    
    Parameters:
    - times: array of times in seconds since year 2000
    - a_arcsec: semi-major axis in arcseconds
    - e: eccentricity
    - i_deg: inclination in degrees (equatorial frame)
    - Omega_deg: longitude of ascending node in degrees (equatorial frame)
    - omega_deg: argument of periapsis in degrees (equatorial frame)
    - T0_year: time of pericenter passage in years
    - P_year: orbital period in years
    - distance_gc_kpc: distance to Galactic Center in kiloparsecs
    
    Returns:
    - positions: flattened array [x1, y1, x2, y2, ...] in arcseconds (equatorial coordinates)
    """
    # Convert to radians
    i = np.radians(i_deg)
    Omega = np.radians(Omega_deg)
    omega = np.radians(omega_deg)
    
    # Convert to physical units
    pc_to_m = 3.085677581e16  # meters
    arcsec_to_rad = np.pi / (180 * 3600)
    distance_gc_m = distance_gc_kpc * 1000 * pc_to_m
    a_rad = a_arcsec * arcsec_to_rad
    a_m = a_rad * distance_gc_m
    
    # Calculate mean motion
    P_seconds = P_year * 365.25 * 24 * 3600
    n = 2 * np.pi / P_seconds
    T0_seconds = (T0_year - 2000.0) * 365.25 * 24 * 3600
    
    # Mean anomaly
    M = n * (times - T0_seconds)
    
    # Solve Kepler's equation: E = M + e*sin(E)
    # Initial guess: E ≈ M for small e, or use M + e*sin(M) for better initial guess
    E = M.copy()
    if e > 0.8:  # For high eccentricity, use better initial guess
        E = M + np.sign(np.sin(M)) * e
    
    if use_damped_solver:
        # Damped Newton: guard against tiny f' and overshoot
        max_iter = 50
        tol = 1e-10
        for _ in range(max_iter):
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            # Avoid division by very small derivative
            small = np.abs(f_prime) < 1e-12
            f_prime_safe = np.where(small, np.sign(f_prime) * 1e-12, f_prime)
            step = f / f_prime_safe
            step = np.clip(step, -0.5, 0.5)  # damping
            E = E - step
            if np.max(np.abs(f)) < tol:
                break
    else:
        # Standard Newton
        for _ in range(50):  # More iterations for convergence
            f = E - e * np.sin(E) - M
            f_prime = 1 - e * np.cos(E)
            E = E - f / f_prime
            if np.max(np.abs(f)) < 1e-10:
                break
    
    # True anomaly
    nu = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )
    
    # Distance from focus
    r = a_m * (1 - e**2) / (1 + e * np.cos(nu))
    
    # Position in orbital plane
    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = np.zeros_like(x_orb)
    
    # Rotation matrices
    R_omega = np.array([
        [np.cos(omega), -np.sin(omega), 0],
        [np.sin(omega), np.cos(omega), 0],
        [0, 0, 1]
    ])
    
    R_i = np.array([
        [1, 0, 0],
        [0, np.cos(i), -np.sin(i)],
        [0, np.sin(i), np.cos(i)]
    ])
    
    R_Omega = np.array([
        [np.cos(Omega), -np.sin(Omega), 0],
        [np.sin(Omega), np.cos(Omega), 0],
        [0, 0, 1]
    ])
    
    R = R_Omega @ R_i @ R_omega
    
    # Transform positions
    pos_orb = np.vstack([x_orb, y_orb, z_orb])
    pos_3d = R @ pos_orb
    
    # Project onto sky plane
    x_sky = pos_3d[0, :] / distance_gc_m / arcsec_to_rad
    y_sky = pos_3d[1, :] / distance_gc_m / arcsec_to_rad
    
    # Return flattened array for curve_fit
    return np.concatenate([x_sky, y_sky])

def fit_orbit_parameters(obs_data, initial_params, use_gaussian=False):
    """
    Fit orbital parameters to observational data using scipy.optimize.curve_fit.
    Uses Gaussian (normal) error distribution assumptions.
    
    Parameters:
    - obs_data: DataFrame with columns 'year', 'ra', 'dec', 'ra_err', 'dec_err', 'telescope'
    - initial_params: dict with initial orbital parameters
    - use_gaussian: if True, explicitly enforce Gaussian error assumptions (default: False, but curve_fit uses Gaussian by default)
    
    Returns:
    - fitted_params: dict with fitted orbital parameters
    - residuals_ra: residuals in RA (arcsec)
    - residuals_dec: residuals in Dec (arcsec)
    - fitted_positions: tuple of (x_fit, y_fit) arrays
    """
    # Convert observation years to seconds since year 2000
    obs_times = (obs_data['year'].values - 2000.0) * 365.25 * 24 * 3600
    
    # Get observed positions
    x_obs = obs_data['ra'].values
    y_obs = obs_data['dec'].values
    
    # Get observation errors (arcsec)
    x_err = obs_data['ra_err'].values
    y_err = obs_data['dec_err'].values
    
    # For Adjusted (Gaussian) model, add an error floor ("jitter") in quadrature
    # to avoid unrealistically small sigmas dominating the fit.
    if use_gaussian:
        jitter_arcsec = 0.0005  # ~0.5 mas
        x_err_eff = np.sqrt(x_err**2 + jitter_arcsec**2)
        y_err_eff = np.sqrt(y_err**2 + jitter_arcsec**2)
    else:
        x_err_eff = x_err
        y_err_eff = y_err
    
    # Create weights: higher weight for Keck data
    weights = np.ones(len(obs_data))
    keck_mask = obs_data['telescope'].str.strip().str.upper() == 'KECK'
    # Prioritize Keck more strongly in Adjusted (Gaussian) fits
    keck_weight = 5.0 if use_gaussian else 2.0
    weights[keck_mask] = keck_weight  # e.g., 5x for Adjusted, 2x for Initial
    # Apply weights to errors (smaller weighted error = higher weight in fit)
    x_err_weighted = x_err_eff / weights
    y_err_weighted = y_err_eff / weights
    
    # Initial parameter values (used as starting point)
    p0 = [
        initial_params['a'],           # a_arcsec
        initial_params['e'],           # e
        initial_params['i'],           # i_deg
        initial_params['Omega'],       # Omega_deg
        initial_params['omega'],       # omega_deg
        initial_params['T0'],          # T0_year
        initial_params['P'],           # P_year
        initial_params['distance_gc']   # distance_gc_kpc
    ]
    
    if use_gaussian:
        # Adjusted Model: data-driven, broad physical bounds (not tied to published S2)
        year_min = float(np.min(obs_data['year'].values))
        year_max = float(np.max(obs_data['year'].values))
        T0_lower = year_min - 5.0
        # Allow T0 to explore beyond the last data point, up to ~2020
        T0_upper = max(year_max + 5.0, 2020.0)
        bounds = (
            [
                0.01,    # a_arcsec: small but positive
                0.0,     # e
                0.0,     # i_deg
                0.0,     # Omega_deg
                0.0,     # omega_deg
                T0_lower,
                5.0,     # P_year
                7.8      # distance_gc_kpc (lower bound)
            ],
            [
                0.5,     # a_arcsec
                0.999,   # e
                180.0,   # i_deg
                360.0,   # Omega_deg
                360.0,   # omega_deg
                T0_upper,
                40.0,    # P_year
                8.6      # distance_gc_kpc (upper bound)
            ]
        )
        # Clamp starting guess into bounds to avoid "x0 is infeasible"
        p0 = [float(np.clip(val, low, high)) for val, low, high in zip(p0, bounds[0], bounds[1])]
    else:
        # Initial Model: keep the original, S2-published-constrained bounds
        bounds = (
            [0.05, 0.5, 100, 200, 0, 2010, 10, 7],      # lower bounds
            [0.3, 0.99, 180, 250, 100, 2025, 20, 9]      # upper bounds
        )
    
    # Fit RA and Dec simultaneously by concatenating both coordinates.
    # For Adjusted Model, perform staged fitting to stabilize the solution.
    def orbit_model_ra_dec(times, a, e, i, Om, om, T0, P, d):
        return calculate_orbit_from_params(times, a, e, i, Om, om, T0, P, d, use_damped_solver=use_gaussian)
    
    try:
        obs_concat = np.concatenate([x_obs, y_obs])
        sigma_concat = np.concatenate([x_err_weighted, y_err_weighted])
        
        if use_gaussian:
            # Helper to clamp any p0 to bounds
            def clamp_to_bounds(p_guess):
                return [float(np.clip(val, low, high)) for val, low, high in zip(p_guess, bounds[0], bounds[1])]
            
            # Stage 1: fit geometry (a, e, P, T0), keep orientation fixed
            def model_stage1(times, a, e, P, T0):
                return calculate_orbit_from_params(
                    times, a, e,
                    p0[2], p0[3], p0[4],  # fixed i, Omega, omega from initial guess
                    T0, P,
                    p0[7],  # distance
                    use_damped_solver=True
                )
            p0_stage1 = [p0[0], p0[1], p0[6], p0[5]]
            bounds_stage1 = (
                [bounds[0][0], bounds[0][1], bounds[0][6], bounds[0][5]],
                [bounds[1][0], bounds[1][1], bounds[1][6], bounds[1][5]],
            )
            p0_stage1 = [float(np.clip(v, lo, hi)) for v, lo, hi in zip(p0_stage1, bounds_stage1[0], bounds_stage1[1])]
            popt_stage1, _ = curve_fit(
                model_stage1, obs_times, obs_concat,
                p0=p0_stage1, sigma=sigma_concat, bounds=bounds_stage1,
                maxfev=6000, method='trf', absolute_sigma=True
            )
            
            # Stage 2: fit orientation (i, Omega, omega), keep geometry from stage 1
            geom_a, geom_e, geom_P, geom_T0 = popt_stage1
            def model_stage2(times, i_val, Om_val, om_val):
                return calculate_orbit_from_params(
                    times, geom_a, geom_e,
                    i_val, Om_val, om_val,
                    geom_T0, geom_P,
                    p0[7],
                    use_damped_solver=True
                )
            p0_stage2 = [p0[2], p0[3], p0[4]]
            bounds_stage2 = (
                [bounds[0][2], bounds[0][3], bounds[0][4]],
                [bounds[1][2], bounds[1][3], bounds[1][4]],
            )
            p0_stage2 = [float(np.clip(v, lo, hi)) for v, lo, hi in zip(p0_stage2, bounds_stage2[0], bounds_stage2[1])]
            popt_stage2, _ = curve_fit(
                model_stage2, obs_times, obs_concat,
                p0=p0_stage2, sigma=sigma_concat, bounds=bounds_stage2,
                maxfev=6000, method='trf', absolute_sigma=True
            )
            
            # Stage 3: joint refinement of all parameters, seeded from stages 1+2
            p0_joint = [
                geom_a, geom_e,
                popt_stage2[0], popt_stage2[1], popt_stage2[2],
                geom_T0, geom_P,
                p0[7]
            ]
            p0_joint = clamp_to_bounds(p0_joint)
            popt, _ = curve_fit(
                orbit_model_ra_dec,
                obs_times,
                obs_concat,
                p0=p0_joint,
                sigma=sigma_concat,
                bounds=bounds,
                maxfev=10000,
                method='trf',
                absolute_sigma=True
            )
        else:
            popt, _ = curve_fit(
                orbit_model_ra_dec,
                obs_times,
                obs_concat,
                p0=p0,
                sigma=sigma_concat,
                bounds=bounds,
                maxfev=8000,
                method='trf',
                absolute_sigma=True
            )
        
        # Extract fitted parameters
        fitted_params = {
            'a': popt[0],
            'e': popt[1],
            'i': popt[2],
            'Omega': popt[3],
            'omega': popt[4],
            'T0': popt[5],
            'P': popt[6],
            'distance_gc': popt[7],
            'M_bh': initial_params['M_bh']  # Keep black hole mass from initial
        }
        
        # Calculate fitted positions using averaged parameters
        pos_fit = calculate_orbit_from_params(
            obs_times, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], popt[6], popt[7]
        )
        n = len(pos_fit) // 2
        x_fit = pos_fit[:n]
        y_fit = pos_fit[n:]
        
    except Exception as e:
        # If fitting fails, use initial parameters
        st.warning(f"Orbit fitting encountered an issue: {e}. Using initial parameters.")
        fitted_params = initial_params.copy()
        x_fit, y_fit, _ = calculate_orbit_positions(obs_times)
    
    # Calculate chi-squared residuals (normalized by errors)
    # Chi-squared residual = (observed - expected) / error
    # Avoid division by zero by using a small minimum error
    x_err_safe = np.where(x_err_eff > 0, x_err_eff,
                          np.median(x_err_eff[x_err_eff > 0]) if np.any(x_err_eff > 0) else 0.001)
    y_err_safe = np.where(y_err_eff > 0, y_err_eff,
                          np.median(y_err_eff[y_err_eff > 0]) if np.any(y_err_eff > 0) else 0.001)
    
    residuals_ra = (x_obs - x_fit) / x_err_safe
    residuals_dec = (y_obs - y_fit) / y_err_safe
    
    # ---------- Diagnostics summary string for Adjusted model ----------
    diagnostics_md = None
    if use_gaussian:
        try:
            lines = []
            lines.append("### Adjusted Model Fit Diagnostics")
            lines.append("")
            lines.append("**Fitted parameters**")
            param_keys = ['a', 'e', 'i', 'Omega', 'omega', 'T0', 'P', 'distance_gc']
            pretty_names = ['a_arcsec', 'e', 'i_deg', 'Omega_deg', 'omega_deg', 'T0_year', 'P_year', 'distance_gc_kpc']
            for key, pretty in zip(param_keys, pretty_names):
                if key in fitted_params:
                    val = fitted_params[key]
                    try:
                        lines.append(f"- **{pretty}** ({key}): {val:.6f}")
                    except Exception:
                        lines.append(f"- **{pretty}** ({key}): {val}")

            # Boundary check
            if 'bounds' in locals():
                lines.append("")
                lines.append("**Boundary check**")
                lower_bounds, upper_bounds = bounds
                for i, (key, pretty) in enumerate(zip(param_keys, pretty_names)):
                    if key in fitted_params:
                        val = fitted_params[key]
                        lower, upper = lower_bounds[i], upper_bounds[i]
                        width = upper - lower
                        if width > 0:
                            at_lower = abs(val - lower) < 0.01 * width
                            at_upper = abs(val - upper) < 0.01 * width
                            if at_lower or at_upper:
                                where = "LOWER" if at_lower else "UPPER"
                                lines.append(f"- ⚠️ **{pretty}** = {val:.4f} is near **{where}** bound [{lower}, {upper}]")

            # Per-telescope residuals (RA/Dec combined norm)
            if 'telescope' in obs_data.columns:
                lines.append("")
                lines.append("**Residuals by telescope**")
                for telescope in obs_data['telescope'].unique():
                    mask = obs_data['telescope'] == telescope
                    ra_res = residuals_ra[mask]
                    dec_res = residuals_dec[mask]
                    if ra_res.size > 0:
                        rms = np.sqrt(np.mean(ra_res**2 + dec_res**2))
                        lines.append(f"- **{telescope}**: RMS = {rms:.2f}σ, N = {mask.sum()}")

            diagnostics_md = "\n".join(lines)
        except Exception:
            diagnostics_md = None
    # -------------------------------------------------------------------------
    
    return fitted_params, residuals_ra, residuals_dec, (x_fit, y_fit), diagnostics_md

def fit_orbit_to_data(obs_data, initial_params, keck_weight=2.0):
    """
    Fit orbital parameters to observational data with higher weight for Keck telescope.
    
    Parameters:
    - obs_data: DataFrame with columns 'year', 'ra', 'dec', 'ra_err', 'dec_err', 'telescope'
    - initial_params: dict with initial orbital parameters
    - keck_weight: weight multiplier for Keck data (default 2.0)
    
    Returns:
    - fitted_params: dict with fitted orbital parameters
    """
    # Create weights: higher weight for Keck data
    weights = np.ones(len(obs_data))
    keck_mask = obs_data['telescope'].str.strip().str.upper() == 'KECK'
    weights[keck_mask] = keck_weight
    
    # Also weight by inverse of error (smaller errors get higher weight)
    # Normalize errors
    ra_err_norm = obs_data['ra_err'].values
    dec_err_norm = obs_data['dec_err'].values
    total_err = np.sqrt(ra_err_norm**2 + dec_err_norm**2)
    # Avoid division by zero
    total_err[total_err == 0] = np.median(total_err[total_err > 0]) if np.any(total_err > 0) else 1.0
    error_weights = 1.0 / total_err
    error_weights = error_weights / np.max(error_weights)  # Normalize to max 1
    
    # Combine telescope weights and error weights
    final_weights = weights * error_weights
    
    # Convert observation years to seconds since year 2000
    obs_times = (obs_data['year'].values - 2000.0) * 365.25 * 24 * 3600
    
    # Use initial parameters as starting point
    # For simplicity, we'll fit a simplified model focusing on the orbit shape
    # We'll use the existing parameters but adjust based on data fit
    
    # Calculate positions using initial parameters
    x_obs = obs_data['ra'].values
    y_obs = obs_data['dec'].values
    
    # For now, return the initial parameters (full orbital fitting is complex)
    # In a more sophisticated implementation, we would optimize parameters
    # Here we'll use the published parameters as they're already well-fitted
    return initial_params

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
        obs_data = load_cds_data(use_galactic=False)
    elif st.session_state.show_table_data_2:
        obs_data = load_cds_data(use_galactic=False)
        use_galactic = False
        coord_label_x = 'Right Ascension Difference (")'
        coord_label_y = 'Declination Difference (")'
    elif st.session_state.show_table_data_3:
        obs_data = load_cds_data(use_galactic=False)
        use_galactic = False
        coord_label_x = 'Right Ascension Difference (")'
        coord_label_y = 'Declination Difference (")'
    
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
    
    # For "Initial Data Model" and "Adjusted Model", fit orbit to data and plot fitted orbit
    # Keep them completely separate to avoid any cross-contamination
    fitted_params_initial = None
    residuals_ra_initial = None
    residuals_dec_initial = None
    fitted_params_adjusted = None
    residuals_ra_adjusted = None
    residuals_dec_adjusted = None
    
    # "Initial Data Model" - independent fitting
    if st.session_state.show_table_data_2 and obs_data is not None:
        fitted_params_initial, residuals_ra_initial, residuals_dec_initial, (x_fit_obs_initial, y_fit_obs_initial), _ = fit_orbit_parameters(obs_data, params)
        
        # Calculate full fitted orbit for visualization
        year_min = obs_data['year'].min()
        year_max = obs_data['year'].max()
        # For Adjusted Model, span exactly the observed time range (e.g., 1992–2009)
        t_start = (year_min - 2000.0) * 365.25 * 24 * 3600
        t_end = (year_max - 2000.0) * 365.25 * 24 * 3600
        num_points = 500
        t_array = np.linspace(t_start, t_end, num_points)
        
        # Calculate positions using fitted parameters
        pos_fit_full_initial = calculate_orbit_from_params(
            t_array, 
            fitted_params_initial['a'], fitted_params_initial['e'], fitted_params_initial['i'],
            fitted_params_initial['Omega'], fitted_params_initial['omega'], fitted_params_initial['T0'],
            fitted_params_initial['P'], fitted_params_initial['distance_gc']
        )
        n_full_initial = len(pos_fit_full_initial) // 2
        x_fit_full_initial = pos_fit_full_initial[:n_full_initial]
        y_fit_full_initial = pos_fit_full_initial[n_full_initial:]
        
        # Plot fitted orbit line (using purple color, not used by any telescope)
        ax.plot(x_fit_full_initial, y_fit_full_initial, 'purple', alpha=0.7, linewidth=2, 
               label='Fitted Orbit Model', zorder=3)
    
    # "Adjusted Model" - independent fitting with Gaussian error assumptions
    diagnostics_adjusted = None
    if st.session_state.show_table_data_3 and obs_data is not None:
        fitted_params_adjusted, residuals_ra_adjusted, residuals_dec_adjusted, (x_fit_obs_adjusted, y_fit_obs_adjusted), diagnostics_adjusted = fit_orbit_parameters(obs_data, params, use_gaussian=True)
        
        # Calculate full fitted orbit for visualization
        year_min = obs_data['year'].min()
        year_max = obs_data['year'].max()
        t_start = (year_min - 2 - 2000.0) * 365.25 * 24 * 3600
        t_end = (year_max + 2 - 2000.0) * 365.25 * 24 * 3600
        num_points = 500
        t_array = np.linspace(t_start, t_end, num_points)
        
        # Calculate positions using fitted parameters
        pos_fit_full_adjusted = calculate_orbit_from_params(
            t_array, 
            fitted_params_adjusted['a'], fitted_params_adjusted['e'], fitted_params_adjusted['i'],
            fitted_params_adjusted['Omega'], fitted_params_adjusted['omega'], fitted_params_adjusted['T0'],
            fitted_params_adjusted['P'], fitted_params_adjusted['distance_gc']
        )
        n_full_adjusted = len(pos_fit_full_adjusted) // 2
        x_fit_full_adjusted = pos_fit_full_adjusted[:n_full_adjusted]
        y_fit_full_adjusted = pos_fit_full_adjusted[n_full_adjusted:]
        
        # Plot fitted orbit line (using purple color, not used by any telescope)
        ax.plot(x_fit_full_adjusted, y_fit_full_adjusted, 'purple', alpha=0.7, linewidth=2, 
               label='Fitted Orbit Model', zorder=3)
    
    # Plot observational data with error bars (if showing table data)
    if (st.session_state.show_table_data or st.session_state.show_table_data_2 or st.session_state.show_table_data_3) and obs_data is not None:
        # Account for jitter by reducing error bars (multiply by 0.5)
        ra_err_plot = obs_data['ra_err'].values * 0.5
        dec_err_plot = obs_data['dec_err'].values * 0.5
        
        # For "All Data", "Initial Data Model", and "Adjusted Model", color code by telescope
        if st.session_state.show_table_data or st.session_state.show_table_data_2 or st.session_state.show_table_data_3:
            # Define color map for telescopes
            telescope_colors = {
                'NTT': 'blue',
                'VLT': 'red',
                'Keck': 'green',
                'GEMINI': 'orange'
            }
            
            # Get unique telescopes in the data
            unique_telescopes = obs_data['telescope'].unique()
            
            # Plot each telescope with different color
            for telescope in unique_telescopes:
                if telescope and telescope.strip():
                    telescope_clean = telescope.strip()
                    mask = obs_data['telescope'] == telescope
                    color = telescope_colors.get(telescope_clean, 'gray')
                    
                    # Get data for this telescope using .loc for consistent indexing
                    tel_data = obs_data.loc[mask]
                    tel_ra_err = tel_data['ra_err'].values * 0.5
                    tel_dec_err = tel_data['dec_err'].values * 0.5
                    
                    ax.errorbar(tel_data['ra'].values, 
                              tel_data['dec'].values,
                              xerr=tel_ra_err, 
                              yerr=tel_dec_err,
                              fmt='o', color=color, markersize=6,
                              capsize=2, capthick=1.0, elinewidth=1.0,
                              label=telescope_clean, zorder=4)
        else:
            # For Version 3, use single color
            # Label for observations
            if use_galactic:
                label_text = 'Observations (Galactic)'
            else:
                label_text = 'Observations (ICRS)'
            
            # For Version 3, use smaller markers
            marker_size = 3
            
            ax.errorbar(obs_data['ra'].values, obs_data['dec'].values,
                       xerr=ra_err_plot, yerr=dec_err_plot,
                       fmt='o', color='blue', markersize=marker_size,
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
    if (st.session_state.show_table_data or st.session_state.show_table_data_2 or st.session_state.show_table_data_3) and obs_data is not None:
        # For table data: use limits based on actual data points with padding for error bars
        # For Version 3, use full error bars instead of 0.5x to get better range
        if st.session_state.show_table_data_3:
            # Use full error bars for Version 3 to ensure proper axis range
            ra_min = (obs_data['ra'].values - obs_data['ra_err'].values).min()
            ra_max = (obs_data['ra'].values + obs_data['ra_err'].values).max()
            dec_min = (obs_data['dec'].values - obs_data['dec_err'].values).min()
            dec_max = (obs_data['dec'].values + obs_data['dec_err'].values).max()
        else:
            # For Version 1 and 2, use reduced error bars (0.5x) as before
            ra_min = (obs_data['ra'].values - obs_data['ra_err'].values * 0.5).min()
            ra_max = (obs_data['ra'].values + obs_data['ra_err'].values * 0.5).max()
            dec_min = (obs_data['dec'].values - obs_data['dec_err'].values * 0.5).min()
            dec_max = (obs_data['dec'].values + obs_data['dec_err'].values * 0.5).max()
        
        # Add padding (30% on each side for better visibility)
        ra_range = ra_max - ra_min
        dec_range = dec_max - dec_min
        padding_factor = 0.3
        ra_padding = ra_range * padding_factor if ra_range > 0 else 0.01
        dec_padding = dec_range * padding_factor if dec_range > 0 else 0.01
        
        # Use the larger range for both axes to maintain aspect ratio
        min_range = 0.02
        max_data_range = max(ra_range + 2*ra_padding, dec_range + 2*dec_padding, min_range)
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
    
    # Adjust legend position based on view
    if st.session_state.show_table_data or st.session_state.show_table_data_2 or st.session_state.show_table_data_3:
        # For "All Data", "Initial Data Model", and "Adjusted Model" views with multiple telescopes, place legend in lower right
        ax.legend(loc='lower right', fontsize=9, 
                 borderpad=0.8, labelspacing=0.8, columnspacing=1.0,
                 frameon=True, fancybox=True, shadow=True)
    else:
        # For other views, use original position
        ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(0, 1),
                 borderpad=1.0, labelspacing=1.2, columnspacing=1.5,
                 frameon=True, fancybox=True, shadow=True)
    
    st.pyplot(fig, use_container_width=True)
    
    # For "Initial Data Model" and "Adjusted Model", add residuals plot
    # "Initial Data Model" residuals
    if st.session_state.show_table_data_2 and obs_data is not None and residuals_ra_initial is not None:
        # Define color map for telescopes (matching main plot)
        telescope_colors = {
            'NTT': 'blue',
            'VLT': 'red',
            'Keck': 'green',
            'GEMINI': 'orange'
        }
        
        # Create residuals plot (RA, Dec, and combined norm)
        fig_res, (ax_res_ra, ax_res_dec, ax_res_norm) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # Get unique telescopes in the data
        unique_telescopes = obs_data['telescope'].unique()
        
        # Plot RA residuals for each telescope with matching colors
        for telescope in unique_telescopes:
            if telescope and telescope.strip():
                telescope_clean = telescope.strip()
                mask = obs_data['telescope'] == telescope
                color = telescope_colors.get(telescope_clean, 'gray')
                
                ax_res_ra.scatter(obs_data.loc[mask, 'year'].values, 
                                 residuals_ra_initial[mask], 
                                 color=color, s=30, alpha=0.6, 
                                 label=f'{telescope_clean} RA', zorder=2)
        
        ax_res_ra.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_res_ra.set_ylabel('RA Chi-Squared Residuals (σ)', fontsize=11)
        ax_res_ra.set_title('Orbital Fit Chi-Squared Residuals', fontsize=12, fontweight='bold')
        ax_res_ra.grid(True, alpha=0.3)
        ax_res_ra.legend(loc='upper right', fontsize=9)
        
        # Plot Dec residuals for each telescope with matching colors
        for telescope in unique_telescopes:
            if telescope and telescope.strip():
                telescope_clean = telescope.strip()
                mask = obs_data['telescope'] == telescope
                color = telescope_colors.get(telescope_clean, 'gray')
                
                ax_res_dec.scatter(obs_data.loc[mask, 'year'].values, 
                                 residuals_dec_initial[mask], 
                                 color=color, s=30, alpha=0.6, 
                                 label=f'{telescope_clean} Dec', zorder=2)
        
        ax_res_dec.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_res_dec.set_xlabel('Observation Year', fontsize=12)
        ax_res_dec.set_ylabel('Dec Chi-Squared Residuals (σ)', fontsize=11)
        ax_res_dec.grid(True, alpha=0.3)
        ax_res_dec.legend(loc='upper right', fontsize=9)
        
        # Combined norm residuals sqrt(RA^2 + Dec^2), color-coded by telescope
        r_norm_initial = np.sqrt(residuals_ra_initial**2 + residuals_dec_initial**2)
        for telescope in unique_telescopes:
            if telescope and telescope.strip():
                telescope_clean = telescope.strip()
                mask = obs_data['telescope'] == telescope
                color = telescope_colors.get(telescope_clean, 'gray')
                ax_res_norm.scatter(obs_data.loc[mask, 'year'].values,
                                    r_norm_initial[mask],
                                    color=color, s=25, alpha=0.7,
                                    label=f'{telescope_clean} norm', zorder=2)
        ax_res_norm.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_res_norm.set_xlabel('Observation Year', fontsize=12)
        ax_res_norm.set_ylabel('Residual Norm (σ)', fontsize=11)
        ax_res_norm.grid(True, alpha=0.3)
        ax_res_norm.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig_res, use_container_width=True)
        
        # Calculate reduced chi-squared
        # Chi-squared = sum of squared normalized residuals
        chi_squared = np.sum(residuals_ra_initial**2) + np.sum(residuals_dec_initial**2)
        
        # Number of data points (RA + Dec for each observation)
        n_data_points = len(residuals_ra_initial) + len(residuals_dec_initial)
        
        # Number of fitted parameters
        n_params = 8  # a, e, i, Omega, omega, T0, P, distance_gc
        
        # Degrees of freedom
        dof = n_data_points - n_params
        
        # Reduced chi-squared
        if dof > 0:
            reduced_chi_squared = chi_squared / dof
        else:
            reduced_chi_squared = np.nan
        
        # Build Initial Model chi-squared summary text to show under the photo
        initial_lines = []
        initial_lines.append(f"Reduced χ² (Initial Model) = {reduced_chi_squared:.3f} "
                             f"(χ² = {chi_squared:.2f}, degrees of freedom = {dof})")
        initial_lines.append("")
        initial_lines.append("**Per-telescope χ² (Initial Model)**")
        for telescope in unique_telescopes:
            if telescope and telescope.strip():
                telescope_clean = telescope.strip()
                mask = obs_data['telescope'] == telescope
                if np.any(mask):
                    chi2_tel = float(np.sum(residuals_ra_initial[mask]**2 +
                                            residuals_dec_initial[mask]**2))
                    n_tel = int(mask.sum())
                    dof_tel = max(1, 2 * n_tel - n_params)
                    red_chi2_tel = chi2_tel / dof_tel
                    initial_lines.append(
                        f"- **{telescope_clean}**: χ² = {chi2_tel:.2f}, "
                        f"reduced χ² ≈ {red_chi2_tel:.2f} (N = {n_tel})"
                    )
        st.session_state.initial_fit_info = "\n".join(initial_lines)
    
    # "Adjusted Model" residuals
    if st.session_state.show_table_data_3 and obs_data is not None and residuals_ra_adjusted is not None:
        # Define color map for telescopes (matching main plot)
        telescope_colors = {
            'NTT': 'blue',
            'VLT': 'red',
            'Keck': 'green',
            'GEMINI': 'orange'
        }
        
        # Create residuals plot (RA, Dec, and combined norm)
        fig_res, (ax_res_ra, ax_res_dec, ax_res_norm) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        # Get unique telescopes in the data
        unique_telescopes = obs_data['telescope'].unique()
        
        # Plot RA residuals for each telescope with matching colors
        for telescope in unique_telescopes:
            if telescope and telescope.strip():
                telescope_clean = telescope.strip()
                mask = obs_data['telescope'] == telescope
                color = telescope_colors.get(telescope_clean, 'gray')
                
                ax_res_ra.scatter(obs_data.loc[mask, 'year'].values, 
                                 residuals_ra_adjusted[mask], 
                                 color=color, s=30, alpha=0.6, 
                                 label=f'{telescope_clean} RA', zorder=2)
        
        ax_res_ra.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_res_ra.set_ylabel('RA Chi-Squared Residuals (σ)', fontsize=11)
        ax_res_ra.set_title('Orbital Fit Chi-Squared Residuals', fontsize=12, fontweight='bold')
        ax_res_ra.grid(True, alpha=0.3)
        ax_res_ra.legend(loc='upper right', fontsize=9)
        
        # Plot Dec residuals for each telescope with matching colors
        for telescope in unique_telescopes:
            if telescope and telescope.strip():
                telescope_clean = telescope.strip()
                mask = obs_data['telescope'] == telescope
                color = telescope_colors.get(telescope_clean, 'gray')
                
                ax_res_dec.scatter(obs_data.loc[mask, 'year'].values, 
                                 residuals_dec_adjusted[mask], 
                                 color=color, s=30, alpha=0.6, 
                                 label=f'{telescope_clean} Dec', zorder=2)
        
        ax_res_dec.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_res_dec.set_xlabel('Observation Year', fontsize=12)
        ax_res_dec.set_ylabel('Dec Chi-Squared Residuals (σ)', fontsize=11)
        ax_res_dec.grid(True, alpha=0.3)
        ax_res_dec.legend(loc='upper right', fontsize=9)
        
        # Combined norm residuals sqrt(RA^2 + Dec^2), color-coded by telescope
        r_norm_adjusted = np.sqrt(residuals_ra_adjusted**2 + residuals_dec_adjusted**2)
        for telescope in unique_telescopes:
            if telescope and telescope.strip():
                telescope_clean = telescope.strip()
                mask = obs_data['telescope'] == telescope
                color = telescope_colors.get(telescope_clean, 'gray')
                ax_res_norm.scatter(obs_data.loc[mask, 'year'].values,
                                    r_norm_adjusted[mask],
                                    color=color, s=25, alpha=0.7,
                                    label=f'{telescope_clean} norm', zorder=2)
        ax_res_norm.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax_res_norm.set_xlabel('Observation Year', fontsize=12)
        ax_res_norm.set_ylabel('Residual Norm (σ)', fontsize=11)
        ax_res_norm.grid(True, alpha=0.3)
        ax_res_norm.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig_res, use_container_width=True)
        
        # Calculate reduced chi-squared
        # Chi-squared = sum of squared normalized residuals
        chi_squared = np.sum(residuals_ra_adjusted**2) + np.sum(residuals_dec_adjusted**2)
        
        # Number of data points (RA + Dec for each observation)
        n_data_points = len(residuals_ra_adjusted) + len(residuals_dec_adjusted)
        
        # Number of fitted parameters
        n_params = 8  # a, e, i, Omega, omega, T0, P, distance_gc
        
        # Degrees of freedom
        dof = n_data_points - n_params
        
        # Reduced chi-squared
        if dof > 0:
            reduced_chi_squared = chi_squared / dof
        else:
            reduced_chi_squared = np.nan
        
        # Store Adjusted chi-squared summary text to show under the photo
        adj_lines = []
        adj_lines.append(
            f"Adjusted Model Reduced χ² = {reduced_chi_squared:.3f} "
            f"(χ² = {chi_squared:.2f}, degrees of freedom = {dof})"
        )
        adj_lines.append("")
        adj_lines.append("**Per-telescope χ² (Adjusted Model)**")
        for telescope in unique_telescopes:
            if telescope and telescope.strip():
                telescope_clean = telescope.strip()
                mask = obs_data['telescope'] == telescope
                if np.any(mask):
                    chi2_tel = float(np.sum(residuals_ra_adjusted[mask]**2 +
                                            residuals_dec_adjusted[mask]**2))
                    n_tel = int(mask.sum())
                    dof_tel = max(1, 2 * n_tel - n_params)
                    red_chi2_tel = chi2_tel / dof_tel
                    adj_lines.append(
                        f"- **{telescope_clean}**: χ² = {chi2_tel:.2f}, "
                        f"reduced χ² ≈ {red_chi2_tel:.2f} (N = {n_tel})"
                    )
        # Black hole mass estimate from Adjusted fit (Kepler's third law)
        if fitted_params_adjusted and 'a' in fitted_params_adjusted:
            M_bh_adjusted_solar = compute_bh_mass_solar(
                fitted_params_adjusted['a'],
                fitted_params_adjusted['P'],
                fitted_params_adjusted['distance_gc']
            )
            adj_lines.append("")
            adj_lines.append("**Black Hole Mass from S2 Orbit (Adjusted Model)**")
            adj_lines.append(
                f"- Estimated mass: {M_bh_adjusted_solar:.3e} M☉"
            )
            adj_lines.append(
                "- Using Kepler's third law: "
                r"$M = \dfrac{4\pi^2 a^3}{G P^2}$, "
                "with `a` = semi-major axis in meters and `P` = period in seconds."
            )
        st.session_state.adjusted_fit_info = "\n".join(adj_lines)
        
        # If we have Adjusted diagnostics text, show it beneath the residuals
        if diagnostics_adjusted:
            st.markdown("---")
            st.markdown(diagnostics_adjusted)
    
    # Show message if no data is displayed
    if (st.session_state.show_table_data or st.session_state.show_table_data_2 or st.session_state.show_table_data_3) and obs_data is None:
        st.warning("No observational data found. Unable to load data from CDS archive.")
    
    # About Star S2 section (moved to left column) - only show for "Show Calculated Orbit"
    if st.session_state.show_calculated_orbit:
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
    
    st.markdown('<p style="font-size:10px; font-style:italic; text-align:center; margin-top:0.25em; margin-bottom:0.75em;">Credit: DOI:10.1007/978-3-642-74391-7_4</p>', unsafe_allow_html=True)
    
    # New text container under the photo for fit summaries (Initial / Adjusted)
    fit_text_lines = []
    if st.session_state.get('show_table_data_2', False) and 'initial_fit_info' in st.session_state:
        fit_text_lines.append(st.session_state.initial_fit_info)
    if st.session_state.get('show_table_data_3', False) and 'adjusted_fit_info' in st.session_state:
        if fit_text_lines:
            fit_text_lines.append("")  # spacer between initial and adjusted if both somehow active
        fit_text_lines.append(st.session_state.adjusted_fit_info)
    
    if fit_text_lines:
        st.markdown("---")
        st.markdown("\n".join(fit_text_lines))
    
    # Orbital Information section - only show for "Show Calculated Orbit"
    if st.session_state.show_calculated_orbit:
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
st.caption("Data sources: CDS Archive (https://cdsarc.cds.unistra.fr/ftp/J/ApJ/707/L114/table1.dat), GRAVITY Collaboration, ESO observations, and published orbital element catalogs")
