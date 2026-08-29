import numpy as np
import pandas as pd

def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 5 dimensionless physics-informed membrane hydrodynamic, 
    steric sieving, and electrostatic coupling parameters from raw experimental columns.
    
    Returns a copy of the dataframe with 5 new engineered columns.
    """
    df_out = df.copy()
    
    # 1. Extract raw physical quantities (safeguarded against division by zero)
    r_solute = df_out['Molecular radius (nm)'].values
    r_pore = np.maximum(df_out['Pore radius (nm)'].values, 1e-6)
    
    flux = df_out['Pure\u2009water flux (L\u00b7m-2\u00b7h-1)'].values if 'Pure\u2009water flux (L\u00b7m-2\u00b7h-1)' in df_out.columns else df_out.iloc[:, 4].values
    pressure = np.maximum(df_out['Pressure (bar)'].values, 1e-6)
    
    charge = df_out['Molecular charge'].values
    zeta = df_out['Zeta potential (mV)'].values
    ph = np.maximum(df_out['pH'].values, 1e-6)
    
    log_d = df_out['log D '].values if 'log D ' in df_out.columns else df_out.iloc[:, 22].values
    contact_angle = df_out['Contact angle (\u00b0)'].values if 'Contact angle (\u00b0)' in df_out.columns else df_out.iloc[:, 19].values
    
    # 2. Compute 5 dimensionless / physical descriptors
    # (a) Steric Sieve Ratio (lambda = r_s / r_p)
    lambda_steric = r_solute / r_pore
    
    # (b) Ferry-Renkin Steric Exclusion Factor: Phi = (1-lambda)^2 * (2 - (1-lambda)^2)
    sieve_term = np.maximum(0.0, 1.0 - lambda_steric)
    phi_ferry = (sieve_term ** 2) * (2.0 - (sieve_term ** 2))
    
    # (c) Membrane Hydraulic Permeability: L_p = Flux / Pressure (L m^-2 h^-1 bar^-1)
    permeability = flux / pressure
    
    # (d) Donnan Electrostatic Exclusion Index: Psi = (Charge * Zeta) / pH
    donnan_electro = (charge * zeta) / ph
    
    # (e) Hydrophobic Surface Affinity: H = logD * cos(theta)
    theta_rad = np.radians(contact_angle)
    hydrophobic_affinity = log_d * np.cos(theta_rad)
    
    # 3. Add new columns
    df_out['Steric_Ratio'] = lambda_steric
    df_out['Ferry_Renkin_Factor'] = phi_ferry
    df_out['Hydraulic_Permeability'] = permeability
    df_out['Donnan_Electro_Index'] = donnan_electro
    df_out['Hydrophobic_Affinity'] = hydrophobic_affinity
    
    return df_out


def extract_features_and_labels(df: pd.DataFrame, use_physics: bool = True):
    """
    Extracts numerical tabular feature matrix X and target rejection labels y.
    
    If use_physics is True: returns (19 raw + 5 physics) = 24 dimensional X.
    If use_physics is False: returns 19 raw dimensional X.
    """
    if use_physics:
        df_proc = add_physics_features(df)
        raw_cols = df_proc.iloc[:, 4:23].values
        physics_cols = df_proc[['Steric_Ratio', 'Ferry_Renkin_Factor', 'Hydraulic_Permeability', 
                                'Donnan_Electro_Index', 'Hydrophobic_Affinity']].values
        X = np.hstack([raw_cols, physics_cols])
    else:
        X = df.iloc[:, 4:23].values
        
    y = df.iloc[:, 23].values
    smiles = df.iloc[:, 3].values
    return X, y, smiles
