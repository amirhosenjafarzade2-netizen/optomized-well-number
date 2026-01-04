import pandas as pd
import re
import os
import streamlit as st
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt

def extract_well_count(filename):
    match = re.search(r'(\d+)\s*wells', filename, re.IGNORECASE)
    return int(match.group(1)) if match else None

def parse_dat_features(content):
    """Extract petroleum engineering relevant features from .dat file."""
    features = {}
    
    # Grid dimensions
    grid_match = re.search(r'GRID CORNER\s+(\d+)\s+(\d+)\s+(\d+)', content)
    if grid_match:
        features['nx'], features['ny'], features['nz'] = map(int, grid_match.groups())
        features['total_cells'] = features['nx'] * features['ny'] * features['nz']
    
    # Block sizes (all 200 ft in your cases)
    di = re.findall(r'DI IVAR\s+((?:\s*\d+\.?\d*){1,40})', content)
    dj = re.findall(r'DJ JVAR\s+((?:\s*\d+\.?\d*){1,25})', content)
    all_dx = [float(x) for s in di for x in s.split()]
    all_dy = [float(x) for s in dj for x in s.split()]
    features['avg_dx'] = np.mean(all_dx) if all_dx else None
    features['avg_dy'] = np.mean(all_dy) if all_dy else None
    
    # Permeability
    permj = re.search(r"RESULTS SPEC 'Permeability J'.*EQUALSI 0\s+([\d.]+)", content)
    features['perm_i_md'] = float(permj.group(1)) if permj else None
    permk = re.search(r"RESULTS SPEC 'Permeability K'.*EQUALSI 1\s+([\d.]+)", content)
    features['perm_k_md'] = float(permk.group(1)) if permk else None
    features['kv_kh_ratio'] = features['perm_k_md'] / features['perm_i_md'] if features['perm_i_md'] and features['perm_k_md'] and features['perm_i_md'] > 0 else None
    
    # Bubble point pressure
    bpp = re.search(r"RESULTS SPEC 'Bubble Point Pressure'.*CON\s+([\d.]+)", content)
    features['bubble_point_psi'] = float(bpp.group(1)) if bpp else None
    
    # Approximate reservoir area and volume (assuming uniform)
    if 'avg_dx' in features and 'avg_dy' in features and features.get('nx') and features.get('ny'):
        features['area_acres'] = (features['nx'] * features['avg_dx'] * features['ny'] * features['avg_dy']) / 43560
    
    return features

def parse_out_production(content):
    """Extract cumulative production and in-place from .out file."""
    prod_match = re.search(r'Cumulative Production\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
    inplace_match = re.search(r'Current Fluids In Place\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
    
    production = {}
    if prod_match:
        production['cum_oil_mstb'] = float(prod_match.group(1))
        production['cum_gas_mmscf'] = float(prod_match.group(2))
        production['cum_water_mstb'] = float(prod_match.group(3))
    
    if inplace_match:
        production['inplace_oil_mstb'] = float(inplace_match.group(1))
        production['inplace_gas_mmscf'] = float(inplace_match.group(2))
        production['inplace_water_mstb'] = float(inplace_match.group(3))
    
    if 'cum_oil_mstb' in production and 'inplace_oil_mstb' in production:
        production['ooip_mstb'] = production['cum_oil_mstb'] + production['inplace_oil_mstb']
        production['recovery_factor'] = production['cum_oil_mstb'] / production['ooip_mstb'] if production['ooip_mstb'] > 0 else 0
    
    return production if production else None

def extract_npv_from_excel(excel_path):
    df = pd.read_excel(excel_path, sheet_name='Sheet1', header=None)
    npv_dict = {}
    start_row = None
    for i in range(len(df)):
        cell = df.iloc[i, 6]
        if pd.notna(cell) and isinstance(cell, str) and 'number of wells' in cell.lower():
            start_row = i + 1
            break
    if start_row:
        for j in range(10):  # more rows in case
            wells = df.iloc[start_row + j, 6]
            npv = df.iloc[start_row + j, 8]
            if pd.notna(wells) and pd.notna(npv):
                try:
                    npv_dict[int(wells)] = float(npv)
                except:
                    pass
    return npv_dict

def build_feature_vector(wells, base_features, production=None):
    """Create physics-informed features for ML, now including production data."""
    base_vec = [wells, wells**2]
    
    if base_features:
        drainage_area_per_well = base_features.get('area_acres', 1000) / wells if wells > 0 else 1000
        interference_factor = 1 / (1 + wells / 10)  # simple diminishing returns proxy
        recovery_potential = base_features.get('perm_i_md', 100) ** 0.5 * np.log(wells + 1)
        base_vec.extend([drainage_area_per_well, interference_factor, recovery_potential, wells * base_features.get('kv_kh_ratio', 0.1)])
    
    if production:
        # Add recovery-related features
        rec_factor = production.get('recovery_factor', 0.25)
        rec_per_well = production.get('cum_oil_mstb', 0) / wells if wells > 0 else 0
        water_cut_impact = -production.get('cum_water_mstb', 0) / (production.get('cum_oil_mstb', 1) + 1)  # negative impact of water
        base_vec.extend([rec_factor, rec_per_well, water_cut_impact])
    
    return np.array(base_vec).reshape(1, -1)

def predict_optimum(npv_data, base_features, production_data, max_wells=15):
    if len(npv_data) < 2:
        raise ValueError("Need at least 2 scenarios.")
    
    wells_list = np.array(sorted(npv_data.keys()))
    npv_list = np.array([npv_data[w] for w in wells_list])
    
    # Build feature matrix, using production if available for that well count
    X = np.array([build_feature_vector(w, base_features, production_data.get(w))[0] for w in wells_list])
    
    # Polynomial degree 3 with physics features
    model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    model.fit(X, npv_list)
    
    # Predict for 4 to max_wells
    candidate_wells = np.arange(4, max_wells + 1)
    # For predictions, use average production scaled by wells (approximate)
    avg_prod = {k: np.mean([p[k] for p in production_data.values() if k in p]) for k in ['recovery_factor', 'cum_oil_mstb', 'cum_water_mstb']}
    X_pred = []
    for w in candidate_wells:
        approx_prod = {
            'recovery_factor': avg_prod['recovery_factor'],
            'cum_oil_mstb': avg_prod['cum_oil_mstb'] * (w / np.mean(wells_list)),  # scale roughly
            'cum_water_mstb': avg_prod['cum_water_mstb'] * (w / np.mean(wells_list))
        }
        X_pred.append(build_feature_vector(w, base_features, approx_prod)[0])
    X_pred = np.array(X_pred)
    pred_npv = model.predict(X_pred)
    
    best_idx = np.argmax(pred_npv)
    best_wells = int(candidate_wells[best_idx])
    best_npv = pred_npv[best_idx]
    
    return best_wells, best_npv, candidate_wells, pred_npv, wells_list, npv_list

# Streamlit App
st.title("🛢 Reservoir Development Optimizer")
st.markdown("**Physics-Informed Optimum Well Count Predictor** using your CMG .dat/.out files + Economics Excel")

uploaded_excel = st.file_uploader("Upload NPV Excel file (with scenarios)", type=["xlsx"])
uploaded_dats = st.file_uploader("Upload CMG .dat files (one per well count scenario)", type=["dat"], accept_multiple_files=True)
uploaded_outs = st.file_uploader("Upload CMG .out files (one per well count scenario)", type=["out"], accept_multiple_files=True)

if uploaded_excel or uploaded_dats or uploaded_outs:
    npv_data = {}
    dat_info = {}
    out_info = {}
    all_features = {}
    production_data = {}
    
    if uploaded_excel:
        excel_path = "temp.xlsx"
        with open(excel_path, "wb") as f:
            f.write(uploaded_excel.getbuffer())
        npv_data = extract_npv_from_excel(excel_path)
        os.remove(excel_path)
    
    for file_list, parser, info_dict, data_dict in [
        (uploaded_dats, parse_dat_features, dat_info, all_features),
        (uploaded_outs, parse_out_production, out_info, production_data)
    ]:
        if file_list:
            for up_file in file_list:
                content = up_file.getvalue().decode('utf-8', errors='ignore')
                wells = extract_well_count(up_file.name)
                if wells:
                    parsed = parser(content)
                    if parsed:
                        info_dict[wells] = {"file": up_file.name, "data": parsed}
                        data_dict[wells] = parsed
    
    if npv_data:
        st.success(f"Extracted NPV for {len(npv_data)} scenarios: {sorted(npv_data.keys())} wells")
        st.write("**NPV values (USD):**", {k: f"${v:,.0f}" for k,v in npv_data.items()})
    
    if dat_info:
        st.success(f"Parsed reservoir parameters from {len(dat_info)} .dat files")
        for w, info in dat_info.items():
            f = info["data"]
            st.write(f"**{w} wells** ({info['file']}): Grid {f.get('nx')}×{f.get('ny')}×{f.get('nz')}, "
                     f"Perm I = {f.get('perm_i_md')} md, Kv/Kh = {f.get('kv_kh_ratio'):.3f}, "
                     f"Area ≈ {f.get('area_acres',0):.0f} acres")
    
    if out_info:
        st.success(f"Parsed production from {len(out_info)} .out files")
        for w, info in out_info.items():
            p = info["data"]
            st.write(f"**{w} wells** ({info['file']}): Cum Oil = {p.get('cum_oil_mstb',0):.1f} MSTB, "
                     f"Recovery Factor = {p.get('recovery_factor',0):.3f}, OOIP ≈ {p.get('ooip_mstb',0):.1f} MSTB")
    
    # Use average features from all .dat runs
    if all_features:
        base_features = {}
        for key in ['area_acres', 'perm_i_md', 'kv_kh_ratio']:
            vals = [f.get(key) for f in all_features.values() if f.get(key) is not None]
            base_features[key] = np.mean(vals) if vals else None
    else:
        base_features = {}
    
    # Run prediction if data available
    if npv_data and len(npv_data) >= 2:
        try:
            best_wells, best_npv, cand_wells, pred_npv, actual_wells, actual_npv = predict_optimum(
                npv_data, base_features, production_data
            )
            
            st.markdown("## 🎯 Prediction Result")
            st.metric("Optimum Number of Wells", best_wells, delta=None)
            st.write(f"**Predicted Maximum NPV**: ${best_npv:,.0f}")
            
            if best_wells > max(npv_data.keys()):
                st.info("Model suggests drilling **more wells** than simulated could increase NPV further.")
            elif best_wells < min(npv_data.keys()):
                st.info("Model suggests **fewer wells** may be optimal (diminishing returns).")
            
            # Plot
            fig, ax = plt.subplots(figsize=(10,6))
            ax.scatter(actual_wells, actual_npv, color='blue', s=100, label='Actual (from Excel)')
            ax.plot(cand_wells, pred_npv, color='red', linewidth=3, label='Physics & Production-informed Prediction')
            ax.axvline(best_wells, color='green', linestyle='--', label=f'Optimum: {best_wells} wells')
            ax.set_xlabel('Number of Wells')
            ax.set_ylabel('NPV (USD)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")
    else:
        st.warning("Upload Excel with NPV summary and at least 2 files (.dat or .out).")
