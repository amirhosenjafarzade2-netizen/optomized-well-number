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
    features = {}
    grid_match = re.search(r'GRID CORNER\s+(\d+)\s+(\d+)\s+(\d+)', content)
    if grid_match:
        features['nx'], features['ny'], features['nz'] = map(int, grid_match.groups())
    
    di = re.findall(r'DI IVAR\s+((?:\s*\d+\.?\d*){1,40})', content)
    dj = re.findall(r'DJ JVAR\s+((?:\s*\d+\.?\d*){1,25})', content)
    all_dx = [float(x) for s in di for x in s.split() if x.strip()]
    all_dy = [float(x) for s in dj for x in s.split() if x.strip()]
    features['avg_dx'] = np.mean(all_dx) if all_dx else None
    features['avg_dy'] = np.mean(all_dy) if all_dy else None
    
    # Fixed regex to capture the value correctly
    permj_match = re.search(r"EQUALSI\s+0\s+([\d.]+)", content, re.IGNORECASE)
    features['perm_i_md'] = float(permj_match.group(1)) if permj_match else None
    
    permk_match = re.search(r"EQUALSI\s+1\s+([\d.]+)", content, re.IGNORECASE)
    features['perm_k_md'] = float(permk_match.group(1)) if permk_match else None
    
    if features.get('perm_i_md') and features.get('perm_k_md') and features['perm_i_md'] > 0:
        features['kv_kh_ratio'] = features['perm_k_md'] / features['perm_i_md']
    
    if all(k in features and features[k] is not None for k in ['nx', 'ny', 'avg_dx', 'avg_dy']):
        features['area_acres'] = (features['nx'] * features['avg_dx'] * features['ny'] * features['avg_dy']) / 43560
    
    return features

def parse_out_production(content):
    prod_match = re.search(r'Cumulative Production\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
    inplace_match = re.search(r'Current Fluids In Place\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', content)
    
    production = {}
    if prod_match:
        production['cum_oil_mstb'] = float(prod_match.group(1))
        production['cum_gas_mmscf'] = float(prod_match.group(2))
        production['cum_water_mstb'] = float(prod_match.group(3))
    
    if inplace_match:
        production['inplace_oil_mstb'] = float(inplace_match.group(1))
    
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
        for j in range(20):
            wells = df.iloc[start_row + j, 6]
            npv = df.iloc[start_row + j, 8]
            if pd.notna(wells) and pd.notna(npv):
                try:
                    npv_dict[int(wells)] = float(npv)
                except:
                    pass
    return npv_dict

def safe_format(value, fmt="{:.3f}", default="N/A"):
    return fmt.format(value) if value is not None else default

def build_feature_vector(wells, base_features, production=None):
    vec = [wells, wells**2]
    
    if base_features:
        area = base_features.get('area_acres', 1000)
        drainage = area / wells if wells > 0 else area
        interference = 1 / (1 + wells / 10)
        perm_sqrt = base_features.get('perm_i_md', 100) ** 0.5 if base_features.get('perm_i_md') else 10
        recovery_pot = perm_sqrt * np.log(wells + 1)
        kvkh_term = wells * base_features.get('kv_kh_ratio', 0.1)
        vec.extend([drainage, interference, recovery_pot, kvkh_term])
    
    if production:
        rf = production.get('recovery_factor', 0.3)
        oil_per_well = production.get('cum_oil_mstb', 0) / wells if wells > 0 else 0
        water_penalty = -production.get('cum_water_mstb', 0) / (production.get('cum_oil_mstb', 1) + 1e-6)
        vec.extend([rf, oil_per_well, water_penalty])
    
    return np.array(vec).reshape(1, -1)

def predict_optimum(npv_data, base_features, production_data, max_wells=20):
    wells_list = np.array(sorted(npv_data.keys()))
    npv_list = np.array([npv_data[w] for w in wells_list])
    
    X = np.vstack([build_feature_vector(w, base_features, production_data.get(w)) for w in wells_list])
    
    model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    model.fit(X, npv_list)
    
    candidate_wells = np.arange(min(wells_list), max_wells + 1)
    X_pred = []
    for w in candidate_wells:
        approx_prod = {}
        if production_data:
            avg_rf = np.mean([p.get('recovery_factor', 0.3) for p in production_data.values()])
            avg_oil = np.mean([p.get('cum_oil_mstb', 1000) for p in production_data.values()])
            avg_water = np.mean([p.get('cum_water_mstb', 500) for p in production_data.values()])
            scale = w / np.mean(wells_list)
            approx_prod = {
                'recovery_factor': avg_rf,
                'cum_oil_mstb': avg_oil * scale,
                'cum_water_mstb': avg_water * scale
            }
        X_pred.append(build_feature_vector(w, base_features, approx_prod)[0])
    X_pred = np.array(X_pred)
    pred_npv = model.predict(X_pred)
    
    best_idx = np.argmax(pred_npv)
    return int(candidate_wells[best_idx]), pred_npv[best_idx], candidate_wells, pred_npv, wells_list, npv_list

# Streamlit UI
st.set_page_config(page_title="Reservoir Well Optimizer", layout="wide")
st.title("🛢 Reservoir Development Optimizer")
st.markdown("**Physics + Production + Economics Informed Well Count Predictor**")

uploaded_excel = st.file_uploader("Upload NPV Excel file", type=["xlsx"])
uploaded_dats = st.file_uploader("Upload CMG .dat files (optional)", type=["dat"], accept_multiple_files=True)
uploaded_outs = st.file_uploader("Upload CMG .out files (recommended)", type=["out", "txt"], accept_multiple_files=True)

npv_data = {}
production_data = {}
base_features = {}

if uploaded_excel:
    with open("temp.xlsx", "wb") as f:
        f.write(uploaded_excel.getbuffer())
    npv_data = extract_npv_from_excel("temp.xlsx")
    os.remove("temp.xlsx")

dat_features = {}
if uploaded_dats:
    for f in uploaded_dats:
        content = f.getvalue().decode('utf-8', errors='ignore')
        wells = extract_well_count(f.name)
        if wells:
            data = parse_dat_features(content)
            if data:
                dat_features[wells] = {"file": f.name, "data": data}

if uploaded_outs:
    for f in uploaded_outs:
        content = f.getvalue().decode('utf-8', errors='ignore')
        wells = extract_well_count(f.name)
        if wells:
            data = parse_out_production(content)
            if data:
                production_data[wells] = data

# Display extracted info
if npv_data:
    st.success(f"Found {len(npv_data)} NPV scenarios: {sorted(npv_data.keys())} wells")
    st.write("**NPV values:**", {k: f"${v:,.0f}" for k, v in npv_data.items()})

if dat_features:
    st.success(f"Parsed reservoir properties from {len(dat_features)} .dat files")
    for w, info in dat_features.items():
        f = info["data"]
        grid_str = f"{f.get('nx','?')}×{f.get('ny','?')}×{f.get('nz','?')}" if f.get('nx') else "?"
        perm_str = f"Perm I = {safe_format(f.get('perm_i_md'), '{:.1f}')} md" if f.get('perm_i_md') is not None else "Perm I = N/A"
        kvkh_str = f"Kv/Kh = {safe_format(f.get('kv_kh_ratio'))}" if f.get('kv_kh_ratio') is not None else "Kv/Kh = N/A"
        area_str = f"Area ≈ {safe_format(f.get('area_acres'), '{:.0f}')} acres" if f.get('area_acres') is not None else "Area = N/A"
        st.write(f"**{w} wells** ({info['file']}): Grid {grid_str}, {perm_str}, {kvkh_str}, {area_str}")

if production_data:
    st.success(f"Extracted production data from {len(production_data)} .out files")
    for w, p in production_data.items():
        rf_str = f"{p.get('recovery_factor',0):.1%}" if p.get('recovery_factor') is not None else "N/A"
        st.write(f"**{w} wells**: Cum Oil = {p.get('cum_oil_mstb',0):.1f} MSTB, "
                 f"Recovery Factor = {rf_str}, OOIP ≈ {p.get('ooip_mstb',0):.1f} MSTB")

# Average base features from .dat files
if dat_features:
    keys = ['area_acres', 'perm_i_md', 'kv_kh_ratio']
    for k in keys:
        vals = [f['data'].get(k) for f in dat_features.values() if f['data'].get(k) is not None]
        if vals:
            base_features[k] = np.mean(vals)

# Prediction
if npv_data and len(npv_data) >= 2:
    try:
        best_wells, best_npv, cand_wells, pred_npv, act_wells, act_npv = predict_optimum(
            npv_data, base_features, production_data
        )
        
        st.markdown("## 🎯 Optimal Development Recommendation")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("**Optimal Number of Wells**", best_wells)
        with col2:
            st.metric("**Predicted Maximum NPV**", f"${best_npv:,.0f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(act_wells, act_npv, color='blue', s=100, label='Actual NPV (Excel)')
        ax.plot(cand_wells, pred_npv, color='red', linewidth=3, label='Predicted Trend')
        ax.axvline(best_wells, color='green', linestyle='--', linewidth=2, label=f'Optimum: {best_wells} wells')
        ax.set_xlabel('Number of Wells')
        ax.set_ylabel('NPV (USD)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        if best_wells > max(act_wells):
            st.success("💡 Model recommends **more wells** — potential upside!")
        elif best_wells < min(act_wells):
            st.warning("⚠️ Model suggests diminishing returns — fewer wells may be optimal.")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
else:
    st.info("👆 Upload your Excel NPV file and at least two scenario files (.dat or .out) to get a prediction.")
