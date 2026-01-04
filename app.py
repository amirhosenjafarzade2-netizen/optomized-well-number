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
    # Find the row containing 'number of wells'
    for i in range(len(df)):
        for col in range(df.shape[1]):
            cell = df.iloc[i, col]
            if pd.isna(cell):
                continue
            cell_str = str(cell).lower()
            if 'number of wells' in cell_str:
                # Found header row i, header col = col for 'number of wells'
                wells_col = col
                # Find the npv column, look in same row for 'npv'
                npv_col = None
                for c in range(col + 1, df.shape[1]):
                    hcell = df.iloc[i, c]
                    if pd.notna(hcell):
                        hcell_str = str(hcell).lower()
                        if 'npv' in hcell_str:
                            npv_col = c
                            break
                if npv_col is None:
                    continue
                # Now read from next rows, same wells_col and npv_col
                start_row = i + 1
                for j in range(20):
                    if start_row + j >= len(df):
                        break
                    wells_cell = df.iloc[start_row + j, wells_col]
                    npv_cell = df.iloc[start_row + j, npv_col]
                    if pd.notna(wells_cell) and pd.notna(npv_cell):
                        try:
                            wells_num = int(float(wells_cell))  # Handle float or str
                            npv_num = float(npv_cell)
                            npv_dict[wells_num] = npv_num
                        except ValueError:
                            pass
                break  # Assume only one such table
        else:
            continue
        break
    return npv_dict

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

def get_production(w, production_data, wells_list):
    if not production_data:
        return None
    avg_rf = np.mean([p.get('recovery_factor', 0.3) for p in production_data.values()])
    avg_oil = np.mean([p.get('cum_oil_mstb', 1000) for p in production_data.values()])
    avg_water = np.mean([p.get('cum_water_mstb', 500) for p in production_data.values()])
    mean_wells = np.mean(list(production_data.keys()))
    scale = w / mean_wells if mean_wells > 0 else 1
    if w in production_data:
        return production_data[w]
    else:
        return {
            'recovery_factor': avg_rf,
            'cum_oil_mstb': avg_oil * scale,
            'cum_water_mstb': avg_water * scale
        }

def predict_optimum(npv_data, base_features, production_data, max_wells=20):
    wells_list = np.array(sorted(npv_data.keys()))
    npv_list = np.array([npv_data[w] for w in wells_list])
  
    X = np.vstack([build_feature_vector(w, base_features, get_production(w, production_data, wells_list)) for w in wells_list])
  
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
            mean_wells = np.mean(list(production_data.keys()))
            scale = w / mean_wells if mean_wells > 0 else 1
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
# File uploaders
uploaded_excel = st.file_uploader("📊 Upload NPV Excel file (Required)", type=["xlsx"])
uploaded_dats = st.file_uploader("📄 Upload CMG .dat files (Optional)", type=["dat"], accept_multiple_files=True)
uploaded_outs = st.file_uploader("📈 Upload CMG .out files (Optional)", type=["out", "txt"], accept_multiple_files=True)
# Initialize storage for parsed data
npv_data = {}
production_data = {}
base_features = {}
dat_features = {}
# Parse uploaded files
if uploaded_excel:
    with open("temp.xlsx", "wb") as f:
        f.write(uploaded_excel.getbuffer())
    npv_data = extract_npv_from_excel("temp.xlsx")
    os.remove("temp.xlsx")
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
# Display file upload status
st.markdown("---")
st.subheader("📁 Upload Status")
col1, col2, col3 = st.columns(3)
with col1:
    if npv_data:
        st.success(f"✅ NPV Data: {len(npv_data)} scenarios")
    else:
        st.warning("⚠️ NPV Data: Not detected or not uploaded")
with col2:
    if dat_features:
        st.success(f"✅ DAT Files: {len(dat_features)} parsed")
    else:
        st.info("ℹ️ DAT Files: Optional")
with col3:
    if production_data:
        st.success(f"✅ OUT Files: {len(production_data)} parsed")
    else:
        st.info("ℹ️ OUT Files: Optional")
# Show detailed information if files are uploaded
if npv_data or dat_features or production_data:
    st.markdown("---")
    st.subheader("📊 Extracted Data Summary")
  
    if npv_data:
        with st.expander("💰 NPV Data Details", expanded=True):
            st.write(f"**Found {len(npv_data)} NPV scenarios:** {sorted(npv_data.keys())} wells")
            npv_df = pd.DataFrame([
                {"Wells": k, "NPV (USD)": f"${v:,.0f}"}
                for k, v in sorted(npv_data.items())
            ])
            st.dataframe(npv_df, use_container_width=True)
  
    if dat_features:
        with st.expander("🗂️ Reservoir Properties (.dat files)", expanded=False):
            for w, info in dat_features.items():
                f = info["data"]
              
                if f.get('nx') and f.get('ny') and f.get('nz'):
                    grid_str = f"{f['nx']}×{f['ny']}×{f['nz']}"
                else:
                    grid_str = "?"
              
                perm_val = f.get('perm_i_md')
                perm_str = f"{perm_val:.1f} md" if perm_val is not None else "N/A"
              
                kvkh_val = f.get('kv_kh_ratio')
                kvkh_str = f"{kvkh_val:.3f}" if kvkh_val is not None else "N/A"
              
                area_val = f.get('area_acres')
                area_str = f"{area_val:.0f} acres" if area_val is not None else "N/A"
              
                st.markdown(f"""
                **{w} wells** (`{info['file']}`)
                - Grid: {grid_str}
                - Permeability I: {perm_str}
                - Kv/Kh Ratio: {kvkh_str}
                - Area: {area_str}
                """)
  
    if production_data:
        with st.expander("📈 Production Data (.out files)", expanded=False):
            for w, p in production_data.items():
                rf_val = p.get('recovery_factor')
                rf_str = f"{rf_val:.1%}" if rf_val is not None else "N/A"
              
                st.markdown(f"""
                **{w} wells**
                - Cumulative Oil: {p.get('cum_oil_mstb', 0):.1f} MSTB
                - Recovery Factor: {rf_str}
                - OOIP: {p.get('ooip_mstb', 0):.1f} MSTB
                """)
# Average base features from .dat files
if dat_features:
    keys = ['area_acres', 'perm_i_md', 'kv_kh_ratio']
    for k in keys:
        vals = [f['data'].get(k) for f in dat_features.values() if f['data'].get(k) is not None]
        if vals:
            base_features[k] = np.mean(vals)
# Prediction section with button
st.markdown("---")
# Check if ready to predict
can_predict = npv_data and len(npv_data) >= 2
if can_predict:
    st.subheader("🚀 Run Optimization Analysis")
  
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"Ready to analyze {len(npv_data)} scenarios and predict optimal well count.")
        if dat_features:
            st.write(f"✓ Using reservoir properties from {len(dat_features)} .dat file(s)")
        if production_data:
            st.write(f"✓ Using production data from {len(production_data)} .out file(s)")
  
    with col2:
        run_optimization = st.button("🎯 Find Optimal Wells", type="primary", use_container_width=True)
  
    if run_optimization:
        with st.spinner("🔄 Running machine learning optimization..."):
            try:
                best_wells, best_npv, cand_wells, pred_npv, act_wells, act_npv = predict_optimum(
                    npv_data, base_features, production_data
                )
              
                st.markdown("---")
                st.markdown("## 🎯 Optimization Results")
              
                # Main metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("**Optimal Well Count**", best_wells, help="Recommended number of wells for maximum NPV")
                with col2:
                    st.metric("**Maximum NPV**", f"${best_npv:,.0f}", help="Predicted Net Present Value at optimal well count")
                with col3:
                    improvement = ((best_npv - max(act_npv)) / max(act_npv) * 100)
                    st.metric("**NPV Improvement**", f"{improvement:+.1f}%", help="Improvement vs best scenario in data")
              
                # Plot
                st.markdown("### 📊 NPV vs Well Count Analysis")
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.scatter(act_wells, act_npv, color='#1f77b4', s=150, label='Actual NPV (Excel Data)',
                          zorder=5, edgecolors='white', linewidth=2)
                ax.plot(cand_wells, pred_npv, color='#ff7f0e', linewidth=3, label='ML Predicted Trend', alpha=0.8)
                ax.axvline(best_wells, color='#2ca02c', linestyle='--', linewidth=2.5,
                          label=f'Optimum: {best_wells} wells', alpha=0.8)
                ax.scatter([best_wells], [best_npv], color='#2ca02c', s=300, marker='*',
                          zorder=6, edgecolors='white', linewidth=2, label='Optimal Point')
              
                ax.set_xlabel('Number of Wells', fontsize=12, fontweight='bold')
                ax.set_ylabel('NPV (USD)', fontsize=12, fontweight='bold')
                ax.set_title('Well Count Optimization Analysis', fontsize=14, fontweight='bold', pad=20)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', fontsize=10, framealpha=0.9)
              
                # Format y-axis as currency
                from matplotlib.ticker import FuncFormatter
                def currency(x, pos):
                    return f'${x/1e6:.1f}M' if abs(x) >= 1e6 else f'${x/1e3:.0f}K'
                ax.yaxis.set_major_formatter(FuncFormatter(currency))
              
                plt.tight_layout()
                st.pyplot(fig)
              
                # Recommendations
                st.markdown("### 💡 Recommendations")
                if best_wells > max(act_wells):
                    st.success(f"""
                    **Upside Potential Identified!**
                  
                    The model predicts that increasing well count to **{best_wells} wells** (beyond your current maximum of {max(act_wells)} wells)
                    could improve NPV by **${(best_npv - max(act_npv)):,.0f}** ({improvement:+.1f}%).
                  
                    Consider running additional scenarios to validate this prediction.
                    """)
                elif best_wells < min(act_wells):
                    st.warning(f"""
                    **Diminishing Returns Detected**
                  
                    The model suggests that **{best_wells} wells** may be optimal, which is fewer than your minimum tested scenario ({min(act_wells)} wells).
                  
                    This indicates potential over-development in current plans. Consider cost-benefit analysis for reduced well counts.
                    """)
                else:
                    st.info(f"""
                    **Optimal Range Identified**
                  
                    The model predicts **{best_wells} wells** is optimal, which falls within your tested range ({min(act_wells)}-{max(act_wells)} wells).
                  
                    Your current scenarios cover the optimal development strategy well.
                    """)
              
                # Model features used
                with st.expander("🔬 Model Features & Methodology", expanded=False):
                    st.markdown("""
                    **Features Used in ML Model:**
                    - Well count and quadratic term (interference effects)
                    - Drainage area per well
                    - Well interference factor
                    - Recovery potential (permeability-based)
                    - Vertical-horizontal permeability effects
                    """)
                  
                    if production_data:
                        st.markdown("- Recovery factor data\n- Oil production per well\n- Water production penalty")
                  
                    st.markdown(f"""
                    **Model Type:** Polynomial Regression (degree 3)
                  
                    **Training Data:** {len(npv_data)} scenarios
                  
                    **Prediction Range:** {min(cand_wells)} to {max(cand_wells)} wells
                    """)
          
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.exception(e)
else:
    st.info("""
    ### ⚠️ Requirements to Run Optimization
  
    Please upload:
    1. **NPV Excel file** with at least **2 scenarios** (different well counts)
    2. Optional: .dat files for reservoir properties
    3. Optional: .out files for production data
  
    The more data you provide, the better the prediction accuracy!
    """)
