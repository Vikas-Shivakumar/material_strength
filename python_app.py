import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import joblib # To load your scaler

# --- 1. MODEL DEFINITION (Must match training) ---
class VSSC_MaterialAI(torch.nn.Module):
    def __init__(self, input_dim=5, vector_dim=100):
        super(VSSC_MaterialAI, self).__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128)
        )
        self.scalar_head = torch.nn.Linear(128, 2)
        self.stress_head = torch.nn.Sequential(
            torch.nn.Linear(128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, vector_dim)
        )
        self.strain_head = torch.nn.Sequential(
            torch.nn.Linear(128, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, vector_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.scalar_head(features), self.stress_head(features), self.strain_head(features)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    model = VSSC_MaterialAI()
    # Ensure these files are in the same folder as app.py
    model.load_state_dict(torch.load('vssc_material_model.pth', map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load('scaler.joblib')
    return model, scaler

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="VSSC Material AI Lab", layout="wide")

st.title("🚀 VSSC Virtual Tensile Testing Lab")
st.markdown("### AI-Powered Stress-Strain Prediction for Weldment Repairs")
st.write("Adjust the alloy composition and repair parameters to see the predicted mechanical behavior.")

with st.sidebar:
    st.header("Alloy Parameters")
    fe = st.slider("Iron (Fe %)", 65.0, 75.0, 70.0)
    cr = st.slider("Chromium (Cr %)", 15.0, 20.0, 18.0)
    ni = st.slider("Nickel (Ni %)", 8.0, 12.0, 10.0)
    
    st.header("Process Parameters")
    v = st.slider("Weld Voltage (V)", 10.0, 15.0, 12.5)
    repair = st.selectbox("Repair Cycles", [0, 1, 2])
    
    predict_btn = st.button("Generate Stress-Strain Curve", type="primary")

# --- 4. PREDICTION LOGIC ---
model, scaler = load_assets()

if predict_btn:
    # Prepare Input
    raw_input = np.array([[fe, cr, ni, v, repair]])
    scaled_input = torch.tensor(scaler.transform(raw_input), dtype=torch.float32)

    with torch.no_grad():
        scalars, stress_vec, strain_vec = model(scaled_input)
    
    yield_val = scalars[0][0].item()
    uts_val = scalars[0][1].item()
    stress = stress_vec.squeeze().numpy()
    strain = strain_vec.squeeze().numpy()

    # --- 5. RESULTS DISPLAY ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric("Predicted Yield Strength", f"{yield_val:.2f} MPa")
        st.metric("Predicted UTS", f"{uts_val:.2f} MPa")
        
        st.info(f"""
        **Analysis:**
        - Alloy: Fe{fe}% Cr{cr}% Ni{ni}%
        - Heat Input: {v}V
        - State: Repair Cycle {repair}
        """)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(strain, stress, color='#1f77b4', linewidth=3, label='AI Predicted Behavior')
        ax.axhline(y=uts_val, color='r', linestyle='--', alpha=0.5, label=f'UTS: {uts_val:.1f} MPa')
        ax.fill_between(strain, stress, alpha=0.1, color='#1f77b4') # Area = Toughness
        
        ax.set_title(f"Predicted Material Flow (R{repair})", fontsize=14)
        ax.set_xlabel("Strain (mm/mm)")
        ax.set_ylabel("Stress (MPa)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)