#!/usr/bin/env python3
"""
rocket_fuel_optimization_pro.py

A 'pro' version of the rocket fuel optimization app, illustrating:
  - Real or synthetic data usage
  - Multi-objective optimization (weighted sum)
  - Domain constraints (e.g., max temperature)
  - HPC-friendly training (n_jobs=-1)
  - Dynamic 3D Plotly charts in Streamlit
  - (Optional) file uploader for user-provided data
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless for server environments
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize

import plotly.graph_objects as go

# -----------------------------
# 1. App Config / HPC Settings
# -----------------------------
st.set_page_config(page_title="Rocket Fuel Optimization", layout="wide")

# ----------------------------------------------------
# 2. (Optional) File Uploader for Real Data
# ----------------------------------------------------
st.sidebar.title("Data Source")
data_choice = st.sidebar.selectbox(
    "Choose Data Source",
    ["Use Synthetic Data", "Upload CSV"]
)

if data_choice == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your rocket data CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.write("Uploaded file shape:", df.shape)
    else:
        st.sidebar.warning("Please upload a CSV file or switch to Synthetic Data.")
        df = None
else:
    df = None

# ----------------------------------------------------
# 3. Generate or Validate the Data
# ----------------------------------------------------
if df is None:
    np.random.seed(42)
    data_size = 500
    O_F_ratio = np.random.uniform(2, 6, data_size)
    chamber_pressure = np.random.uniform(1, 10, data_size)
    combustion_temp = np.random.uniform(2500, 4000, data_size)
    specific_impulse = np.random.uniform(200, 450, data_size)
    thrust = specific_impulse * chamber_pressure * 9.81 * 0.1

    df = pd.DataFrame({
        'O/F Ratio': O_F_ratio,
        'Chamber Pressure (MPa)': chamber_pressure,
        'Combustion Temp (K)': combustion_temp,
        'Specific Impulse (ISP)': specific_impulse,
        'Thrust (kN)': thrust
    })

required_cols = ['O/F Ratio', 'Chamber Pressure (MPa)', 'Combustion Temp (K)',
                 'Specific Impulse (ISP)', 'Thrust (kN)']
for c in required_cols:
    if c not in df.columns:
        st.error(f"Data must have column: '{c}'. Please fix or upload valid data.")
        st.stop()

# ----------------------------------------------------
# 4. Train/Test Split & Model Training
# ----------------------------------------------------
X = df[['O/F Ratio', 'Chamber Pressure (MPa)', 'Combustion Temp (K)', 'Specific Impulse (ISP)']]
y = df['Thrust (kN)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# ----------------------------------------------------
# 5. Hyperparameter Tuning (Optional)
# ----------------------------------------------------
tune_model = st.sidebar.checkbox("Run Hyperparameter Tuning (GridSearchCV)?", value=False)
if tune_model:
    st.sidebar.info("This may take some time depending on data size and HPC resources.")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    st.sidebar.write("Best Params:", grid_search.best_params_)

    y_pred_best = best_model.predict(X_test)
    mae_best = mean_absolute_error(y_test, y_pred_best)
    r2_best = r2_score(y_test, y_pred_best)

    st.sidebar.write(f"MAE (Best): {mae_best:.2f}")
    st.sidebar.write(f"R² (Best): {r2_best:.3f}")
else:
    best_model = model

# ----------------------------------------------------
# 6. Multi-Objective Optimization
# ----------------------------------------------------
st.sidebar.title("Multi-Objective Weights")
alpha = st.sidebar.slider("Weight for Thrust (0=Ignore Thrust, 1=Only Thrust)", 0.0, 1.0, 0.5)
max_temp_constraint = st.sidebar.slider("Max Temperature Constraint (K)", 3000, 5000, 4000)

def objective_multi(params):
    O_F, p, T = params
    df_in = pd.DataFrame({
        'O/F Ratio': [O_F],
        'Chamber Pressure (MPa)': [p],
        'Combustion Temp (K)': [T],
        'Specific Impulse (ISP)': [300]
    })
    pred_thrust = best_model.predict(df_in)[0]
    return alpha * (-pred_thrust) + (1 - alpha) * T

constraints_multi = [
    {'type': 'ineq', 'fun': lambda x: max_temp_constraint - x[2]}  # T <= max_temp_constraint
]
bounds = [(2, 6), (1, 10), (2500, 5000)]
initial_guess = [3.5, 5.0, 3000.0]

result_multi = minimize(objective_multi, initial_guess,
                        bounds=bounds,
                        constraints=constraints_multi)
opt_multi = result_multi.x

opt_thrust_df = pd.DataFrame({
    'O/F Ratio': [opt_multi[0]],
    'Chamber Pressure (MPa)': [opt_multi[1]],
    'Combustion Temp (K)': [opt_multi[2]],
    'Specific Impulse (ISP)': [300]
})
opt_thrust_val = best_model.predict(opt_thrust_df)[0]

# ----------------------------------------------------
# 7. Display Results in Main Page
# ----------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance (Train/Test)")
    st.write(f"MAE (Initial): {mae:.2f}")
    st.write(f"R² (Initial): {r2:.3f}")
    if tune_model:
        st.write("**After Tuning**:")
        st.write(f"MAE (Best): {mae_best:.2f}")
        st.write(f"R² (Best): {r2_best:.3f}")

with col2:
    st.subheader("Multi-Objective Optimization Results")
    st.write(f"**Alpha:** {alpha:.2f}")
    st.write(f"**Max Temp Constraint (K):** {max_temp_constraint}")
    st.write("**Optimal Fuel Mixture** (Weighted-sum objective):")
    st.write(f"- O/F Ratio: {opt_multi[0]:.3f}")
    st.write(f"- Pressure (MPa): {opt_multi[1]:.3f}")
    st.write(f"- Temp (K): {opt_multi[2]:.3f}")
    st.write(f"**Predicted Thrust (kN):** {opt_thrust_val:.2f}")

# ----------------------------------------------------
# 8. Interactive Sliders for Single-Point Prediction
# ----------------------------------------------------
st.title("Rocket Fuel Mixture - Interactive Prediction")

O_F_user = st.slider("Oxidizer to Fuel Ratio", 2.0, 6.0, 3.5)
pressure_user = st.slider("Chamber Pressure (MPa)", 1.0, 10.0, 5.0)
temp_user = st.slider("Combustion Temperature (K)", 2500.0, 5000.0, 3000.0)
isp_user = st.slider("Specific Impulse (ISP)", 200.0, 450.0, 300.0)

user_in = pd.DataFrame({
    'O/F Ratio': [O_F_user],
    'Chamber Pressure (MPa)': [pressure_user],
    'Combustion Temp (K)': [temp_user],
    'Specific Impulse (ISP)': [isp_user]
})
user_thrust = best_model.predict(user_in)[0]
st.write(f"**Predicted Thrust (kN):** {user_thrust:.2f}")

# ----------------------------------------------------
# 9. Dynamic 3D Plot: vary Pressure & ISP, fix O/F & Temp
# ----------------------------------------------------
st.subheader("3D Thrust Surface")

def create_3d_plot(O_F_val, temp_val):
    p_vals = np.linspace(1, 10, 30)
    isp_vals = np.linspace(200, 450, 30)

    P_grid, ISP_grid = np.meshgrid(p_vals, isp_vals)
    n_points = P_grid.size

    df_3d = pd.DataFrame({
        'O/F Ratio': [O_F_val]*n_points,
        'Chamber Pressure (MPa)': P_grid.ravel(),
        'Combustion Temp (K)': [temp_val]*n_points,
        'Specific Impulse (ISP)': ISP_grid.ravel()
    })
    thrust_3d = best_model.predict(df_3d).reshape(P_grid.shape)

    fig = go.Figure(data=[
        go.Surface(x=P_grid, y=ISP_grid, z=thrust_3d, colorscale='Viridis')
    ])
    fig.update_layout(
        title=f"Thrust vs Pressure vs ISP (O/F={O_F_val:.1f}, Temp={temp_val:.0f}K)",
        scene=dict(
            xaxis_title="Chamber Pressure (MPa)",
            yaxis_title="Specific Impulse (ISP)",
            zaxis_title="Thrust (kN)"
        )
    )
    return fig

fig_dynamic = create_3d_plot(O_F_user, temp_user)
st.plotly_chart(fig_dynamic, use_container_width=True)

# ----------------------------------------------------
# 10. Additional Tips / Next Steps
# ----------------------------------------------------
st.markdown("""
### Next Steps & Tips
1. **Replace Synthetic Data** with real rocket test data or NASA CEA outputs.
2. **Add More Constraints**: structural/thermal limits, cost, mass, etc.
3. **Multi-Objective**: use advanced libraries (pymoo, DEAP) for Pareto optimization.
4. **HPC / Distributed**: if data is large or models are complex, explore Dask or Ray.
5. **Uncertainty Analysis**: integrate SALib to see how input variation affects thrust.
6. **Deployment**: host on [Streamlit Cloud](https://streamlit.io/cloud) or Dockerize for easy lab usage.
""")
