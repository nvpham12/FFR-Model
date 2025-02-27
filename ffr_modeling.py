import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_vif(X):
    """Function to calculate Variance Inflation Factor (VIF)."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def error_metrics(y, y_pred):
    """Function to compute model error metrics."""
    mse = round(mean_squared_error(y, y_pred), 3)
    rmse = round(np.sqrt(mse), 3)
    mae = round(mean_absolute_error(y, y_pred), 3)
    mpe = round(np.mean((y - y_pred) / y) * 100, 3)
    mape = round(np.mean(np.abs((y - y_pred) / y)) * 100, 3)
    
    return {
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse,
        'Mean Absolute Error': mae,
        'Mean Percentage Error': f"{mpe}%",
        'Mean Absolute Percentage Error': f"{mape}%"
    }

st.title("Federal Funds Effective Rate Predictive Modeling")

# Load all datasets and store them as DataFrames
FFER = pd.read_csv("https://raw.githubusercontent.com/nvpham12/FFR-Model-Data/refs/heads/main/FFER.csv", 
                   parse_dates=["observation_date"])
PGDP = pd.read_csv("https://raw.githubusercontent.com/nvpham12/FFR-Model-Data/refs/heads/main/PGDP.csv", 
                   parse_dates=["observation_date"])
RGDP = pd.read_csv("https://raw.githubusercontent.com/nvpham12/FFR-Model-Data/refs/heads/main/RGDP.csv", 
                   parse_dates=["observation_date"])
CPI = pd.read_csv("https://raw.githubusercontent.com/nvpham12/FFR-Model-Data/refs/heads/main/CPI.csv", 
                  parse_dates=["observation_date"])
FFTR_lower = pd.read_csv("https://raw.githubusercontent.com/nvpham12/FFR-Model-Data/refs/heads/main/FFTR_lower.csv", 
                         parse_dates=["observation_date"])
FFTR_upper = pd.read_csv("https://raw.githubusercontent.com/nvpham12/FFR-Model-Data/refs/heads/main/FFTR_upper.csv", 
                         parse_dates=["observation_date"])
FFTR_old = pd.read_csv("https://raw.githubusercontent.com/nvpham12/FFR-Model-Data/refs/heads/main/FFTR_old.csv", 
                       parse_dates=["observation_date"])
UNRATE = pd.read_csv("https://raw.githubusercontent.com/nvpham12/FFR-Model-Data/refs/heads/main/UNRATE.csv", 
                     parse_dates=["observation_date"])

# The data for CPI can be used to find inflation rates. 
# This is done to obtain a seasonally adjusted inflation dataset that isn't available on FRED.
inflation = pd.DataFrame()
inflation["observation_date"] = CPI["observation_date"] 
inflation["Inflation"] = CPI["CPIAUCSL"].pct_change(periods=12) * 100
inflation.dropna(inplace=True)

# Resample datasets so that they are daily frequency.
inflation = inflation.set_index("observation_date").resample("D").ffill().reset_index()
PGDP = PGDP.set_index("observation_date").resample("D").ffill().reset_index()
RGDP = RGDP.set_index("observation_date").resample("D").ffill().reset_index()
UNRATE = UNRATE.set_index("observation_date").resample("D").ffill().reset_index()

# The Federal Funds Target Rate (FFTR) is set by the Federal Reserve. 
# They used to set a single value, but they shifted to setting a range.
# Find the midpoint of the range and consider that the FFTR.
FFTR_midpoint = pd.DataFrame()
FFTR_midpoint["observation_date"] = FFTR_upper["observation_date"] 
FFTR_midpoint["Target"] = FFTR_upper["DFEDTARU"] - FFTR_lower["DFEDTARL"]

# Combine the midpoint with the old FFTR to get a complete FFTR dataset.
# Set the column names the same, then use concat() to combine the 2 datasets.
FFTR_old = FFTR_old.rename(columns = {"observation_date": "observation_date", "DFEDTAR": "Target"})
FFTR = pd.concat([FFTR_old, FFTR_midpoint])

# Merge the dataframes
df = FFER.merge(inflation, on = "observation_date", how = "outer") \
        .merge(PGDP, on = "observation_date", how = "outer") \
        .merge(RGDP, on = "observation_date", how = "outer") \
        .merge(UNRATE, on = "observation_date", how = "outer") \
        .merge(FFTR, on = "observation_date", how = "outer")

# Set date as an index and rename the columns
df = df.set_index("observation_date")
df.columns = ["Federal Funds Rate", "Inflation", "Potential GDP", "GDP", "Unemployment", "Target"]

# Find the inflation gap and output gap and add them to the dataframe
df["Inflation Gap"] = df["Inflation"] - 2
df["Output Gap"] = df["GDP"] - df["Potential GDP"]

# Find Lagged Inflation and Output Gap for Karakas Model
inflation_lag = df["Inflation"].shift(1)
output_gap_lag = df["Output Gap"].shift(1)

# Create a percentage versions of Output Gap and Inflation Lag for Karakas Model
df["Output Gap Lag %"] = (output_gap_lag / df["Potential GDP"]) * 100
df["Inflation Lag %"] = inflation_lag * 100

# Drop rows with missing values from table
df.dropna(inplace = True)

# Drop Potential GDP and GDP columns as they are not needed
df = df.drop(["Potential GDP", "GDP"], axis=1)

# Apply Standard Scaler to independent variables
scaler = StandardScaler()
df["Unemployment"] = scaler.fit_transform(df[["Unemployment"]])
df["Target"] = scaler.fit_transform(df[["Target"]])
df["Inflation Gap"] = scaler.fit_transform(df[["Inflation Gap"]])
df["Output Gap"] = scaler.fit_transform(df[["Output Gap"]])
df["Output Gap Lag %"] = scaler.fit_transform(df[["Output Gap Lag %"]])
df["Inflation Lag %"] = scaler.fit_transform(df[["Inflation Lag %"]])

# Set variables for Taylor's Rule Model
X_taylor = df[["Output Gap", "Inflation Gap"]]
X_taylor = sm.add_constant(X_taylor)

# Set variables for Karakas Model
X_karakas = df[["Output Gap Lag %", "Inflation Lag %"]]
X_karakas = sm.add_constant(X_karakas)

# Set variables for Target Model (Taylor's Rule Model plus Federal Funds Target Rate)
X_target = df[["Output Gap", "Inflation Gap", "Target"]]
X_target = sm.add_constant(X_target)

# Set variables for Unemployment Model (Taylor's Rule Model plus Unemployment)
X_unemployment = df[["Output Gap", "Inflation Gap", "Unemployment"]]
X_unemployment = sm.add_constant(X_unemployment)

# Set variables for Model with Target and Unemployment
X_both = df[["Output Gap", "Inflation Gap", "Target", "Unemployment"]]
X_both = sm.add_constant(X_both)

# Model Fitting
# Set dependent variable
y = df["Federal Funds Rate"]

# Fit Taylor's Rule Model
taylor = sm.OLS(y, X_taylor).fit()
y_pred_taylor = taylor.predict(X_taylor)

# Fit Karakas Model
karakas = sm.OLS(y, X_karakas).fit()
y_pred_karakas = karakas.predict(X_karakas)

# Fit Target Model
target = sm.OLS(y, X_target).fit()
y_pred_target = target.predict(X_target)

# Fit the Unemployment Model
unemployment = sm.OLS(y, X_unemployment).fit()
y_pred_unemployment = unemployment.predict(X_unemployment)

# Fit the Model with Target and Unemployment
both = sm.OLS(y, X_both).fit()
y_pred_both = both.predict(X_both)

tabs = st.tabs(["Data Exploration", "Variance Inflation Factors (VIFs)", "Model Summary", "Error Metrics", "Interactive Plot", "Taylor vs Karakas"])

with tabs[0]:
    # Data Exploration Section
    st.header("Data Exploration")
    st.write("Note: Unemployment, Target, Inflation Gap, Output Gap, Output Gap Lag %, and Inflation Lag % are scaled values.")
    with st.expander("First 10 Rows"):
        st.write(df.head(10))

    with st.expander("Last 10 Rows"):
        st.write(df.tail(10))

    with st.expander("Summary Statistics"):
        st.write(df.describe())

with tabs[1]:
    # VIF Section
    st.header("Variance Inflation Factors (VIFs)")

    with st.expander("Taylor Model VIFs"):
        st.write(calculate_vif(X_taylor))

    with st.expander("Karakas Model VIFs"):
        st.write(calculate_vif(X_karakas))

    with st.expander("Target Model VIFs"):
        st.write(calculate_vif(X_target))

    with st.expander("Unemployment Model VIFs"):
        st.write(calculate_vif(X_unemployment))

    with st.expander("Target and Unemployment Model VIFs"):
        st.write(calculate_vif(X_both))

with tabs[2]:
    # Model Summary Section
    with st.expander("Taylor Model Summary"):
        st.write(taylor.summary())

    with st.expander("Karakas Model Summary"):
        st.write(karakas.summary())

    with st.expander("Target Model Summary"):
        st.write(target.summary())

    with st.expander("Unemployment Model Summary"):
        st.write(unemployment.summary())
        
    with st.expander("Target and Unemployment Model Summary"):
        st.write(both.summary())

with tabs[3]:
    # Error Metrics Section
    st.header("Error Metrics")
    with st.expander("Taylor Model Metrics"):
        st.write(error_metrics(y, y_pred_taylor))

    with st.expander("Karakas Model Metrics"):
        st.write(error_metrics(y, y_pred_karakas))

    with st.expander("Target Model Metrics"):
        st.write(error_metrics(y, y_pred_target))

    with st.expander("Unemployment Model Metrics"):
        st.write(error_metrics(y, y_pred_unemployment))

    with st.expander("Target and Unemployment Model Metrics"):
        st.write(error_metrics(y, y_pred_both))

with tabs[4]:
    # Interactive Plot Section
    # Model Selector
    st.header("Interactive Plot")
    st.subheader("Add Models to Plot with Federal Funds Rates")
    models_to_plot = st.multiselect(
        "Choose models:", 
        ["Taylor", "Karakas", "Target", "Unemployment", "Target and Unemployment"], 
        default=[]
    )
    # Define a color-blind-friendly color palette
    color_palette = px.colors.qualitative.Safe  # Safe for color blindness

    # Assign colors manually
    actual_color = color_palette[0]  # Blue
    taylor_color = color_palette[1]  # Orange
    karakas_color = color_palette[2]  # Green
    target_color = color_palette[3]  # Red
    unemployment_color = color_palette[4]  # Purple
    both_color = color_palette[5]  # Brown

    # Initialize figure
    fig = go.Figure()

    # Plot Federal Funds Rate
    fig.add_trace(go.Scatter(
        x = df.index, 
        y = y, 
        mode = 'lines', 
        name = 'Actual',
        line = dict(color = actual_color, width = 2)
        ))

    # Add Taylor Model Predictions
    if "Taylor" in models_to_plot:
        fig.add_trace(go.Scatter(
            x = df.index, 
            y = y_pred_taylor, 
            mode = 'lines', 
            name = 'Taylor Predictions',
            line = dict(color = taylor_color, dash = 'dash')
            ))

    # Add Karakas Model Predictions
    if "Karakas" in models_to_plot:
        fig.add_trace(go.Scatter(
            x = df.index, 
            y = y_pred_karakas, 
            mode = 'lines', 
            name = 'Karakas Predictions',
            line = dict(color = karakas_color, dash ='dot')
            ))

    # Add Target Model Predictions
    if "Target" in models_to_plot:
        fig.add_trace(go.Scatter(
            x = df.index, 
            y = y_pred_target, 
            mode = 'lines', 
            name = 'Target Predictions',
            line = dict(color = target_color)
            ))

    # Add Unemployment Model Predictions
    if "Unemployment" in models_to_plot:
        fig.add_trace(go.Scatter(
            x = df.index, 
            y = y_pred_unemployment, 
            mode = 'lines', 
            name = 'Unemployment Predictions',
            line = dict(color = unemployment_color)
            ))

    # Add Target and Unemployment Model Predictions
    if "Target and Unemployment" in models_to_plot:
        fig.add_trace(go.Scatter(
            x = df.index, 
            y = y_pred_both, 
            mode = 'lines', 
            name = 'Target and Unemployment Predictions',
            line=dict(color=both_color)
            ))

    # Generate Plot
    st.plotly_chart(fig)

with tabs[5]:
    # Taylor vs Karakas Model Comparison
    st.subheader("Residual Sum Of Squares (RSS)")
    st.write(f"RSS for Taylor Model: {np.sum((y - y_pred_taylor) ** 2):.3f}")
    st.write(f"RSS for Karakas Model: {np.sum((y - y_pred_karakas) ** 2):.3f}")
    
    st.subheader("Sum Of Absolute Errors (SAE)")
    st.write(f"SAE for Taylor Model: {np.sum(np.abs(y - y_pred_taylor)):.3f}")
    st.write(f"SAE for Karakas Model: {np.sum(np.abs(y - y_pred_karakas)):.3f}")
    
    st.subheader("Plot of the Model Predictions Against Each Other")
    # Create scatter plot
    fig2 = px.scatter(
        x = y_pred_taylor, 
        y = y_pred_karakas, 
        labels = {'x': "Taylor Predictions", 'y': "Karakas Predictions"}, 
        title = "Taylor vs Karakas Predictions")

    # Add 45-degree reference line (y = x)
    min_val = min(min(y_pred_taylor), min(y_pred_karakas))
    max_val = max(max(y_pred_taylor), max(y_pred_karakas))

    fig2.add_trace(go.Scatter(
        x = [min_val, max_val], 
        y = [min_val, max_val], 
        mode = "lines", 
        line = dict(color = "red", dash = "dash"), 
        name = "Reference Line"
    ))
    st.plotly_chart(fig2)