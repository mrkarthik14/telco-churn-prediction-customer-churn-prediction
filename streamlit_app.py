import streamlit as st
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import yaml
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.feature_engineering import FeatureEngineer

# Page Config
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load Assets
@st.cache_resource
def load_assets():
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load all models
    models = {}
    model_names = ['xgboost', 'random_forest', 'lightgbm', 'logistic_regression']
    for name in model_names:
        model_path = f"models/{name}_model.joblib"
        if Path(model_path).exists():
            models[name] = joblib.load(model_path)
    
    # Load fitted feature engineer
    feature_engineer = joblib.load("models/feature_engineer.joblib")
    
    # Load training results
    with open('models/training_results.json', 'r') as f:
        training_results = json.load(f)
    
    return models, feature_engineer, config, training_results

try:
    models, feature_engineer, config, training_results = load_assets()
    st.success(f"✅ Loaded {len(models)} models successfully")
except Exception as e:
    st.error(f"❌ Error loading assets: {str(e)}")
    st.stop()

# Header
st.title("📊 Telco Customer Churn Prediction Dashboard")
st.markdown("Predict customer churn probability with advanced ML models and detailed analytics")

# Sidebar - Model Selection
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Selection
    st.subheader("Model Selection")
    model_display_names = {
        'xgboost': 'XGBoost',
        'random_forest': 'Random Forest', 
        'lightgbm': 'LightGBM',
        'logistic_regression': 'Logistic Regression'
    }
    
    selected_model_key = st.selectbox(
        "Choose Model",
        options=list(models.keys()),
        format_func=lambda x: model_display_names.get(x, x),
        index=0
    )
    
    model = models[selected_model_key]
    
    # Display model metrics
    if selected_model_key in training_results:
        metrics = training_results[selected_model_key]
        st.markdown(f"**Model Performance:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
            st.metric("F1 Score", f"{metrics.get('f1', 0):.2%}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
            st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
    
    st.divider()
    
    # Customer Information Input
    st.header("👤 Customer Information")
    
    # Demographics
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    
    # Service Details
    st.subheader("Service Details")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    # Additional Services
    st.subheader("Additional Services")
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    # Contract & Payment
    st.subheader("Contract & Payment")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", 
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

# Create DataFrame from inputs
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Helper Functions
def generate_billing_history(tenure, monthly_charges):
    """Generate synthetic billing history"""
    months = min(tenure, 24)  # Last 24 months max
    if months == 0:
        return pd.DataFrame({'Month': [], 'Amount': []})
    
    dates = [(datetime.now() - timedelta(days=30*i)).strftime('%Y-%m') for i in range(months-1, -1, -1)]
    # Add some realistic variation
    amounts = [monthly_charges * (1 + np.random.normal(0, 0.05)) for _ in range(months)]
    
    return pd.DataFrame({'Month': dates, 'Amount': amounts})

def create_gauge_chart(probability):
    """Create churn probability gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Risk %", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#90EE90'},
                {'range': [30, 60], 'color': '#FFD700'},
                {'range': [60, 100], 'color': '#FF6B6B'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=50, b=10))
    return fig

def create_feature_importance_chart(model, feature_names, top_n=10):
    """Create feature importance bar chart"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        fig = go.Figure(go.Bar(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            orientation='h',
            marker=dict(color=importances[indices], colorscale='Viridis')
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Feature Importances",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        return fig
    return None

# Main Content - Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📈 Analysis", "💰 Billing History"])

with tab1:
    st.subheader("Make Prediction")
    
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        try:
            # Preprocess
            processed_data = feature_engineer.engineer_features(input_data, fit=False)
            
            # Predict
            probability = model.predict_proba(processed_data)[0, 1]
            prediction = "Churn" if probability >= 0.5 else "No Churn"
            
            # Determine risk level
            if probability >= 0.6:
                risk_level = "High"
                risk_color = "🔴"
            elif probability >= 0.3:
                risk_level = "Medium"
                risk_color = "🟡"
            else:
                risk_level = "Low"
                risk_color = "🟢"
            
            # Display Results
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Churn Prediction", prediction)
            with col2:
                st.metric("Probability", f"{probability:.1%}")
            with col3:
                st.metric("Risk Level", f"{risk_color} {risk_level}")
            
            # Gauge Chart
            st.plotly_chart(create_gauge_chart(probability), use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if probability > 0.6:
                st.error("⚠️ **High Risk Customer** - Immediate Action Required")
                if contract == "Month-to-month":
                    st.markdown("- 🎯 Offer **12-month contract discount** (15-20% off)")
                if payment_method == "Electronic check":
                    st.markdown("- 💳 Encourage **automatic payment** setup with incentive")
                if internet_service == "Fiber optic" and online_security == "No":
                    st.markdown("- 🛡️ Promote **security bundle** at discounted rate")
                if tenure < 12:
                    st.markdown("- 🤝 Assign **dedicated account manager** for onboarding")
            elif probability > 0.3:
                st.warning("⚠️ **Medium Risk** - Monitor & Engage")
                st.markdown("- 📧 Send **satisfaction survey**")
                st.markdown("- 🎁 Offer **loyalty rewards** or service upgrades")
            else:
                st.success("✅ **Low Risk** - Maintain Engagement")
                st.markdown("- 📬 Continue **standard communications**")
                st.markdown("- 🌟 Consider **referral program** invitation")
                
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")

with tab2:
    st.subheader("Detailed Analysis")
    
    # Customer Profile Summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📋 Customer Profile**")
        profile_data = {
            "Tenure": f"{tenure} months",
            "Contract": contract,
            "Internet": internet_service,
            "Monthly Spend": f"${monthly_charges:.2f}",
            "Total Spend": f"${total_charges:.2f}"
        }
        for key, val in profile_data.items():
            st.markdown(f"- **{key}**: {val}")
    
    with col2:
        st.markdown("**🔍 Risk Factors**")
        risk_factors = []
        if contract == "Month-to-month":
            risk_factors.append("⚠️ Month-to-month contract")
        if tenure < 12:
            risk_factors.append("⚠️ New customer (<12 months)")
        if payment_method == "Electronic check":
            risk_factors.append("⚠️ Electronic check payment")
        if monthly_charges > 70:
            risk_factors.append("⚠️ High monthly charges")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.markdown("- ✅ No major risk factors")
    
    # Feature Importance (if available)
    if st.button("Show Feature Importance"):
        try:
            processed_data = feature_engineer.engineer_features(input_data, fit=False)
            feature_names = processed_data.columns.tolist()
            fig = create_feature_importance_chart(model, feature_names)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for this model.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab3:
    st.subheader("💰 Billing History & Trends")
    
    # Generate billing history
    billing_df = generate_billing_history(tenure, monthly_charges)
    
    if not billing_df.empty:
        # Time period filter
        col_filter1, col_filter2 = st.columns([3, 1])
        with col_filter1:
            time_filter = st.radio(
                "Select Time Period:",
                options=["1 Month", "3 Months", "6 Months", "1 Year", "Entire History"],
                index=4,  # Default to "Entire History"
                horizontal=True
            )
        
        # Filter data based on selection
        filter_map = {
            "1 Month": 1,
            "3 Months": 3,
            "6 Months": 6,
            "1 Year": 12,
            "Entire History": len(billing_df)
        }
        months_to_show = filter_map[time_filter]
        filtered_df = billing_df.tail(months_to_show)
        
        # Calculate risk thresholds and min/max
        overall_min = billing_df['Amount'].min()
        overall_max = billing_df['Amount'].max()
        overall_avg = billing_df['Amount'].mean()
        
        # Define risk thresholds (example: low = avg - 10%, high = avg + 10%)
        low_risk_threshold = overall_avg * 0.9  # 10% below average
        high_risk_threshold = overall_avg * 1.1  # 10% above average
        
        # Create enhanced line chart
        fig = px.line(filtered_df, x='Month', y='Amount', 
                     title=f'Billing History ({time_filter})',
                     markers=True)
        fig.update_traces(line_color='#1f77b4', line_width=3, name='Billing Amount')
        
        # Add horizontal lines for risk thresholds
        fig.add_hline(
            y=low_risk_threshold, 
            line_dash="dash", 
            line_color="green",
            annotation_text=f"Low Risk Threshold (${low_risk_threshold:.2f})",
            annotation_position="right"
        )
        fig.add_hline(
            y=high_risk_threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"High Risk Threshold (${high_risk_threshold:.2f})",
            annotation_position="right"
        )
        
        # Add horizontal lines for overall min/max
        fig.add_hline(
            y=overall_min, 
            line_dash="dot", 
            line_color="blue",
            annotation_text=f"Overall Min (${overall_min:.2f})",
            annotation_position="left",
            opacity=0.6
        )
        fig.add_hline(
            y=overall_max, 
            line_dash="dot", 
            line_color="orange",
            annotation_text=f"Overall Max (${overall_max:.2f})",
            annotation_position="left",
            opacity=0.6
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Amount ($)",
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced Stats with min/max
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Bill", f"${billing_df['Amount'].mean():.2f}")
        with col2:
            trend = "↗️ Increasing" if billing_df['Amount'].iloc[-1] > billing_df['Amount'].iloc[0] else "↘️ Decreasing"
            st.metric("Trend", trend)
        with col3:
            st.metric("Overall Min", f"${overall_min:.2f}", delta=f"-${overall_avg - overall_min:.2f}", delta_color="inverse")
        with col4:
            st.metric("Overall Max", f"${overall_max:.2f}", delta=f"+${overall_max - overall_avg:.2f}")
        
        # Risk Analysis Summary
        st.markdown("---")
        st.markdown("**📊 Risk Threshold Legend:**")
        col_legend1, col_legend2, col_legend3, col_legend4 = st.columns(4)
        with col_legend1:
            st.markdown("🟢 **Low Risk**: Below 90% of average")
        with col_legend2:
            st.markdown("🔴 **High Risk**: Above 110% of average")
        with col_legend3:
            st.markdown("🔵 **Overall Min**: Lowest bill amount")
        with col_legend4:
            st.markdown("🟠 **Overall Max**: Highest bill amount")
        
        # Show data table
        with st.expander("📊 View Billing Data"):
            st.dataframe(filtered_df, use_container_width=True)
    else:
        st.info("No billing history available for new customers (tenure = 0 months)")
