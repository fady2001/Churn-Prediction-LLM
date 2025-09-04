import io
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

# Add src to path so we can import our modules
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import MODELS_DIR
from src.llm import CustomerInfoExtractor
from src.predictor import ChurnPredictor

# Configure Streamlit page
st.set_page_config(
    page_title="Churn Prediction LLM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Load the trained predictor and LLM extractor"""
    try:
        # Load the trained ML model
        predictor = ChurnPredictor.load(MODELS_DIR / "artifacts.pkl")
        
        # Initialize the LLM extractor
        extractor = CustomerInfoExtractor()
        
        return predictor, extractor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def customer_data_to_dataframe(customer_data) -> pd.DataFrame:
    """Convert CustomerData object to DataFrame for prediction"""
    # Convert to dict and handle None values
    data_dict = customer_data.model_dump()
    
    # Create DataFrame with single row
    df = pd.DataFrame([data_dict])
    
    # Handle data type conversions to match training data
    numeric_columns = ['tenure', 'Monthly_Charges', 'Total_Charges']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def display_prediction_results(results: pd.DataFrame, customer_text: str = None):
    """Display prediction results in a formatted way"""
    for idx, row in results.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                if customer_text:
                    st.write("**Customer Description:**")
                    st.write(customer_text[:200] + "..." if len(customer_text) > 200 else customer_text)
            
            with col2:
                # Prediction with color coding
                prediction = row['prediction_label']
                churn_prob = row['churn_probability']
                
                if prediction == 'Yes':
                    st.error(f"🚨 **WILL CHURN**")
                else:
                    st.success(f"✅ **WILL STAY**")
                
                st.metric("Churn Probability", f"{churn_prob:.1%}")
            
            with col3:
                confidence = row['confidence']
                st.metric("Confidence", f"{confidence:.1%}")
                
                # Risk level based on probability
                if churn_prob >= 0.7:
                    st.error("High Risk")
                elif churn_prob >= 0.4:
                    st.warning("Medium Risk")
                else:
                    st.success("Low Risk")
            
            st.divider()

def main():
    st.title("🔮 Churn Prediction with LLM")
    st.markdown("Predict customer churn using natural language descriptions powered by Large Language Models")
    
    # Load models
    predictor, extractor = load_models()
    
    if predictor is None or extractor is None:
        st.stop()
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📋 Model Information")
        st.info("**LLM Model:** Gemma 3 1B")
        st.info("**ML Pipeline:** Trained Classification Model")
        
        # Show target encoding mapping
        try:
            mapping = predictor.get_target_mapping()
            st.write("**Target Mapping:**")
            for k, v in mapping.items():
                st.write(f"- {k} → {v}")
        except:
            pass
    
    # Main content tabs
    tab1, tab2 = st.tabs(["👤 Single Customer", "👥 Batch Customers"])
    
    with tab1:
        st.header("Single Customer Prediction")
        st.markdown("Enter a natural language description of a customer to predict their churn probability.")
        
        # Example descriptions
        with st.expander("💡 See Example Descriptions"):
            st.markdown("""
            **Example 1 (High Risk):**
            Customer 4593 is a male senior citizen who has been with the company for 2 months. He is single with no dependents. He has phone service and fiber optic internet with no additional services. He has a month-to-month contract and pays by electronic check. His monthly charges are 85.50.
            
            **Example 2 (Low Risk):**
            Customer 1234 is a married female with children who has been with us for 3 years. She has phone service and fiber optic internet with all premium services including streaming TV and movies. She has a two-year contract with paperless billing and pays by automatic bank transfer. Her monthly charges are 95.00.
            
            **Example 3 (Medium Risk):**
            A young professional male customer has been with the company for 18 months. He has DSL internet with basic services, uses paperless billing, and has a one-year contract. He pays around 65 dollars monthly.
            """)
        
        # Text input for customer description
        customer_description = st.text_area(
            "Customer Description",
            placeholder="Enter a detailed description of the customer including demographics, services, contract details, and billing information...",
            height=150
        )
        
        if st.button("🔍 Predict Churn", type="primary"):
            if customer_description.strip():
                with st.spinner("Extracting customer information and making prediction..."):
                    try:
                        # Extract customer info using LLM
                        customer_data = extractor.extract_customer_info(customer_description)
                        
                        # Convert to DataFrame
                        df = customer_data_to_dataframe(customer_data)
                        
                        # Make prediction
                        results = predictor.predict_with_interpretation(df)
                        
                        # Display extracted information
                        st.subheader("📋 Extracted Customer Information")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.json(customer_data.model_dump(), expanded=False)
                        
                        with col2:
                            st.subheader("🎯 Prediction Results")
                            display_prediction_results(results, customer_description)
                            
                    except Exception as e:
                        st.error(f"Error processing prediction: {str(e)}")
                        st.error("Please check that Ollama is running and the model is available.")
            else:
                st.warning("Please enter a customer description.")
    
    with tab2:
        st.header("Batch Customer Predictions")
        st.markdown("Process multiple customer descriptions at once.")
        
        # Option 1: Text area for multiple descriptions
        st.subheader("📝 Enter Multiple Descriptions")
        st.markdown("Enter one customer description per line:")
        
        batch_descriptions = st.text_area(
            "Customer Descriptions (one per line)",
            placeholder="Customer 1: A senior citizen male...\nCustomer 2: A young female professional...\nCustomer 3: A married couple with children...",
            height=200
        )
        
        # Option 2: File upload
        st.subheader("📁 Upload Text File")
        uploaded_file = st.file_uploader(
            "Upload a text file with customer descriptions (one per line)",
            type=['txt']
        )
        
        descriptions_to_process = []
        
        # Process text area input
        if batch_descriptions.strip():
            descriptions_to_process = [desc.strip() for desc in batch_descriptions.split('\n') if desc.strip()]
        
        # Process uploaded file
        if uploaded_file is not None:
            try:
                content = uploaded_file.read().decode('utf-8')
                file_descriptions = [desc.strip() for desc in content.split('\n') if desc.strip()]
                descriptions_to_process.extend(file_descriptions)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        if descriptions_to_process:
            st.info(f"Found {len(descriptions_to_process)} customer descriptions to process")
            
            if st.button("🚀 Process Batch Predictions", type="primary"):
                with st.spinner(f"Processing {len(descriptions_to_process)} customer descriptions..."):
                    try:
                        all_results = []
                        all_extracted_data = []
                        
                        progress_bar = st.progress(0)
                        
                        for i, description in enumerate(descriptions_to_process):
                            # Update progress
                            progress_bar.progress((i + 1) / len(descriptions_to_process))
                            
                            # Extract customer info
                            customer_data = extractor.extract_customer_info(description)
                            all_extracted_data.append(customer_data.model_dump())
                            
                            # Convert to DataFrame and predict
                            df = customer_data_to_dataframe(customer_data)
                            results = predictor.predict_with_interpretation(df)
                            
                            # Add description to results
                            results['customer_description'] = description
                            results['customer_index'] = i + 1
                            all_results.append(results)
                        
                        # Combine all results
                        combined_results = pd.concat(all_results, ignore_index=True)
                        
                        # Display summary
                        st.subheader("📊 Batch Prediction Summary")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            total_customers = len(combined_results)
                            st.metric("Total Customers", total_customers)
                        
                        with col2:
                            churn_count = (combined_results['prediction_label'] == 'Yes').sum()
                            st.metric("Predicted Churners", churn_count)
                        
                        with col3:
                            churn_rate = churn_count / total_customers * 100
                            st.metric("Churn Rate", f"{churn_rate:.1f}%")
                        
                        with col4:
                            avg_churn_prob = combined_results['churn_probability'].mean()
                            st.metric("Avg Churn Probability", f"{avg_churn_prob:.1%}")
                        
                        # Detailed results
                        st.subheader("🔍 Detailed Results")
                        
                        # Sort by churn probability (highest risk first)
                        combined_results = combined_results.sort_values('churn_probability', ascending=False)
                        
                        for idx, row in combined_results.iterrows():
                            st.write(f"**Customer {row['customer_index']}**")
                            display_prediction_results(pd.DataFrame([row]), row['customer_description'])
                        
                        # Download results
                        st.subheader("💾 Download Results")
                        
                        # Prepare download data
                        download_data = combined_results[['customer_index', 'customer_description', 
                                                        'prediction_label', 'churn_probability', 
                                                        'confidence']].copy()
                        
                        csv = download_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="batch_churn_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error processing batch predictions: {str(e)}")
                        st.error("Please check that Ollama is running and the model is available.")
        else:
            st.info("Enter customer descriptions above or upload a text file to get started.")

if __name__ == "__main__":
    main()