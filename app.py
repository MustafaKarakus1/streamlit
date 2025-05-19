import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import shap

# Set page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load models and data
@st.cache_resource
def load_model():
    model = pickle.load(open('bank_marketing_model_V2.pkl', 'rb'))
    feature_names = pickle.load(open('feature_names.pkl', 'rb'))
    label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
    deployment_info = pickle.load(open('deployment_info.pkl', 'rb'))
    return model, feature_names, label_encoder, deployment_info

# Load models and data
try:
    model, feature_names, label_encoder, deployment_info = load_model()
    model_loaded = True
except:
    st.error("Model files not found. Please upload model files to continue.")
    model_loaded = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box-yes {
        background-color: #DCEDC8;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    .prediction-box-no {
        background-color: #FFCDD2;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Bank Marketing Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Predict if a client will subscribe to a term deposit</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-building.png", width=100)
st.sidebar.title("Navigation")

# Navigation options
page = st.sidebar.radio("Go to", ["Home", "Make Prediction", "Model Insights", "About"])

# Home page
if page == "Home":
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    ## Welcome to the Bank Marketing Prediction App
    
    This application predicts whether a client will subscribe to a term deposit based on various factors.
    
    ### Dataset Information
    The dataset contains information about direct marketing campaigns (phone calls) of a Portuguese banking institution. 
    The classification goal is to predict if the client will subscribe to a term deposit (variable y).
    
    ### How to use this app
    1. Navigate to the "Make Prediction" page to input client details and get a prediction
    2. Explore "Model Insights" to understand the model's performance and important features
    3. Learn more about the project in the "About" section
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    if model_loaded:
        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
        st.markdown("### Model Performance Metrics")

        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{deployment_info['performance_metrics']['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{deployment_info['performance_metrics']['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{deployment_info['performance_metrics']['recall']:.4f}")
        with col4:
            st.metric("F1 Score", f"{deployment_info['performance_metrics']['f1_score']:.4f}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Show model type
        st.info(f"Current Model: {deployment_info['model_name']}")

        # Sample data visualization
        st.markdown("### Sample Data Visualization")
        try:
            # Try to load the original dataset if available
            df = pd.read_csv('bank-additional.csv', sep=';')

            # Create tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Class Distribution", "Feature Correlation", "Age Distribution"])

            with tab1:
                # Class distribution
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(x='y', data=df, ax=ax)
                ax.set_title("Class Distribution in Target Variable")
                ax.set_xlabel("Subscribed Term Deposit")
                ax.set_ylabel("Count")
                st.pyplot(fig)

            with tab2:
                # Correlation heatmap (only numeric columns)
                numeric_df = df.select_dtypes(include=[np.number])
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                ax.set_title("Correlation Matrix")
                st.pyplot(fig)

            with tab3:
                # Age distribution by subscription
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x='age', hue='y', multiple='stack', ax=ax)
                ax.set_title("Age Distribution by Subscription Status")
                ax.set_xlabel("Age")
                ax.set_ylabel("Count")
                st.pyplot(fig)

        except Exception as e:
            st.warning(f"Could not load or visualize sample data: {e}")
            st.write("To view sample visualizations, please make sure the original dataset 'bank-additional.csv' is in the same directory.")
    else:
        st.warning("Model files not found. Please ensure the model files are uploaded correctly.")

# Make Prediction page
elif page == "Make Prediction":
    st.markdown("<h2 class='sub-header'>Make a Prediction</h2>", unsafe_allow_html=True)

    if model_loaded:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.write("Enter client information to predict if they will subscribe to a term deposit.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Create two columns for input fields
        col1, col2 = st.columns(2)

        with col1:
            # Personal information
            st.subheader("Personal Information")
            age = st.slider("Age", min_value=18, max_value=95, value=40)

            job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                           'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
            job = st.selectbox("Job", job_options)

            marital_options = ['divorced', 'married', 'single', 'unknown']
            marital = st.selectbox("Marital Status", marital_options)

            education_options = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
                                 'professional.course', 'university.degree', 'unknown']
            education = st.selectbox("Education", education_options)

            housing_loan = st.selectbox("Housing Loan", ['yes', 'no', 'unknown'])
            personal_loan = st.selectbox("Personal Loan", ['yes', 'no', 'unknown'])

        with col2:
            # Campaign information
            st.subheader("Campaign Information")
            contact_type = st.selectbox("Contact Communication Type", ['cellular', 'telephone'])

            month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            month = st.selectbox("Last Contact Month", month_options)

            day_options = ['mon', 'tue', 'wed', 'thu', 'fri']
            day = st.selectbox("Last Contact Day of Week", day_options)

            campaign_contacts = st.slider("Number of Contacts During This Campaign", min_value=1, max_value=50, value=2)
            previous_contacts = st.slider("Number of Contacts Before This Campaign", min_value=0, max_value=20, value=0)

            # Economic indicators (using data from the deployment info to set reasonable ranges)
            st.subheader("Economic Indicators")
            emp_var_rate = st.slider("Employment Variation Rate", min_value=-4.0, max_value=2.0, value=-0.1, step=0.1)
            cons_price_idx = st.slider("Consumer Price Index", min_value=90.0, max_value=95.0, value=93.2, step=0.1)
            cons_conf_idx = st.slider("Consumer Confidence Index", min_value=-50.0, max_value=-25.0, value=-40.0, step=0.1)
            euribor3m = st.slider("Euribor 3 Month Rate", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
            nr_employed = st.slider("Number of Employees", min_value=4900.0, max_value=5300.0, value=5100.0, step=10.0)

        # Create a dictionary with the input features
        input_dict = {
            'Age': age,
            'Job': job,
            'Marital': marital,
            'Education': education,
            'HousingLoan': housing_loan,
            'PersonalLoan': personal_loan,
            'ContactCommunicationType': contact_type,
            'LastContactMonth': month,
            'LastContactDayOfWeek': day,
            'CampaignContacts': campaign_contacts,
            'PreviousCampaignContacts': previous_contacts,
            'EmploymentVarRate': emp_var_rate,
            'ConsumerPriceIndex': cons_price_idx,
            'ConsumerConfidenceIndex': cons_conf_idx,
            'Euribor3M': euribor3m,
            'NumberOfEmployees': nr_employed
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Prepare data for prediction (this would need to match the preprocessing in your model)
        # Here you would need to add all the preprocessing steps that were used during model training
        # For example, one-hot encoding categorical variables, feature scaling, etc.

        # Education level mapping for ordinal encoding
        education_order = {
            'illiterate': 0,
            'basic.4y': 1,
            'basic.6y': 2,
            'basic.9y': 3,
            'high.school': 4,
            'professional.course': 5,
            'university.degree': 6,
            'unknown': 3  # middle value for unknown
        }

        # Add education level as ordinal feature
        input_df['EducationLevel'] = input_df['Education'].map(education_order)

        # Add season based on month
        season_map = {
            'mar': 'Spring', 'apr': 'Spring', 'may': 'Spring',
            'jun': 'Summer', 'jul': 'Summer', 'aug': 'Summer',
            'sep': 'Fall', 'oct': 'Fall', 'nov': 'Fall',
            'dec': 'Winter', 'jan': 'Winter', 'feb': 'Winter'
        }
        input_df['Season'] = input_df['LastContactMonth'].map(season_map)

        # Add previous contact flag
        input_df['previous_contact'] = (input_df['PreviousCampaignContacts'] > 0).astype(int)

        # Button to make prediction
        if st.button("Make Prediction"):
            try:
                # In a real application, you would need to preprocess the input data exactly as done during training
                # For simplicity, we'll assume the model can handle the direct input after one-hot encoding

                # One-hot encode categorical variables
                categorical_cols = ['Job', 'Marital', 'HousingLoan', 'PersonalLoan',
                                    'ContactCommunicationType', 'LastContactDayOfWeek', 'Season']

                # Get dummies (one-hot encoding)
                input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

                # Ensure all columns from training are present (add missing columns with 0s)
                for col in feature_names:
                    if col not in input_encoded.columns:
                        input_encoded[col] = 0

                # Select only the columns that were used for training
                input_final = input_encoded[feature_names]

                # Make prediction
                prediction = model.predict(input_final)
                prediction_proba = model.predict_proba(input_final)

                # Display prediction
                if prediction[0] == 1:
                    st.markdown("<div class='prediction-box-yes'>", unsafe_allow_html=True)
                    st.markdown("### Prediction: <span style='color:green'>**YES**</span>", unsafe_allow_html=True)
                    st.markdown("This client is likely to subscribe to a term deposit.", unsafe_allow_html=True)
                    st.markdown(f"Probability: {prediction_proba[0][1]:.2%}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='prediction-box-no'>", unsafe_allow_html=True)
                    st.markdown("### Prediction: <span style='color:red'>**NO**</span>", unsafe_allow_html=True)
                    st.markdown("This client is unlikely to subscribe to a term deposit.", unsafe_allow_html=True)
                    st.markdown(f"Probability: {prediction_proba[0][0]:.2%}", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Show top factors influencing the prediction
                st.subheader("Top Factors Influencing This Prediction")

                # This is a simplified version - in a real application you would use SHAP values
                # or another method to interpret the model's prediction
                if "RandomForest" in deployment_info['model_name'] or "GradientBoosting" in deployment_info['model_name']:
                    # For tree-based models, we can extract feature importances
                    importances = model.named_steps['classifier'].feature_importances_
                    feature_importance = pd.Series(importances, index=feature_names)
                    sorted_importances = feature_importance.sort_values(ascending=False)

                    # Display top 10 features
                    st.bar_chart(sorted_importances.head(10))
                else:
                    st.info("Feature importance visualization is only available for tree-based models.")

            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.write("Please check the input values and try again.")
    else:
        st.warning("Model not loaded. Please ensure the model files are uploaded correctly.")

# Model Insights page
elif page == "Model Insights":
    st.markdown("<h2 class='sub-header'>Model Insights</h2>", unsafe_allow_html=True)

    if model_loaded:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.write("Explore the model's performance and understand which features contribute most to the predictions.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Create tabs for different insights
        tab1, tab2, tab3 = st.tabs(["Model Performance", "Feature Importance", "Model Information"])

        with tab1:
            st.subheader("Model Performance Metrics")

            # Create metrics display
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Accuracy", f"{deployment_info['performance_metrics']['accuracy']:.4f}")
                st.metric("Precision", f"{deployment_info['performance_metrics']['precision']:.4f}")

            with col2:
                st.metric("Recall", f"{deployment_info['performance_metrics']['recall']:.4f}")
                st.metric("F1 Score", f"{deployment_info['performance_metrics']['f1_score']:.4f}")

            st.metric("ROC AUC", f"{deployment_info['performance_metrics']['roc_auc']:.4f}")

            # Create and display confusion matrix visualization
            st.subheader("Confusion Matrix Visualization")

            # This is a placeholder - in a real application, you would load the actual confusion matrix
            conf_matrix = np.array([[3000, 200], [150, 769]])  # Example values

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Not Subscribed', 'Subscribed'],
                        yticklabels=['Not Subscribed', 'Subscribed'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title('Confusion Matrix')
            st.pyplot(fig)

        with tab2:
            st.subheader("Feature Importance")

            if "RandomForest" in deployment_info['model_name'] or "GradientBoosting" in deployment_info['model_name']:
                # For tree-based models, we can extract feature importances
                importances = model.named_steps['classifier'].feature_importances_
                feature_importance = pd.Series(importances, index=feature_names)
                sorted_importances = feature_importance.sort_values(ascending=False)

                # Display top 15 features
                fig, ax = plt.subplots(figsize=(10, 8))
                sorted_importances.head(15).plot(kind='barh', ax=ax)
                plt.title("Top 15 Most Important Features")
                plt.xlabel("Relative Importance")
                st.pyplot(fig)

                # Create a table with feature importances
                st.subheader("Feature Importance Table")
                importance_df = pd.DataFrame({
                    'Feature': sorted_importances.index,
                    'Importance': sorted_importances.values
                }).head(15)

                st.dataframe(importance_df, width=600)

            else:
                st.info("Feature importance visualization is only available for tree-based models.")

            # Explain how features affect predictions
            st.subheader("Feature Impact Explanation")
            st.write("""
            ### How to interpret feature importance:
            
            1. **Higher importance values** indicate features that have a stronger influence on the model's predictions.
            
            2. **Economic indicators** like Euribor3M and Employment Variation Rate often have high importance, reflecting the impact of economic conditions on customers' financial decisions.
            
            3. **Contact and communication features** show the effectiveness of different communication strategies.
            
            4. **Demographic information** such as age, job, and education reflect customer segments that are more likely to subscribe.
            """)

        with tab3:
            st.subheader("Model Information")

            # Display model type
            st.info(f"Model Type: {deployment_info['model_name']}")

            # Display model parameters
            st.subheader("Model Parameters")
            st.json(deployment_info['best_parameters'])

            # Display features used
            st.subheader("Features Used")
            st.write(f"Number of features: {len(feature_names)}")
            st.write("List of features:")
            st.write(feature_names)

    else:
        st.warning("Model not loaded. Please ensure the model files are uploaded correctly.")

# About page
elif page == "About":
    st.markdown("<h2 class='sub-header'>About This Project</h2>", unsafe_allow_html=True)

    st.markdown("<div class='info-box'>", unsafe_allow_html=True)
    st.markdown("""
    ### Project Overview
    
    This project aims to predict whether a client will subscribe to a term deposit based on various features related to the client's demographics, previous contacts, and economic indicators.
    
    ### Dataset
    
    The dataset contains information about direct marketing campaigns (phone calls) of a Portuguese banking institution. The marketing campaigns were based on phone calls, and often more than one contact to the same client was required. The classification goal is to predict if the client will subscribe to a term deposit.
    
    ### Model Development Process
    
    1. **Data Cleaning**: Handled missing values, removed duplicates, and addressed data quality issues.
    
    2. **Feature Engineering**: Created new features such as Season, CallDurationCategory, and EducationLevel to improve model performance.
    
    3. **Model Selection**: Compared multiple models including Logistic Regression, Random Forest, Gradient Boosting, and KNN.
    
    4. **Hyperparameter Tuning**: Used GridSearchCV to find the optimal parameters for each model.
    
    5. **Evaluation**: Selected the best model based on F1 score, precision, recall, and accuracy.
    
    6. **Deployment**: Created this Streamlit application for easy use of the model.
    
    ### Course Information
    
    This project was developed as part of the ADA 442 Statistical Learning | Classification course.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    ### References
    
    - [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
    
    - [Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
    """)

    st.markdown("### Developer Information")
    st.write("Created by: Student Name")
    st.write("Course: ADA 442 Statistical Learning | Classification")
    st.write("Instructor: Dr. Hakan Emekci")
    st.write("Date: May 2025")

# Footer
st.markdown("---")
st.markdown("© 2025 Bank Marketing Prediction App | ADA 442 Project")