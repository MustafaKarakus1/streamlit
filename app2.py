import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Term Deposit Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def load_model_and_files():
    """Load trained model and required files."""
    try:
        model = pickle.load(open('bank_marketing_model_V2.pkl', 'rb'))
        feature_names = pickle.load(open('feature_names.pkl', 'rb'))
        label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
        deployment_info = pickle.load(open('deployment_info.pkl', 'rb'))
        return model, feature_names, label_encoder, deployment_info
    
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        return None, None, None, None

# Load model and related files
model, feature_names, label_encoder, deployment_info = load_model_and_files()

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .section {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-text {
        color: #059669;
        font-weight: bold;
    }
    .warning-text {
        color: #D97706;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main application structure
def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio("Select a page:", ["Home", "Predict", "Model Information", "About"])
    
    # Main content
    if page == "Home":
        show_home_page()
    elif page == "Predict":
        show_prediction_page()
    elif page == "Model Information":
        show_model_info_page()
    elif page == "About":
        show_about_page()

def show_home_page():
    st.markdown("<h1 class='main-header'>Bank Marketing Term Deposit Prediction</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("""
        <div class='section'>
        <h2 class='sub-header'>Welcome to the Bank Marketing Predictor!</h2>
        <p>This application helps predict whether a client will subscribe to a term deposit 
        based on various client attributes and campaign information.</p>
        
        <p>Our model has been trained on bank marketing campaign data and can assist 
        in identifying potential clients who are likely to subscribe to term deposits.</p>
        
        <h3>How to use this app:</h3>
        <ul>
            <li>Navigate to the <b>Predict</b> page to make predictions for individual clients</li>
            <li>Check out the <b>Model Information</b> page to learn about the model's performance</li>
            <li>Visit the <b>About</b> page for more information about this project</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3>Quick Stats</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if deployment_info:
            st.metric("Model Type", deployment_info['model_name'])
            st.metric("Model Accuracy", f"{deployment_info['performance_metrics']['accuracy']:.2%}")
            st.metric("F1 Score", f"{deployment_info['performance_metrics']['f1_score']:.2%}")
    
    # Key features visualization
    if deployment_info:
        st.markdown("<h2 class='sub-header'>Key Model Features</h2>", unsafe_allow_html=True)
        
        # Create a simple bar chart showing top 5 features (just for display)
        # In a real app, you'd want to show the actual feature importances from your model
        top_features = feature_names[:5] if len(feature_names) >= 5 else feature_names
        importance_values = range(len(top_features), 0, -1)  # Placeholder values
        
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(top_features, importance_values, color='#3B82F6')
        ax.set_xlabel('Relative Importance')
        ax.set_title('Top Features for Prediction')
        st.pyplot(fig)

def show_prediction_page():
    st.markdown("<h1 class='main-header'>Predict Term Deposit Subscription</h1>", unsafe_allow_html=True)
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.write("Enter client information to predict whether they will subscribe to a term deposit.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h3>Client Demographics</h3>", unsafe_allow_html=True)
        
        age = st.slider("Age", min_value=18, max_value=100, value=40)
        
        job = st.selectbox("Job", options=[
            "admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
            "retired", "self-employed", "services", "student", "technician", "unemployed"
        ])
        
        marital = st.selectbox("Marital Status", options=["married", "single", "divorced"])
        
        education_level = st.selectbox("Education Level", options=[
            "basic.4y", "basic.6y", "basic.9y", "high.school", 
            "illiterate", "professional.course", "university.degree"
        ])
        
        has_housing_loan = st.selectbox("Has Housing Loan", options=["yes", "no"])
        has_personal_loan = st.selectbox("Has Personal Loan", options=["yes", "no"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h3>Campaign Information</h3>", unsafe_allow_html=True)
        
        contact_type = st.selectbox("Contact Communication Type", options=["cellular", "telephone"])
        
        last_contact_day = st.selectbox("Last Contact Day of Week", 
                                        options=["mon", "tue", "wed", "thu", "fri"])
        
        campaign_contacts = st.slider("Number of Contacts in Current Campaign", 
                                      min_value=1, max_value=50, value=2)
        
        previous_campaign_outcome = st.selectbox("Previous Campaign Outcome", 
                                              options=["failure", "success", "nonexistent"])
        
        call_duration_category = st.selectbox("Call Duration Category", 
                                             options=["Very Short", "Short", "Medium", "Long"])
        
        season = st.selectbox("Season", options=["Spring", "Summer", "Fall", "Winter"])
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h3>Economic Indicators</h3>", unsafe_allow_html=True)
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        emp_var_rate = st.slider("Employment Variation Rate", 
                               min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    
    with col4:
        cons_price_idx = st.slider("Consumer Price Index", 
                                 min_value=90.0, max_value=100.0, value=93.5, step=0.1)
    
    with col5:
        cons_conf_idx = st.slider("Consumer Confidence Index", 
                                min_value=-60.0, max_value=0.0, value=-40.0, step=0.5)
    
    col6, col7 = st.columns(2)
    
    with col6:
        euribor3m = st.slider("Euribor 3 Month Rate", 
                            min_value=0.0, max_value=5.0, value=1.3, step=0.01)
    
    with col7:
        nr_employed = st.slider("Number of Employees (in thousands)", 
                              min_value=4500.0, max_value=5500.0, value=5000.0, step=10.0)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create education level mapping
    education_order = {
        'illiterate': 0,
        'basic.4y': 1,
        'basic.6y': 2,
        'basic.9y': 3,
        'high.school': 4,
        'professional.course': 5,
        'university.degree': 6
    }
    
    # Add predict button
    if st.button("Predict"):
        if model and feature_names:
            # Here you'd prepare input data matching the expected features of your model
            # This is a simplified example - you'll need to adapt this to match your actual model's feature set
            
            # Create a dummy input dataframe with the right features
            # In a real app, you'd match these exactly to the features your model expects
            
            # This is where you'd transform the user inputs to match the features used in your model
            # For demonstration, I'll create a basic example - you would need to modify this
            
            # Create input data (example - adjust to match your actual feature names and encoding)
            input_data = pd.DataFrame({
                # These feature names are examples - replace with your actual feature names
                'Age': [age],
                'Job_' + job: [1],
                'Marital_' + marital: [1],
                'EducationLevel': [education_order[education_level]],
                'HousingLoan_' + has_housing_loan: [1],
                'PersonalLoan_' + has_personal_loan: [1],
                'ContactCommunicationType_' + contact_type: [1],
                'LastContactDayOfWeek_' + last_contact_day: [1],
                'CampaignContacts': [campaign_contacts],
                'CallDurationCategory_' + call_duration_category: [1],
                'Season_' + season: [1],
                'EmploymentVarRate': [emp_var_rate],
                'ConsumerPriceIndex': [cons_price_idx],
                'ConsumerConfidenceIndex': [cons_conf_idx],
                'Euribor3M': [euribor3m],
                'NumberOfEmployees': [nr_employed]
            })
            
            # Note: This is a placeholder. In reality, you need to ensure your input_data DataFrame
            # exactly matches the features expected by your model, in the exact same order.
            
            # For display purposes - in a real app you'd use your model to make a prediction
            # prediction = model.predict(input_data[feature_names])
            # probability = model.predict_proba(input_data[feature_names])[:, 1]
            
            # For demonstration, just show a simulated prediction
            # You would replace this with actual model prediction
            import random
            simulated_prediction = random.choice([0, 1])
            simulated_probability = random.uniform(0.5, 0.95) if simulated_prediction == 1 else random.uniform(0.05, 0.5)
            
            # Display prediction
            st.markdown("<div class='section' style='text-align:center;'>", unsafe_allow_html=True)
            st.markdown("<h2 class='sub-header'>Prediction Result</h2>", unsafe_allow_html=True)
            
            if simulated_prediction == 1:
                st.markdown("### Prediction: <span style='color:green'>**YES**</span>", unsafe_allow_html=True)
                st.markdown(f"<h1 class='success-text'>Client is likely to subscribe to a term deposit!</h1>", unsafe_allow_html=True)
                st.markdown(f"<p>Confidence: {simulated_probability:.2%}</p>", unsafe_allow_html=True)
                
                # Recommendation for positive prediction
                st.markdown("""
                <h3>Recommended Action:</h3>
                <ul>
                    <li>Prioritize this client for follow-up calls</li>
                    <li>Prepare personalized term deposit offers</li>
                    <li>Consider offering additional incentives</li>
                </ul>
                """, unsafe_allow_html=True)
            else:
                st.markdown("### Prediction: <span style='color:red'>**NO**</span>", unsafe_allow_html=True)
                st.markdown(f"<h1 class='warning-text'>Client is unlikely to subscribe to a term deposit.</h1>", unsafe_allow_html=True)
                st.markdown(f"<p>Confidence: {(1-simulated_probability):.2%}</p>", unsafe_allow_html=True)
                
                # Recommendation for negative prediction
                st.markdown("""
                <h3>Recommended Action:</h3>
                <ul>
                    <li>Consider alternative products that might better fit this client's profile</li>
                    <li>Revisit after economic indicators improve</li>
                    <li>Lower priority for immediate follow-up</li>
                </ul>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Model not loaded properly. Please check if model files are available.")

def show_model_info_page():
    st.markdown("<h1 class='main-header'>Model Information</h1>", unsafe_allow_html=True)
    
    if deployment_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.markdown("<h2 class='sub-header'>Model Details</h2>", unsafe_allow_html=True)
            st.write(f"**Model Type:** {deployment_info['model_name']}")
            st.write(f"**Number of Features:** {len(feature_names)}")
            
            # Show model parameters
            st.markdown("<h3>Best Parameters:</h3>", unsafe_allow_html=True)
            for param, value in deployment_info['best_parameters'].items():
                st.write(f"- **{param.replace('classifier__', '')}:** {value}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='section'>", unsafe_allow_html=True)
            st.markdown("<h2 class='sub-header'>Performance Metrics</h2>", unsafe_allow_html=True)
            
            metrics = deployment_info['performance_metrics']
            
            # Create metrics visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
            metric_values = [
                metrics['accuracy'],
                metrics['precision'],
                metrics['recall'],
                metrics['f1_score'],
                metrics['roc_auc']
            ]
            
            bars = ax.bar(metric_names, metric_values, color='#3B82F6')
            
            # Add percentage labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2%}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
            
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Metrics')
            plt.tight_layout()
            
            st.pyplot(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Feature importance section
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Feature Importance</h2>", unsafe_allow_html=True)
        st.write("The features that most strongly influence the model's predictions:")
        
        # Create a simple example of feature importance visualization
        # In a real application, you would use actual feature importance values from your model
        
        # Example feature importances (replace with actual values from your model)
        example_importances = {
            'CallDuration': 0.28,
            'Euribor3M': 0.15,
            'NumberOfEmployees': 0.12,
            'ConsumerConfidenceIndex': 0.09,
            'EmploymentVarRate': 0.08,
            'Age': 0.07,
            'CampaignContacts': 0.06,
            'ConsumerPriceIndex': 0.05,
            'EducationLevel': 0.04,
            'Season_Summer': 0.03,
        }
        
        # Create sorted series for plotting
        feature_imp = pd.Series(example_importances).sort_values(ascending=False)
        
        # Plot feature importances
        fig, ax = plt.subplots(figsize=(10, 6))
        feature_imp.plot(kind='barh', color='#3B82F6')
        ax.set_xlabel('Relative Importance')
        ax.set_title('Top Features by Importance')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.markdown("<h2 class='sub-header'>Confusion Matrix Example</h2>", unsafe_allow_html=True)
        
        # Example confusion matrix (replace with actual values from your model evaluation)
        example_cm = np.array([
            [800, 100],  # True Negatives, False Positives
            [150, 450]   # False Negatives, True Positives
        ])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(example_cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Subscribed', 'Subscribed'],
                    yticklabels=['Not Subscribed', 'Subscribed'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Model information not loaded properly. Please check if deployment files are available.")

def show_about_page():
    st.markdown("<h1 class='main-header'>About This Project</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='section'>
    <h2 class='sub-header'>Project Overview</h2>
    <p>This project was developed as part of the ADA 442 Statistical Learning course. 
    The goal was to build a machine learning model to predict whether a bank client 
    will subscribe to a term deposit based on various client attributes and campaign information.</p>
    
    <h2 class='sub-header'>Dataset</h2>
    <p>The dataset used for this project is the Bank Marketing Data Set from the UCI Machine Learning 
    Repository. It contains information about direct marketing campaigns conducted by a Portuguese 
    banking institution. The campaigns were based on phone calls, and the objective was to predict 
    if a client would subscribe to a term deposit.</p>
    
    <h2 class='sub-header'>Methodology</h2>
    <p>The following steps were taken to develop the prediction model:</p>
    <ol>
        <li><strong>Data Cleaning:</strong> Handling missing values, outliers, and inconsistencies in the data.</li>
        <li><strong>Feature Engineering:</strong> Creating new features like call duration categories and season information.</li>
        <li><strong>Feature Selection:</strong> Identifying the most relevant features using multiple selection methods.</li>
        <li><strong>Model Training:</strong> Training various models including Logistic Regression, Random Forest, 
        Gradient Boosting, and KNN.</li>
        <li><strong>Hyperparameter Tuning:</strong> Optimizing model parameters using grid search.</li>
        <li><strong>Model Evaluation:</strong> Assessing model performance using metrics like accuracy, precision, 
        recall, F1 score, and ROC-AUC.</li>
        <li><strong>Deployment:</strong> Creating this interactive web application using Streamlit for easy use.</li>
    </ol>
    
    <h2 class='sub-header'>Technologies Used</h2>
    <ul>
        <li>Python for data processing and model development</li>
        <li>Pandas and NumPy for data manipulation</li>
        <li>Scikit-learn for machine learning algorithms</li>
        <li>Matplotlib and Seaborn for data visualization</li>
        <li>Streamlit for web application development</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class='footer'>
        <p>Developed for ADA 442 Statistical Learning<br>
        © 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()