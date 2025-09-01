import streamlit as st
import joblib
import pandas as pd

st.title('Bank Customer Churn Prediction')

# Load the saved model and scaler
model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

st.header('Enter Customer Details:')

# Input fields for customer data
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
country_options = {'France': 0, 'Germany': 1, 'Spain': 2}
country = st.selectbox('Country', country_options.keys())
gender_options = {'Female': 0, 'Male': 1}
gender = st.selectbox('Gender', gender_options.keys())
age = st.number_input('Age', min_value=18, max_value=100, value=40)
tenure = st.number_input('Tenure (years)', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0.0, value=75000.0)
products_number = st.number_input('Number of Products', min_value=1, max_value=4, value=2)
credit_card_options = {0: 'No', 1: 'Yes'}
credit_card = st.selectbox('Has Credit Card?', credit_card_options.keys(), format_func=lambda x: credit_card_options[x])
active_member_options = {0: 'No', 1: 'Yes'}
active_member = st.selectbox('Is Active Member?', active_member_options.keys(), format_func=lambda x: active_member_options[x])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=100000.0)

if st.button('Predict Churn'):
    # Create a DataFrame from the input data
    input_data = {
        'credit_score': [credit_score],
        'country': [country_options[country]],
        'gender': [gender_options[gender]],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'products_number': [products_number],
        'credit_card': [credit_card],
        'active_member': [active_member],
        'estimated_salary': [estimated_salary]
    }
    input_df = pd.DataFrame(input_data)

    # Scale the input data
    scaled_input_data = scaler.transform(input_df)

    # Make a prediction
    churn_prediction = model.predict(scaled_input_data)

    # Display the prediction result
    st.subheader("Prediction Result:")
    if churn_prediction[0] == 1:
        st.write("The customer is predicted to churn.")
    else:
        st.write("The customer is predicted not to churn.")
