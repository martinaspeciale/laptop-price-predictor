import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

icon = "icon.png"
st.sidebar.image(icon, caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")

# Define the feature names
feature_names = [
    'Company', 'TypeName', 'RAM(GB)', 'OpSys', 'Weight(kg)',
    'TouchScreen', 'IPS', 'PPI', 'CPU', 'HDD', 'SSD', 'GPU brand'
]

# Create the Streamlit app
st.title('Laptop Price Predictor')

data = pd.read_csv("csv/preprocesseddata.csv", index_col=0)

# Store the RMSE to compute the Prediction Interval
df_rmse = pd.read_csv("csv/rmse.csv", index_col='Model')
dict_rmse = dict(zip(df_rmse.index, df_rmse.RMSE))


# Decide model to use 
default_model = st.toggle("Default Model", True)

models = {
    'Linear Regression' : 'lr', 
    'Ridge Regression' : 'ridge', 
    'Lasso Regression' : 'lasso', 
    'Decision Tree' : 'dt', 
    'K-Nearest Neighbors (KNN)' : 'knn'
}


if not default_model:
    to_use = st.selectbox('Model', ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Decision Tree', 'K-Nearest Neighbors (KNN)'])


# Create input fields for each feature
input_data = {}
input_data['Company'] = st.selectbox('Company', data['Company'].unique())
input_data['TypeName'] = st.selectbox('TypeName', data['TypeName'].unique())
input_data['RAM(GB)'] = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
input_data['OpSys'] = st.selectbox('OpSys', data['OpSys'].unique())
input_data['Weight(kg)'] = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, step=0.1)
input_data['TouchScreen'] = st.selectbox('TouchScreen', ['No', 'Yes'])
input_data['IPS'] = st.selectbox('IPS', ['No', 'Yes'])

inches = st.number_input('Screen Size (inches)', min_value=11.0, step=0.1)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
input_data['PPI'] = None

input_data['CPU'] = st.selectbox('CPU', data['CPU'].unique())
input_data['HDD'] = st.number_input('HDD (GB)', min_value=0, step=128)
input_data['SSD'] = st.number_input('SSD (GB)', min_value=0, step=128)
input_data['GPU brand'] = st.selectbox('GPU Brand', data['GPU Brand'].unique())


# When the user clicks the 'Predict' button
if st.button('Predict'):

    if default_model:
        st.write("Selected Default Model: Random Forest")
        rmse = dict_rmse['Default']
        #st.write(rmse)
        # Load the pre-trained model (the one that performs best)
        with open('pipe.pkl', 'rb') as file:
            model = pickle.load(file)

    else :
        st.write(f'Selected Model: \t {to_use}')
        st.write(f'opening: pipes/pipe_{models[to_use]}.pkl')

        rmse = dict_rmse[to_use]
        #st.write(rmse)
        #st.write(f'Prediction Interval computed using RMSE: \t {rmse}')
        with open(f'pipes/pipe_{models[to_use]}.pkl', 'rb') as file:
            model = pickle.load(file)


    # Convert categorical binary inputs to numeric
    input_data['TouchScreen'] = 1 if input_data['TouchScreen'] == 'Yes' else 0
    input_data['IPS'] = 1 if input_data['IPS'] == 'Yes' else 0

    # Compute PPI 
    X_resolution = int(resolution.split('x')[0])
    Y_resolution = int(resolution.split('x')[1])
    input_data['PPI'] = ((X_resolution**2)+(Y_resolution**2))**0.5/(inches)

    # os.write(1,b'Something was executed.\n')


    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Apply the column transformer
    transformed_input = model.named_steps['step1'].transform(input_df)

    # Predict using the trained model
    prediction = model.named_steps['step2'].predict(transformed_input)

    # Calculate the exponential of the predicted value
    exp_prediction = np.exp(prediction[0])
    str_prediction = str(round(exp_prediction,2))

    # Display the prediction
    st.title('Predicted Price:')
    # st.title(f':green[€{exp_prediction:.2f}]')
    st.markdown(f"<h1 style='text-align: center; color: #168118;'><span style='color:#FFD700;'>€</span> {str_prediction}</h1>", unsafe_allow_html=True)
    st.title('Prediction Interval: ')
    # st.title(f'€{exp_prediction:.2f} - :red[{2*rmse:.2f}]; {exp_prediction:.2f} + :red[{2*rmse:.2f}]')
    lb = np.exp(prediction[0] - 2*rmse)
    ub = np.exp(prediction[0] + 2*rmse)
    st.markdown(f"<h1 style='text-align: center;'><span style='color:#FFD700;'>€</span> [ <span style='color: #168118;'>{lb:.2f}</span>; <span style='color: #168118;'>{ub:.2f} </span>]</h1>", unsafe_allow_html=True)




