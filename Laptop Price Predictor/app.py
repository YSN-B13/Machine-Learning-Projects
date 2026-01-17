import streamlit as st
import pickle
import numpy as np
import pandas as pd

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title('Laptop Predictor')

Company = st.selectbox('Brand', df['Company'].unique())

Type = st.selectbox('Type', df['TypeName'].unique())

Ram = st.selectbox('Ram (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

Weight = st.number_input("Weight")

Touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

Ips = st.selectbox('IPS', ['No', 'Yes'])

screen_size = st.number_input("Screen Size")

resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900',
'3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440',
'2304x1440' ])

Cpu = st.selectbox('Cpu', df['Cpu Brand'].unique())

Hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

Ssd = st.selectbox('SSD(in GB)', [0,8, 128, 256, 512, 1024])

Gpu = st.selectbox('Gpu', df['Gpu Brand'].unique())

Os = st.selectbox('OS', df['OS'].unique())

if st.button('Predict Price'):
    if Touchscreen == "Yes":
        Touchscreen = 1
    else :
        Touchscreen = 0

    if Ips == "Yes":
        Ips = 1
    else :
        Ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    Ppi = (((X_res**2) + (Y_res**2))**0.5)/screen_size

    query_data = {
            'Company': [Company],
            'TypeName': [Type],
            'Ram': [Ram],
            'Weight': [Weight],
            'Touchscreen': [Touchscreen],
            'IPS' : [Ips],
            'PPI': [Ppi],
            'Cpu Brand': [Cpu],
            'HDD': [Hdd],
            'SSD': [Ssd],
            'Gpu Brand': [Gpu],
            'OS': [Os]
        }
        
    query_df = pd.DataFrame(query_data)

    prediction = pipe.predict(query_df)

    predicted_price = np.exp(prediction[0]) 
            
    st.success(f"The predicted price of this configuration is: {int(predicted_price)}")