import streamlit as st
import pandas as pd 
from src.logger import logging 
import pickle
import os
import sys
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

st.sidebar.title('Input')
carat=st.number_input('carat'),
depth=st.number_input('depth'),
table=st.number_input('table'),
x=st.number_input('x'),
y=st.number_input('y'),
z=st.number_input('z'),
cut=st.selectbox('cut',['Premium','Very Good','Ideal','Good','Fair']),
color=st.selectbox('color',['F','J','G','E','D','H','I']),
clarity=st.selectbox('clarity',['VS2','SI2','VS1','SI1','IF','VVS2','VVS1','I1'])

submitted=st.sidebar.button('Submit')




if submitted:
    
    # Create CustomData object
    data = CustomData(carat=carat, depth=depth, table=table, x=x, y=y, z=z, cut=cut, color=color, clarity=clarity)
    
    pred_df=data.get_data_dataframe()
    
    st.write(pred_df)
    
    predict_pipeline=PredictPipeline()
    
    result=predict_pipeline.predict(pred_df)
    
    
    # Display prediction or any other result
    st.write("Prediction:", result[0])
    




