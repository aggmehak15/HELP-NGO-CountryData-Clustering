import streamlit as st
import numpy as np
import pandas as pd
import joblib

#Lets load the joblib instances over here
with open('pipeline.joblib','rb') as file:
    preprocess= joblib.load(file)

with open('model.joblib','rb') as file:
    model= joblib.load(file)

#Lets take the inputs from the user
st.title('HELP NGO Organization')
st.subheader('This application will help to identify the development category of a country using socio-economic factors. Original data had been clustered using KMeans')

#Lets take inputs:
gdpp= st.number_input('Enter the GDPP of a country')
income= st.number_input('Enter the income of a country')
imports= st.number_input('Enter the imports of goods and services per capita. Given as %age of the GDP per capita')
exports= st.number_input('Enter the exports of goods and services per capita. Given as %age of the GDP per capita')
inflation= st.number_input('Inflation: The measurement of the annual growth rate of the Total GDP')
lf_expcy= st.number_input('Life Expectancy: The average number of years a new born child would live if the current mortality patterns are to remain the same')
fert= st.number_input('Fertility: The number of children that would be born to each woman if the current age-fertility rates remain the same.')
health= st.number_input('Total health spending per capita. Given as %age of GDP per capita')
child_mort= st.number_input('Child Mortality: Death of children under 5 years of age per 1000 live births')

input_list= [child_mort, exports,health, imports,income, inflation, lf_expcy, fert, gdpp]

final_input_list= preprocess.transform([input_list])

if st.button('Predict'):
    prediction= model.predict(final_input_list)[0]
    if prediction==0:
        st.success('Developing')
    elif prediction==1:
        st.success('Developed')
    else:
        st.error('Underdeveloped')