import streamlit as st
import pandas as pd

df = pd.read_csv('csv/laptop_data.csv', index_col=0)
# df['Price'] = df['Price'].apply(lambda x: 0.0110699 * x)
df.rename(columns={'Price':'Price(₹)'}, inplace=True)  

st.title("Original Dataset")
st.dataframe(df, use_container_width=True)


df_ = pd.read_csv('csv/preprocesseddata.csv', index_col=0)

st.title("Preprocessed Dataset")
st.dataframe(df_, use_container_width=True)
