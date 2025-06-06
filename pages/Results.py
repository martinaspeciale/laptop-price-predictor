import streamlit as st
import pandas as pd

df = pd.read_csv('csv/performance_obtained.csv', index_col=0)
df_paper = pd.read_csv('csv/paper_performance.csv', index_col=0)

st.title("Performance Obtained")
st.dataframe(df, use_container_width=True)

st.title("Performance from Paper")
st.dataframe(df_paper, use_container_width=True)
url = 'https://ieeexplore.ieee.org/document/10093357'
st.write("Reference to the article: [Laptop Price Prediction using Machine Learning Algorithms](%s)" % url)
