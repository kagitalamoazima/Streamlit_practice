from ucimlrepo import fetch_ucirepo, list_available_datasets
list_available_datasets()
#heart_disease = fetch_ucirepo(id=42)
df = fetch_ucirepo(name="Heart Disease")
X = df.data.features
y = df.data.targets
X
X.shape
y.shape
import pandas as pd 
import numpy as np
df = pd.concat([X,y],axis=1)
df.shape
df.info()
df.isna().sum()
df.dropna(inplace=True)
df.shape
import streamlit as sl
Category  = sl.selectbox("Select category",df["sex"].unique())
