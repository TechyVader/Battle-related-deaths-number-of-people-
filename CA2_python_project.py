import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
df = pd.read_csv("WB_WDI_VC_BTL_DETH.csv")
print(df)
print(df.info())
print("-------------------------")
print(df.describe())
print("-------------------------")
print(df.isnull().sum())
