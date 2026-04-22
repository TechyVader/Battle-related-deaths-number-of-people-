
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind


df = pd.read_csv("Video_Games.csv")

print(df.head())
print(df.info())
print(df.describe())

# Handle missing values
df = df.dropna()

# =========================
# Visualization
# =========================
sns.set_style("whitegrid")
sns.set_palette("Set2")   # try: Set1, Set3, coolwarm, viridis
plt.rcParams['figure.figsize'] = (10,6)

# Top 10 Genres
##plt.figure(figsize=(10,5))
##df['Genre'].value_counts().head(10).plot(kind='bar')
##plt.title("Top Genres")
##plt.xlabel("Genre")
##plt.ylabel("Count")
##plt.show()
plt.figure()
df['Genre'].value_counts().head(10).plot(
    kind='bar',
    color=sns.color_palette("viridis", 10),
    edgecolor='black'
)

plt.title("Top 10 Game Genres 🎮", fontsize=16, fontweight='bold')
plt.xlabel("Genre")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Platform distribution
##plt.figure(figsize=(10,5))
##sns.countplot(x='Platform', data=df)
##plt.xticks(rotation=90)
##plt.title("Platform Distribution")
##plt.show()
plt.figure()
sns.countplot(
    x='Platform',
    data=df,
    palette='coolwarm'
)

plt.title("Platform Distribution 📊", fontsize=16, fontweight='bold')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Sales distribution
##plt.figure(figsize=(8,5))
##sns.histplot(df['Global_Sales'], kde=True)
##plt.title("Global Sales Distribution")
##plt.show()
plt.figure()
sns.histplot(
    df['Global_Sales'],
    kde=True,
    color='purple',
    bins=30
)

plt.title("Global Sales Distribution 📈", fontsize=16, fontweight='bold')
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()

# Boxplot (outliers)
##sns.boxplot(x=df['Global_Sales'])
##plt.title("Outlier Detection")
##plt.show()
plt.figure()
sns.boxplot(
    x=df['Global_Sales'],
    color='orange'
)

plt.title("Outlier Detection 📦", fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.show()
# =========================
# EDA
# =========================

print(df.describe())

#Correlation
##corr = df.corr(numeric_only=True)
##sns.heatmap(corr, annot=True)
##plt.title("Correlation Matrix")
##plt.show()
plt.figure(figsize=(8,6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap='coolwarm',linewidths=0.5,fmt=".2f")

plt.title("Correlation Matrix 🔥", fontsize=16, fontweight='bold')
plt.show()

# Outlier Detection (IQR)
Q1 = df['Global_Sales'].quantile(0.25)
Q3 = df['Global_Sales'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['Global_Sales'] < Q1 - 1.5*IQR) |
              (df['Global_Sales'] > Q3 + 1.5*IQR)]

print("Outliers:", outliers.shape[0])

#Statistical Analysis

print("Mean:", df['Global_Sales'].mean())
print("Median:", df['Global_Sales'].median())
print("Std Dev:", df['Global_Sales'].std())

#Hypothesis Testing
#T-Test

t_stat, p_val = ttest_1samp(df['Global_Sales'], 1)
print(t_stat, p_val)


#Probability Distribution
sns.histplot(df['Global_Sales'], kde=True)
plt.show()



# =========================
# Machine Learning (Prediction)
# =========================



# Encode categorical data
le = LabelEncoder()
df['Platform'] = le.fit_transform(df['Platform'])
df['Genre'] = le.fit_transform(df['Genre'])
df['Publisher'] = le.fit_transform(df['Publisher'])
X = df[['Platform','Genre','Year_of_Release']]
y = df['Global_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

#Visualization of Prediction

plt.figure()
plt.scatter(
    y_test,
    y_pred,
    color='blue',
    alpha=0.6,
    edgecolor='black'
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linewidth=2
)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales 🎯", fontsize=16, fontweight='bold')
plt.grid(alpha=0.3)
plt.show()





























