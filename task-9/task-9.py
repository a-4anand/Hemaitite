import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_scoredea

def load_data(filename):
    return pd.read_csv(filename)

def check_missing_values(df):
    return df.isnull().sum()

def statistical_summary(df):
    return df.describe()

def plot_target_distribution(df):
    sns.countplot(x='target', data=df)
    plt.title('Count of Each Target Value')
    plt.show()

def visualize_relationships(df, columns):
    sns.pairplot(df[columns], hue='target')
    plt.show()

def correlation_heatmap(df):
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.show()

def split_data(df):
    X = df.drop(columns=['target'])
    y = df['target']
    return train_test_split(X, y, test_size=0.1, random_state=101)

def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

df = load_data('heart.csv')
print(df.head())
print(check_missing_values(df))
print(statistical_summary(df))
plot_target_distribution(df)
visualize_relationships(df, ['age', 'trestbps', 'chol', 'thalach', 'target'])
correlation_heatmap(df)
X_train, X_test, y_train, y_test = split_data(df)
X_train_scaled, X_test_scaled = normalize_data(X_train, X_test)
model = train_model(X_train_scaled, y_train)
accuracy = evaluate_model(model, X_test_scaled, y_test)
print(f'Model Accuracy: {accuracy:.2f}')
