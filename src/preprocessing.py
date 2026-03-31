"""Module for data preprocessing and feature engineering"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fill_missing_values(df):
    """Fill missing values in Sleep Disorder column"""
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('healthy')
    return df

def split_blood_pressure(df):
    """Split blood pressure into high and low components"""
    df['BP_High'] = df['Blood Pressure'].apply(lambda x: x.split('/')[0]).astype(int)
    df['BP_Low'] = df['Blood Pressure'].apply(lambda x: x.split('/')[1]).astype(int)
    df = df.drop('Blood Pressure', axis=1)
    return df

def replace_bmi_values(df):
    """Replace normal weight with normal in BMI Category"""
    df['BMI Category'] = df['BMI Category'].replace('Normal Weight', 'Normal')
    return df

def encode_features(df):
    """
    Label encode BMI Category and Sleep Disorder, 
    One-hot encode Gender and Occupation
    """
    le = LabelEncoder()
    
    # Label encode BMI Category
    df['BMI Category'] = le.fit_transform(df['BMI Category'])
    bmi_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("BMI Category Encoding:", bmi_mapping)
    
    # Label encode Sleep Disorder
    df['Sleep Disorder'] = le.fit_transform(df['Sleep Disorder'])
    disorder_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Sleep Disorder Encoding:", disorder_mapping)
    
    # One-hot encode Gender and Occupation
    df = pd.get_dummies(df, columns=['Gender', 'Occupation'], dtype=int)
    
    return df, bmi_mapping, disorder_mapping

def preprocess_data(df):
    """Pipeline to preprocess the entire dataset"""
    df = fill_missing_values(df)
    df = split_blood_pressure(df)
    df = replace_bmi_values(df)
    df, bmi_map, disorder_map = encode_features(df)
    return df, bmi_map, disorder_map

def get_column_types(df):
    """Get categorical and numerical columns"""
    cat_colm = df.select_dtypes(include=['object']).columns
    num_colm = df.select_dtypes(include=['int64', 'float64']).columns
    return cat_colm, num_colm