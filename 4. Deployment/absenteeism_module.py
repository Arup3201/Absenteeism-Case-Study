#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pickle

# Create the same CustomScaler class to scale specified columns only (the code is same as previous one)
class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:, ~X.columns.isin(self.columns)]
        return pd.concat([X_scaled, X_not_scaled], axis=1)[init_col_order]
    
# Create the special class that will be used to predict outputs for new data
class absenteeism_model():
    
    def __init__(self, model_file, scaler_file):
        # read model and scaler files which are read
        with open('model', 'rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
            
    # take a data file (*.csv) and preprocess it in the same way
    def load_and_clean_data(self, data_file):
        
        # import data
        df = pd.read_csv(data_file, delimiter=',')
        
        # store the variable in a new variable for later use
        self.df_with_predictions = df.copy()
        
        #drop 'ID' column
        df = df.drop(["ID"], axis=1)
        
        # to preserve the code we've created previously, we will add a new column with 'NaN' values
        df['Absenteeism Time in Hours'] = 'NaN'
        
        # Create a seperate dataframe that will store dummy values of ALL available reasons
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
        
        # split reason columns into 4 types
        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)
        
        # to avoid multicollinearity problem, drop the 'Reason for Absence' column
        df = df.drop(['Reason for Absence'], axis=1)
        
        # concatenate df and 4 types of reasons for absence
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
        
        # assign names to the reason for absence columns
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                       'Daily Work Load Average', 'Body Mass Index', 'Education',
                       'Children', 'Pets', 'Absenteeism Time in Hours', 'Reason_1', "Reason_2", "Reason_3", "Reason_4"]
        df = df[column_names]
        
        # reorder the columns
        column_names_reordered = ['Reason_1', "Reason_2", "Reason_3", "Reason_4", 
                          'Date', 'Transportation Expense', 'Distance to Work', 'Age', 
                          'Daily Work Load Average', 'Body Mass Index', 'Education',
                          'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]
        
        # convert the 'Date' column into datetime
        df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
        
        # Create a list with months retreived from the 'Date' column
        list_months = []
        for i in range(df_reason_mod.shape[0]):
            list_months.append(df_reason_mod['Date'][i].month)
            
        # Insert the values in df, inside new column named 'Month Value'
        df['Month Value'] = list_months
        
        # Create a new column to store the day of the week as 'Day of the Week'
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())
        
        # Drop the 'Date' column
        df = df.drop(['Date'], axis=1)
        
        # reorder the column names
        column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value',
                                   'Day of the Week', 'Transportation Expense', 'Distance to Work', 'Age',
                                   'Daily Work Load Average', 'Body Mass Index', 'Education',
                                   'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_upd]
        
        # Map the 'Education' variables; the results are dummy
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})
        
        # fill 'NaN' with 0
        df = df.fillna(value=0)
        
        # Drop the 'Absenteeism Time in Hours' column
        df = df.drop(['Absenteeism Time in Hours'], axis=1)
        
        # Drop the columns that were decided to be left
        df = df.drop(['Day of the Week', 'Distance to Work', 'Daily Work Load Average'], axis=1)
        
        # we have this line in case you want to call the 'preprocessed data'
        self.preprocessed_data = df.copy()
        
        # We need this line so that we can use it in the next function
        self.data = self.scaler.transform(df)
        
    # a function that outputs probablity of a data point to be 1
    def predicted_probablity(self):
        if (self.data is not None):
            pred = self.reg.predict_proba(self.data)[:, 1]
            return pred
        
    # a function which outputs 0 or 1 based on our model
    def predicted_output_category(self):
        if (self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
    
    # a function to predict output and probablities and
    # add columns with these values at the end of the new data
    def predicted_outputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probablity'] = self.reg.predict_proba(self.data)[:, 1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data

