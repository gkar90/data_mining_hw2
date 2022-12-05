#!/usr/bin/env python
# coding: utf-8

# ### Getting started notebook for HW2

# #### Import Modules

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import shap

import pickle

# pd.set_option('display.max_rows', None)


# #### Import our Feature Table

# In[43]:


#load preprocessed data
stocks_df1=pd.read_csv("hw2_feature_table")
stocks_df1 = stocks_df1.drop(columns = ["Unnamed: 0"])


# ___
# #### Create Clean Functions for Viewing SHAP 

# In[40]:


def preprocess_modeling_data(df, stock, modeling_days):
    """
    modularize the preprocessing for easy manipulation
    """
    
    gk_stock = stock

    #use this section to pick/change/adjust the features
    feature_list = ["Sector_enc", "return_lag1", "return_rolling5", 'Volume_lag1', 
                    "stock_VIX", "sector_VIX", "moving_avg_50d", "moving_avg_200d", 'open_close_over_high_low_ratio_lag1',
                    "exp_moving_avg_50d", "exp_moving_avg_200d", 'percent_daily_volume_total_market_lag1']

    recent_date = df.loc[df.index[modeling_days],'Date']
    
    #create a list of today's stocks EXCLUDING the one we are interested in
    today_stocks = stocks_df1[np.logical_and(stocks_df1['Date'] == recent_date, 
                                             stocks_df1['Symbol'] != gk_stock)]
    
    #create a train/test split for early stopping. Just using about 50 stocks
    X_train, X_test, y_train, y_test = train_test_split(today_stocks[feature_list], 
                                                        today_stocks['return'], 
                                                        test_size=0.1, 
                                                        random_state=500)
    
    return X_train, X_test, y_train, y_test


# In[41]:


def display_shap_barplot(df, stock, modeling_days, model, X_train):
    """
    use this to display our shap bar plot
    """
    
    gk_stock = stock

    #use this section to pick/change/adjust the features
    feature_list = ["Sector_enc", "return_lag1", "return_rolling5", 'Volume_lag1', 
                    "stock_VIX", "sector_VIX", "moving_avg_50d", "moving_avg_200d", 'open_close_over_high_low_ratio_lag1',
                    "exp_moving_avg_50d", "exp_moving_avg_200d", 'percent_daily_volume_total_market_lag1']

    recent_date = df.loc[df.index[modeling_days],'Date']
    
   #input for only this stock
    this_data = df[np.logical_and(df['Date'] == recent_date,
                                  df['Symbol'] == gk_stock)][feature_list]

    this_actual = df[np.logical_and(df['Date'] == recent_date, 
                                    df['Symbol'] == gk_stock)]['return']
    
    this_data_actual = this_data.copy()

    # compute SHAP values
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    
    shap.plots.bar(shap_values, max_display = 15)


# In[42]:


#View Our SHAP Plots below

for day in list(range(1, 11, 1)):
    xgb = pickle.load(open(f'model_day{day}.pkl', 'rb'))
    print(f"Model Day {day}")
    X_train, X_test, y_train, y_test = preprocess_modeling_data(stocks_df1, "AAPL", -(day))
    print(f"Model Day {day} Importances")
    display_shap_barplot(stocks_df1, "AAPL", -(day), xgb, X_train)


# In[ ]:





# ___
# Flask below

# In[28]:


from flask import Flask, request


# In[29]:


with open("model.pkl", "rb") as f:
    model = pickle.load(f)
    
app = Flask(__name__)


# In[32]:


if __name__ == "__main__":
    app.run()


# In[ ]:




