#!/usr/bin/env python
# coding: utf-8

# ### Getting started notebook for HW2

# #### Import Modules

# In[24]:


#standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#modeling and shap
import shap
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

#metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import pickle


# #### Get Our Feature Table

# In[25]:


#load preprocessed data
stocks_df1=pd.read_csv("hw2_feature_table")
stocks_df1 = stocks_df1.drop(columns = ["Unnamed: 0"])


# In[44]:


stocks_df1.tail(2)


# #### Begin Modeling work here

# In[26]:


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


# In[27]:


X_train, X_test, y_train, y_test = preprocess_modeling_data(stocks_df1, "AAPL", -1)


# ### Features
# 
# The features we chose are 
# 1. Sector_enc
# 2. return_lag1
# 3. return_rolling5
# 4. Volume_lag1
# 5. stock_VIX
# 6. sector_VIX
# 7. moving_avg_50d
# 8. moving_avg_200d
# 9. open_close_over_high_low_ratio_lag1
# 10. exp_moving_avg_50d
# 11. exp_moving_avg_200d
# 12. percent_daily_volume_total_market_lag1
# 
# We picked the above 12 features based off them being fairly uncorrelated to each other, but also due to their shap importances. As we can see below, the SHAP importances/feature importances for each of the features are non-zero values, and we can especially see the moving averages, the stocks VIX, and the sector encoding are quite important features in determening the stocks returns

# ___
# XGBoost Model no gridsearch

# In[28]:


xgb = xgb.XGBRegressor(colsample_bylevel=1, colsample_bynode=1, colsample_bytree=.75, 
                       gamma=0,learning_rate=0.1, max_delta_step=0,missing=-99999, 
                       n_estimators=300, random_state=0,reg_alpha=0, reg_lambda=1, 
                       scale_pos_weight=1, seed=500, silent=None, subsample=.5, verbosity=1)


# In[29]:


xgb.fit(X_train, y_train)


# In[30]:


def display_shap_barplot(df, stock, modeling_days, model, X_train):
    """
    use this to display our shap bar plot
    """
    
    gk_stock = stock

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


# In[31]:


print("Model Day 1 Importances")
display_shap_barplot(stocks_df1, "AAPL", -1, xgb, X_train)


# ___
# LightGBM Model

# In[32]:


import lightgbm as lgbm


# In[33]:


lgbm = lgbm.LGBMRegressor(boosting_type='gbdt', num_leaves=50, max_depth=3, learning_rate=0.1, n_estimators=100,
                          subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, 
                          min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0,
                          colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=0, n_jobs=None,
                          importance_type='split')


# In[34]:


lgbm.fit(X_train,y_train)


# In[35]:


print("Model Day 1 Importances")
display_shap_barplot(stocks_df1, "AAPL", -1, lgbm, X_train)


# ___
# Gradient Boosting Regressor Model

# In[36]:


from sklearn.ensemble import GradientBoostingRegressor


# In[37]:


gbr = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, subsample=1.0, 
                                criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
                                min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 
                                init=None, random_state=500, max_features=None, alpha=0.9, verbose=0, 
                                max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, 
                                n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)


# In[38]:


gbr.fit(X_train,y_train)


# In[39]:


print("Model Day 1 Importances")
display_shap_barplot(stocks_df1, "AAPL", -1, gbr, X_train)


# In[40]:


def scores_per_model_per_day(X_test, y_test):
    gbr_score = gbr.score(X_test, y_test)
    lgbm_score = lgbm.score(X_test, y_test)
    xgb_score = xgb.score(X_test, y_test)

    gbr_score, lgbm_score, xgb_score
    
    gbr_preds = gbr.predict(X_test)
    lgbm_preds = lgbm.predict(X_test)
    xgb_preds = xgb.predict(X_test)
    
    gbr_mse = mean_squared_error(y_test, gbr_preds)
    lgbm_mse = mean_squared_error(y_test, lgbm_preds)
    xgb_mse = mean_squared_error(y_test, xgb_preds)

    gbr_mse, lgbm_mse, xgb_mse
    
    gbr_rmse = mean_squared_error(y_test, gbr_preds, squared = False)
    lgbm_rmse = mean_squared_error(y_test, lgbm_preds, squared = False)
    xgb_rmse = mean_squared_error(y_test, xgb_preds, squared = False)

    gbr_rmse, lgbm_rmse, xgb_rmse
    
    gbr_mae = mean_absolute_error(y_test, gbr_preds)
    lgbm_mae = mean_absolute_error(y_test, lgbm_preds)
    xgb_mae = mean_absolute_error(y_test, xgb_preds)

    gbr_mae, lgbm_mae, xgb_mae
    
    gbr_mape = mean_absolute_percentage_error(y_test, gbr_preds)
    lgbm_mape = mean_absolute_percentage_error(y_test, lgbm_preds)
    xgb_mape = mean_absolute_percentage_error(y_test, xgb_preds)

    gbr_mape, lgbm_mape, xgb_mape
    
    print(f"""
    
    Model Scores:
        GBR: {gbr_score}; LGBM: {lgbm_score}; XGB: {xgb_score}
    MSE:
        GBR: {gbr_mse}; LGBM: {lgbm_mse}; XGB: {xgb_mse}
    RMSE:
        GBR: {gbr_rmse}; LGBM: {lgbm_rmse}; XGB: {xgb_rmse}
    MAE:
        GBR: {gbr_mae}; LGBM: {lgbm_mae}; XGB: {xgb_mae}
    MAPE:
        GBR: {gbr_mape}; LGBM: {lgbm_mape}; XGB: {xgb_mape}
    
    """)


# In[41]:


scores_per_model_per_day(X_test, y_test)


# ___
# As we've chosen XGB, we can now save the model as day1 model since were going to be saving it for every day for 10 days
# 

# In[42]:


pickle.dump(xgb, open('model_day1.pkl', 'wb'))


# ___

# As we've chosen XGBoost for its better results, now run that same process except for 10 days

# In[43]:


#repeat the process for days 2-10 since the above is for day 1

for day in list(range(2, 11, 1)):
    print(f"Model Day {day}")
    X_train, X_test, y_train, y_test = preprocess_modeling_data(stocks_df1, "AAPL", -(day))
    xgb.fit(X_train, y_train)
    pickle.dump(xgb, open(f'model_day{day}.pkl', 'wb'))
    scores_per_model_per_day(X_test, y_test)


# #### Quick Analysis
# 
# The above has given us the models saved as pickle files, and the model results above per day. On average, XGB gave us the best results, though the results for each model tend to be quite close.
# 
# Why selection of these features?
# After multiple runs of the models and looking at their SHAP values, these features tended to come up with the largest SHAP values.
# 
# Things that were excluded after testing were open/close, high/low, 52w high, 52w low, adjusted close, etc... The SHAP values for each of these were significantly less (and almost close to zero) than the features like sector encoding, stock VIX, sector VIX, moving averages, and rolling returns. 
# 
# Why choose XGB?
# As we can see from the model results above, XGB tends to give better results, though not by a great deal. There's obviously fluctuation day by day, but on average the RMSE and MAE tend to be the lowest error, giving us the best result.

# In[ ]:





# In[ ]:




