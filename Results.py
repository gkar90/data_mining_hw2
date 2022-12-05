#!/usr/bin/env python
# coding: utf-8

# # Results
# 

# Homework can be found at:
#     
#     'https://github.com/gkar90/data_mining_hw2'
#     
# 

# ## Part A

# This project is broken down into 3 seperate notebooks:
# 1. Data Preprocessing
# 2. Modeling
# 3. Shap Values
# 
# In the data preprocessing notebook, we take the original CSV file that we had, and we clean, process, and feature engineer on it. We took the original dataset which had just the date, symbol, volume,  open/close prices, and high/low prices, and we created features that tracked numerous different things such as :
# 1. Sectors and their encodings
# 2. Rolling Volumes and Returns
# 3. Volatility Trackers
# 4. High/Low prices and ratio's to intraday moves
# 5. Various Moving Averages
# 
# When running multiple iterations of the model, highly correlated features like 52w_h, 52w_l, open/close, high/low, etc...really had little impact, as shown by their low SHAP values and low feature importances. On the other hand, things like the VIX features, the moving averages, and the sector (both the vix and the actual sector) had quite big importances in determining which model to pick. As the prompt asked that we stay within 10-15 features, the features I chose are in the Modeling notebook, which in essence are the stocks sector, volatility, returns, and moving averages.
# 
# In the Modeling notebook, we run the model off the feature table created in the Data Preprocessing notebook. In here, we tested 3 different models: XGBoost, LightGBM, and GradientBoostingRegressor. For the most part, the models ended up being relatively close (looking at RMSE and MAE, the results of each model per day are not that far apart)...however, due to familiarity, and better results on some days, I chose the XGBoost model, which is the model that is saved in the pickle file for modeling each day. In this notebook we also follow the prompt of following hte last 10 days, and we have a loop that saves the data and the models to be used in the SHAP Values notebook next.
# 
# Lastly we have the Shap Values notebook, which takes in the daily model pickle file, and then creates the SHAP Bar plot for each day to show the importances of the features on those days. Interestingly, the features do vary quite a bit day to day, which is interesting as it shows that every day a new variable/feature could be impacting the returns. This contradicts a bit the technical analysis believers who swear solely on the moving averages, as we can see that there are more factors at play than just the moving averages (some days the VIX is tops, some days its the sector movement, other days its the rolling returns...) 
# 
# 
# Lastly, I was able to creat the Docker Image, however I kept running into an error on building the container (not sure why lightgbm wouldnt load).. I've attached the image of the broken container.
# 
# Also, hosting the shap barplots on Flask also gave me 401 errors (which you can see in the code). Also attached an image there.

# In[ ]:




