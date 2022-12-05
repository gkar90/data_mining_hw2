#!/usr/bin/env python
# coding: utf-8

# ### Data Preprocessing

# #### Import Modules

# In[26]:


import pandas as pd
import numpy as np

from sklearn.preprocessing import OrdinalEncoder

import shap
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split


# #### Get Data from CSV

# In[8]:


all_stocks=pd.read_csv("sp500_stocks.csv")
sector=pd.read_csv("sp500_companies.csv")

all_stocks=all_stocks.merge(sector[['Symbol','Sector']],how='left',on='Symbol')


# In[12]:


#we sill have some missing sectors - figure out what to do here
all_stocks[all_stocks['Sector'].isnull()]['Symbol'].unique()


# In[13]:


def remove_sector_nulls(df):
    """
    These are the stocks that have NaN under Sector. We fill these in with their proper sectors
    """
    df.loc[df['Symbol']=='CEG','Sector']='Utilities'
    df.loc[df['Symbol']=='ELV','Sector']='Healthcare'
    df.loc[df['Symbol']=='META','Sector']='Communication Services'
    df.loc[df['Symbol']=='PARA','Sector']='Communication Services'
    df.loc[df['Symbol']=='SBUX','Sector']='Consumer Cyclical'
    df.loc[df['Symbol']=='V','Sector']='Financial Services'
    df.loc[df['Symbol']=='WBD','Sector']='Communication Services'
    df.loc[df['Symbol']=='WTW','Sector']='Financial Services'
    df.loc[df['Symbol']=='GEN','Sector']='Communication Services'
    
    return df
    


# In[14]:


all_stocks = remove_sector_nulls(all_stocks)


# ### In order to understand what factors may be driving returns we first need to calcualte returns

# In[16]:


#calculate return as a log-difference
all_stocks = all_stocks.sort_values(['Symbol','Date']).reset_index(drop=True)

all_stocks['adj_close_lag1'] = all_stocks[['Symbol','Date','Adj Close']].groupby(['Symbol']).shift(1)['Adj Close'].reset_index(drop=True)

all_stocks['return']=np.log(all_stocks['Adj Close']/all_stocks['adj_close_lag1'])


# ### Think about how to use the other features - DO NOT USE FEATURES FROM TODAY TO MODEL TODAY'S RETURN

# ### Feature Engineering Below

# In[17]:


def create_lagged_features(df,var):
    df[var+'_lag1']=df[['Symbol','Date',var]].groupby(['Symbol']).shift(1)[var].reset_index(drop=True)
    df[var+'_rolling5']=df[['Symbol','Date',var+'_lag1']].groupby(['Symbol'])[var+'_lag1'].rolling(5).sum().reset_index(drop=True)
    df[var+'_rolling15']=df[['Symbol','Date',var+'_lag1']].groupby(['Symbol'])[var+'_lag1'].rolling(15).sum().reset_index(drop=True)
    return df


# In[18]:


all_stocks = create_lagged_features(all_stocks, 'return')
all_stocks = create_lagged_features(all_stocks, 'Volume')


# In[19]:


all_stocks['relative_vol_1_15'] = all_stocks['Volume_lag1']/all_stocks['Volume_rolling15']
all_stocks['relative_vol_5_15'] = all_stocks['Volume_rolling5']/all_stocks['Volume_rolling15']


# ### Transform Sector for modeling

# In[35]:


#perform frequency based encoding 
#(usually this would only use training porttion to fit transform, but need to keep transform constant across days)

sector_counts = all_stocks['Sector'].value_counts()

enc = OrdinalEncoder(categories=[list(sector_counts.index)])


# In[36]:


all_stocks['Sector_enc']=enc.fit_transform(all_stocks[['Sector']])


# ___
# 
# So essentially what we have to do here is start feature engineering. After some research, the features I will add onto the dataset, on top of the features already made, will be:
# 
# 1. Each stock's daily volatility (calculated as the VIX -- labeled stock_VIX)
# 2. Each sectors daily volatility (sector_VIX)
# 3. The Open/Close Price ratio over the High/Low price ratio (open_close_over_high_low_ratio_lag1)
# 4. 52 week highs and lows (52w_h, 52w_l)
# 5. The percent daily volume that stock makes up of the entire stock market (percent_daily_volume_total_market_lag1)
# 6. The simple moving average (10, 50, and 200 day)
# 7. The exponential moving average 
# 
# After some research, it was found that the various moving averages make up fundamental technical analysis in stock trading, and this is it's own school of thought, so my idea is that these features must be important in modeling returns if there is an entire school of thought behind it. 

# #### Feature Engineering HW Section

# In[37]:


stocks_df = all_stocks.copy()


# In[38]:


#calculate the stocks volatility
stocks_df['stock_VIX'] = stocks_df["return"].rolling(30).std(ddof=0)*np.sqrt(252)

#caclulate the sectors volatility
sum_vol_daily = stocks_df.groupby(["Sector_enc", "Date"])['return'].mean().reset_index()
sum_vol_daily['sector_VIX'] = sum_vol_daily["return"].rolling(30).std(ddof=0)*np.sqrt(252)


# In[39]:


stocks_df1 = pd.merge(stocks_df, 
               sum_vol_daily[["Sector_enc", "Date", "sector_VIX"]], 
               on = ['Sector_enc', 'Date'], 
               how = 'left')


# In[40]:


#calculate the o/c h/l ratio
stocks_df1["open_close_over_high_low_ratio_lag1"] = ((stocks_df1['Open'].shift(1)                                                      /stocks_df1["Close"].shift(1))-1)/((stocks_df1['High'].shift(1)   /stocks_df1["Low"].shift(1))-1)


# In[41]:


#calculate the different moving averages
stocks_df1['moving_avg_10d'] = stocks_df1["Adj Close"].rolling(10).mean()
stocks_df1['moving_avg_50d'] = stocks_df1["Adj Close"].rolling(50).mean()
stocks_df1['moving_avg_200d'] = stocks_df1["Adj Close"].rolling(200).mean()

stocks_df1['exp_moving_avg_10d'] = stocks_df1["Adj Close"].ewm(10, adjust = False).mean()
stocks_df1['exp_moving_avg_50d'] = stocks_df1["Adj Close"].ewm(50, adjust = False).mean()
stocks_df1['exp_moving_avg_200d'] = stocks_df1["Adj Close"].ewm(200, adjust = False).mean()


# In[42]:


daily_volume = stocks_df1.groupby(["Date"])["Volume"].sum()

stocks_df1 = pd.merge(stocks_df1, daily_volume, on = "Date", how = "left")


# In[43]:


#calculate the stocks daily volume percentage of the entire market
stocks_df1['percent_daily_volume_total_market_lag1'] = stocks_df1["Volume_x"].shift(1)/stocks_df1["Volume_y"].shift(1)

stocks_df1 = stocks_df1.drop(columns = ["Volume_y"]).rename(columns = {"Volume_x" : "Volume"})


# In[44]:


#calculate the 52 week high and lows
#note -- only 252 trading days in the year
stocks_df1["52w_h"] = stocks_df1["High"].rolling(252).max() 
stocks_df1["52w_l"] = stocks_df1["Low"].rolling(252).min() 


# In[45]:


stocks_df1.tail()


# In[46]:


#write the feature table back to csv so other notebooks can access it
stocks_df1.to_csv("hw2_feature_table")


# ___

# In[ ]:




