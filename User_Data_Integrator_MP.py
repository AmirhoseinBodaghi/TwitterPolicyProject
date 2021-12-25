import vaex as vx
import matplotlib
##matplotlib.use('Agg') #this is because generating matplot figures on the server requires X running, however when we bring up idle we automatically do it by calling xmanager, but when we run the code in the backend (by nohup) then we need to bring this command
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib.ticker as mticke
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.formula.api as sm
from pandas.plotting import autocorrelation_plot
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression import linear_model
from patsy import dmatrices
import statsmodels.graphics.tsaplots as tsa
from statsmodels.tsa.arima.model import ARIMA as ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from pmdarima.arima import ADFTest
import warnings
from multiprocessing import Process, current_process
from datetime import datetime
from pandas import read_csv
import csv
#----------------------
#turn to favorite dataframe
def Desired_Data_Frame(address_input_1):
    df = vx.open (address_input_1)
    cols = ['created_at', 'id', 'tweetid', 'retweeted_userid', 'is_quote_status', 'replyto_userid']
    df = df[cols]
    
    df['retweeted_userid'] = df['retweeted_userid'].fillna(0) #after we found that all the three feature of the above are same, we choose one of them ie. 'retweeted_userid' to be the representative of the being retweet
    df['retweeted_userid'] = df.func.where(df.retweeted_userid != 0, 1, df.retweeted_userid)
    df['is_quote_status'] = df.func.where(df.is_quote_status == False, 0, df.is_quote_status) #automatically True truns to 1
    df['replyto_userid'] = df['replyto_userid'].fillna(0)
    df['replyto_userid'] = df.func.where(df.replyto_userid != 0, 1, df.replyto_userid)

    df['tweetid']=df.tweetid.values.astype('int')
    df['retweeted_userid']=df.retweeted_userid.values.astype('int')
    df['replyto_userid']=df.replyto_userid.values.astype('int')
    

    df['original'] = df['retweeted_userid'] + df['replyto_userid'] + df['is_quote_status']
    df['original'] = df['original'].where(df['original'] == 0, 1) #automatically turns 0 all values of df['original'] != 0

    df['modes'] = 1000*df.retweeted_userid + 100*df.is_quote_status + 10*df.replyto_userid + df.original 


    df['created_at'] = df.created_at.astype('datetime64') #turn the created_at column to timestamp

    return df
#----------------------
def Hybrid_to_Pure_Conversion (df):
    df ['is_quote_status'] = df.func.where(df.retweeted_userid == 1, 0, df.is_quote_status) # turning 1100 mode ---> 1000 mode 
    df ['replyto_userid'] = df.func.where(df.is_quote_status == 1, 0, df.replyto_userid) # turning 110 mode ---> 100 mode
    df ['modes'] = 1000*df.retweeted_userid + 100*df.is_quote_status + 10*df.replyto_userid + df.original # now we don't have hybrid modes in the dataset anymore
    return df
#----------------------
def Make_Panda_DataFrame_For_Each_User (df):
    df_pandas = df.to_pandas_df()
    df_pandas['created_at'] = pd.to_datetime(df_pandas['created_at'])
    df_pandas.set_index('created_at', inplace=True)
    grouper = [g[1] for g in df_pandas.groupby('id')]
    return grouper
#----------------------    
def Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (user_dataframe):
    df = user_dataframe.groupby([user_dataframe.index.date])[["retweeted_userid","is_quote_status","replyto_userid","original"]].sum() #setting daily unit for timeseries
    index = pd.DatetimeIndex(df.index)
    df.set_index(index, inplace=True) # turning the index to timeseries index
    df = df.asfreq(freq = 'D', fill_value=0) # setting the frequency as daily and filling the gaps with days with 0 value for all the columns

    df_pre = df['2019-10-09':'2020-10-08'] # we only consider 1 year before the issuing the policy, and ignore the previous tweets (if ever user had had)
    df_pre = df_pre.copy()
    df_pre['row_number'] = np.arange(df_pre.shape[0]) #this column is useful for regrerssion
    
    df_within = df['2020-10-09':'2020-12-15']
    df_within = df_within.copy()
    df_within['row_number'] = np.arange(df_within.shape[0]) #this column is useful for regrerssion
    
    df_post = df['2020-12-16':]
    df_post = df_post.copy()
    df_post['row_number'] = np.arange(df_post.shape[0]) #this column is useful for regrerssion

    return df_pre, df_within, df_post   

#----------------------
def count_number_tweets_and_datapoints_in_intervals (user_dataframe):

    df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (user_dataframe)

    #pre interval ---------
    if df_pre.shape[0] > 0:
        retweet_pre_number_nonzero_datapoints = (df_pre["retweeted_userid"] != 0).sum()
        retweet_pre_count = df_pre["retweeted_userid"].sum()
        reply_pre_number_nonzero_datapoints = (df_pre["replyto_userid"] != 0).sum()
        reply_pre_count = df_pre["replyto_userid"].sum()
        quote_pre_number_nonzero_datapoints = (df_pre["is_quote_status"] != 0).sum()
        quote_pre_count = df_pre["is_quote_status"].sum()
        original_pre_number_nonzero_datapoints = (df_pre["original"] != 0).sum()
        original_pre_count = df_pre["original"].sum()
    else:
        retweet_pre_number_nonzero_datapoints = 0
        retweet_pre_count = 0
        reply_pre_number_nonzero_datapoints = 0
        reply_pre_count = 0
        quote_pre_number_nonzero_datapoints = 0
        quote_pre_count = 0
        original_pre_number_nonzero_datapoints = 0
        original_pre_count = 0
    #---------------------

    #within interval -----    
    if df_within.shape[0] > 0:
        retweet_within_number_nonzero_datapoints = (df_within["retweeted_userid"] != 0).sum()
        retweet_within_count = df_within["retweeted_userid"].sum()
        reply_within_number_nonzero_datapoints = (df_within["replyto_userid"] != 0).sum()
        reply_within_count = df_within["replyto_userid"].sum()
        quote_within_number_nonzero_datapoints = (df_within["is_quote_status"] != 0).sum()
        quote_within_count = df_within["is_quote_status"].sum()
        original_within_number_nonzero_datapoints = (df_within["original"] != 0).sum()
        original_within_count = df_within["original"].sum()
    else:
        retweet_within_number_nonzero_datapoints = 0
        retweet_within_count = 0
        reply_within_number_nonzero_datapoints = 0
        reply_within_count = 0
        quote_within_number_nonzero_datapoints = 0
        quote_within_count = 0
        original_within_number_nonzero_datapoints = 0
        original_within_count = 0
    #----------------------

    #post interval --------
    if df_post.shape[0] > 0:
        retweet_post_number_nonzero_datapoints = (df_post["retweeted_userid"] != 0).sum()
        retweet_post_count = df_post["retweeted_userid"].sum()
        reply_post_number_nonzero_datapoints = (df_post["replyto_userid"] != 0).sum()
        reply_post_count = df_post["replyto_userid"].sum()
        quote_post_number_nonzero_datapoints = (df_post["is_quote_status"] != 0).sum()
        quote_post_count = df_post["is_quote_status"].sum()
        original_post_number_nonzero_datapoints = (df_post["original"] != 0).sum()
        original_post_count = df_post["original"].sum()
    else:
        retweet_post_number_nonzero_datapoints = 0
        retweet_post_count = 0
        reply_post_number_nonzero_datapoints = 0
        reply_post_count = 0
        quote_post_number_nonzero_datapoints = 0
        quote_post_count = 0
        original_post_number_nonzero_datapoints = 0
        original_post_count = 0
    #----------------------    

    return retweet_pre_count,retweet_pre_number_nonzero_datapoints,retweet_within_count,retweet_within_number_nonzero_datapoints,retweet_post_count,retweet_post_number_nonzero_datapoints,reply_pre_count,reply_pre_number_nonzero_datapoints,reply_within_count,reply_within_number_nonzero_datapoints,reply_post_count,reply_post_number_nonzero_datapoints,quote_pre_count,quote_pre_number_nonzero_datapoints,quote_within_count,quote_within_number_nonzero_datapoints,quote_post_count,quote_post_number_nonzero_datapoints,original_pre_count,original_pre_number_nonzero_datapoints,original_within_count,original_within_number_nonzero_datapoints,original_post_count,original_post_number_nonzero_datapoints    
#----------------------
def form_the_user_data_chunk_chunk (grouper, df_all_data, df_retweet, df_reply, df_quote, df_original, i):
    n_user = 5000*(i)
    name_output_file = 'User_Data' + str(i) + '.csv'
    address_output = "//home//abodaghi//Twitter_Project//Data_Processing//Results//User_Data_Analysis//UD//"
    df_users = read_csv("//home//abodaghi//Twitter_Project//Data_Processing//Results//User_Data_Analysis//User_Data_File//userInf.csv", encoding="ISO-8859-1", low_memory=False)
    df_users['id'] = df_users['id'].astype('int')
    grouper_users = [g[1] for g in df_users.groupby('id')]
    
    if n_user != 80000: # Early chunks contains 5k users each
        while n_user < 5000*(i+1):
            df_all_data.at[n_user,'id'] = int (grouper[n_user]['id'].iloc[0])
            # Adding id ----------------------------------------------

            # Adding metadata like:  followers_count , friends_count , statuses_count , year
            for user in grouper_users:
                if user['id'].iloc[0] == int(grouper[n_user]['id'].iloc[0]):
                    df_all_data.at[n_user,'followers_count'] = int (user['followers_count'].iloc[0])
                    df_all_data.at[n_user,'friends_count'] = int (user['friends_count'].iloc[0])
                    df_all_data.at[n_user,'statuses_count'] = int (user['statuses_count'].iloc[0])
                    df_all_data.at[n_user,'year'] = int (user['year'].iloc[0])
                    break
            # -------------------------------------------------------------------------------

            # Count number of datapoints and tweets 
            retweet_pre_count,retweet_pre_number_nonzero_datapoints,retweet_within_count,retweet_within_number_nonzero_datapoints,retweet_post_count,retweet_post_number_nonzero_datapoints,reply_pre_count,reply_pre_number_nonzero_datapoints,reply_within_count,reply_within_number_nonzero_datapoints,reply_post_count,reply_post_number_nonzero_datapoints,quote_pre_count,quote_pre_number_nonzero_datapoints,quote_within_count,quote_within_number_nonzero_datapoints,quote_post_count,quote_post_number_nonzero_datapoints,original_pre_count,original_pre_number_nonzero_datapoints,original_within_count,original_within_number_nonzero_datapoints,original_post_count,original_post_number_nonzero_datapoints = count_number_tweets_and_datapoints_in_intervals (grouper[n_user]) 
            # -------------------------------------------------------------------------------
            
            # Adding the following information in each interval in each tweet type: 'number tweets', 'number datapoints', 'slope', 'slopechange', 'significance status'
            # Retweet --------
            df_all_data.at[n_user,'slopepre_retweet'] = df_retweet['A3'].iloc[n_user]
            df_all_data.at[n_user,'slopewithin_retweet'] = df_retweet['B3'].iloc[n_user]
            df_all_data.at[n_user,'slopepost_retweet'] = df_retweet['C3'].iloc[n_user]
                
            df_all_data.at[n_user,'retweet_pre_count'] = retweet_pre_count
            df_all_data.at[n_user,'retweet_pre_number_nonzero_datapoints'] = retweet_pre_number_nonzero_datapoints
            df_all_data.at[n_user,'retweet_within_count'] = retweet_within_count
            df_all_data.at[n_user,'retweet_within_number_nonzero_datapoints'] = retweet_within_number_nonzero_datapoints
            df_all_data.at[n_user,'retweet_post_count'] = retweet_post_count
            df_all_data.at[n_user,'retweet_post_number_nonzero_datapoints'] = retweet_post_number_nonzero_datapoints

            df_all_data.at[n_user,'Significance_Status_Pre_Retweet'] = df_retweet['A4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Within_Retweet'] = df_retweet['B4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Post_Retweet'] = df_retweet['C4'].iloc[n_user] 
            df_all_data.at[n_user,'SlopeChangePreWithinRetweet'] = df_retweet['B3'].iloc[n_user] - df_retweet['A3'].iloc[n_user]
            df_all_data.at[n_user,'SlopeChangeWithinPostRetweet'] = df_retweet['C3'].iloc[n_user] - df_retweet['B3'].iloc[n_user]
            # ----------------

            # Reply --------
            df_all_data.at[n_user,'slopepre_reply'] = df_reply['A3'].iloc[n_user]
            df_all_data.at[n_user,'slopewithin_reply'] = df_reply['B3'].iloc[n_user]
            df_all_data.at[n_user,'slopepost_reply'] = df_reply['C3'].iloc[n_user]
                
            df_all_data.at[n_user,'reply_pre_count'] = reply_pre_count
            df_all_data.at[n_user,'reply_pre_number_nonzero_datapoints'] = reply_pre_number_nonzero_datapoints
            df_all_data.at[n_user,'reply_within_count'] = reply_within_count
            df_all_data.at[n_user,'reply_within_number_nonzero_datapoints'] = reply_within_number_nonzero_datapoints
            df_all_data.at[n_user,'reply_post_count'] = reply_post_count
            df_all_data.at[n_user,'reply_post_number_nonzero_datapoints'] = reply_post_number_nonzero_datapoints

            df_all_data.at[n_user,'Significance_Status_Pre_Reply'] = df_reply['A4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Within_Reply'] = df_reply['B4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Post_Reply'] = df_reply['C4'].iloc[n_user] 
            df_all_data.at[n_user,'SlopeChangePreWithinReply'] = df_reply['B3'].iloc[n_user] - df_reply['A3'].iloc[n_user]
            df_all_data.at[n_user,'SlopeChangeWithinPostReply'] = df_reply['C3'].iloc[n_user] - df_reply['B3'].iloc[n_user]
            # ----------------

            # Quote --------
            df_all_data.at[n_user,'slopepre_quote'] = df_quote['A3'].iloc[n_user]
            df_all_data.at[n_user,'slopewithin_quote'] = df_quote['B3'].iloc[n_user]
            df_all_data.at[n_user,'slopepost_quote'] = df_quote['C3'].iloc[n_user]
                
            df_all_data.at[n_user,'quote_pre_count'] = quote_pre_count
            df_all_data.at[n_user,'quote_pre_number_nonzero_datapoints'] = quote_pre_number_nonzero_datapoints
            df_all_data.at[n_user,'quote_within_count'] = quote_within_count
            df_all_data.at[n_user,'quote_within_number_nonzero_datapoints'] = quote_within_number_nonzero_datapoints
            df_all_data.at[n_user,'quote_post_count'] = quote_post_count
            df_all_data.at[n_user,'quote_post_number_nonzero_datapoints'] = quote_post_number_nonzero_datapoints

            df_all_data.at[n_user,'Significance_Status_Pre_Quote'] = df_quote['A4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Within_Quote'] = df_quote['B4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Post_Quote'] = df_quote['C4'].iloc[n_user] 
            df_all_data.at[n_user,'SlopeChangePreWithinQuote'] = df_quote['B3'].iloc[n_user] - df_quote['A3'].iloc[n_user]
            df_all_data.at[n_user,'SlopeChangeWithinPostQuote'] = df_quote['C3'].iloc[n_user] - df_quote['B3'].iloc[n_user]
            # ----------------

            # Original --------
            df_all_data.at[n_user,'slopepre_original'] = df_original['A3'].iloc[n_user]
            df_all_data.at[n_user,'slopewithin_original'] = df_original['B3'].iloc[n_user]
            df_all_data.at[n_user,'slopepost_original'] = df_original['C3'].iloc[n_user]
                
            df_all_data.at[n_user,'original_pre_count'] = original_pre_count
            df_all_data.at[n_user,'original_pre_number_nonzero_datapoints'] = original_pre_number_nonzero_datapoints
            df_all_data.at[n_user,'original_within_count'] = original_within_count
            df_all_data.at[n_user,'original_within_number_nonzero_datapoints'] = original_within_number_nonzero_datapoints
            df_all_data.at[n_user,'original_post_count'] = original_post_count
            df_all_data.at[n_user,'original_post_number_nonzero_datapoints'] = original_post_number_nonzero_datapoints

            df_all_data.at[n_user,'Significance_Status_Pre_Original'] = df_original['A4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Within_Original'] = df_original['B4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Post_Original'] = df_original['C4'].iloc[n_user] 
            df_all_data.at[n_user,'SlopeChangePreWithinOriginal'] = df_original['B3'].iloc[n_user] - df_original['A3'].iloc[n_user]
            df_all_data.at[n_user,'SlopeChangeWithinPostOriginal'] = df_original['C3'].iloc[n_user] - df_original['B3'].iloc[n_user]
            # ----------------
            # ------------------------------------------------

    ##        print (n_user)
            n_user += 1
        df_all_data['id'] = df_all_data['id'].astype('int') # big numbers (bigger than 15 digits) if do not become 'int' before restoring in .csv would turn into a different number for example: 900780202304909312 would saved as 900780202304909184                
        df_all_data.to_csv(address_output + name_output_file, index=False)
    else :   # the last chunk goes from user number 80k to the last user in the dataset (+86k)   
        while n_user < len(grouper):
            df_all_data.at[n_user,'id'] = int (grouper[n_user]['id'].iloc[0])
            # Adding id ----------------------------------------------

            # Adding metadata like:  followers_count , friends_count , statuses_count , year
            for user in grouper_users:
                if user['id'].iloc[0] == int (grouper[n_user]['id'].iloc[0]):
                    df_all_data.at[n_user,'followers_count'] = int (user['followers_count'].iloc[0])
                    df_all_data.at[n_user,'friends_count'] = int (user['friends_count'].iloc[0])
                    df_all_data.at[n_user,'statuses_count'] = int (user['statuses_count'].iloc[0])
                    df_all_data.at[n_user,'year'] = int (user['year'].iloc[0])
                    break
            # -------------------------------------------------------------------------------

            # Count number of datapoints and tweets 
            retweet_pre_count,retweet_pre_number_nonzero_datapoints,retweet_within_count,retweet_within_number_nonzero_datapoints,retweet_post_count,retweet_post_number_nonzero_datapoints,reply_pre_count,reply_pre_number_nonzero_datapoints,reply_within_count,reply_within_number_nonzero_datapoints,reply_post_count,reply_post_number_nonzero_datapoints,quote_pre_count,quote_pre_number_nonzero_datapoints,quote_within_count,quote_within_number_nonzero_datapoints,quote_post_count,quote_post_number_nonzero_datapoints,original_pre_count,original_pre_number_nonzero_datapoints,original_within_count,original_within_number_nonzero_datapoints,original_post_count,original_post_number_nonzero_datapoints = count_number_tweets_and_datapoints_in_intervals (grouper[n_user]) 
            # -------------------------------------------------------------------------------
            
            # Adding the following information in each interval in each tweet type: 'number tweets', 'number datapoints', 'slope', 'slopechange', 'significance status'
            # Retweet --------
            df_all_data.at[n_user,'slopepre_retweet'] = df_retweet['A3'].iloc[n_user]
            df_all_data.at[n_user,'slopewithin_retweet'] = df_retweet['B3'].iloc[n_user]
            df_all_data.at[n_user,'slopepost_retweet'] = df_retweet['C3'].iloc[n_user]
                
            df_all_data.at[n_user,'retweet_pre_count'] = retweet_pre_count
            df_all_data.at[n_user,'retweet_pre_number_nonzero_datapoints'] = retweet_pre_number_nonzero_datapoints
            df_all_data.at[n_user,'retweet_within_count'] = retweet_within_count
            df_all_data.at[n_user,'retweet_within_number_nonzero_datapoints'] = retweet_within_number_nonzero_datapoints
            df_all_data.at[n_user,'retweet_post_count'] = retweet_post_count
            df_all_data.at[n_user,'retweet_post_number_nonzero_datapoints'] = retweet_post_number_nonzero_datapoints

            df_all_data.at[n_user,'Significance_Status_Pre_Retweet'] = df_retweet['A4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Within_Retweet'] = df_retweet['B4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Post_Retweet'] = df_retweet['C4'].iloc[n_user] 
            df_all_data.at[n_user,'SlopeChangePreWithinRetweet'] = df_retweet['B3'].iloc[n_user] - df_retweet['A3'].iloc[n_user]
            df_all_data.at[n_user,'SlopeChangeWithinPostRetweet'] = df_retweet['C3'].iloc[n_user] - df_retweet['B3'].iloc[n_user]
            # ----------------

            # Reply --------
            df_all_data.at[n_user,'slopepre_reply'] = df_reply['A3'].iloc[n_user]
            df_all_data.at[n_user,'slopewithin_reply'] = df_reply['B3'].iloc[n_user]
            df_all_data.at[n_user,'slopepost_reply'] = df_reply['C3'].iloc[n_user]
                
            df_all_data.at[n_user,'reply_pre_count'] = reply_pre_count
            df_all_data.at[n_user,'reply_pre_number_nonzero_datapoints'] = reply_pre_number_nonzero_datapoints
            df_all_data.at[n_user,'reply_within_count'] = reply_within_count
            df_all_data.at[n_user,'reply_within_number_nonzero_datapoints'] = reply_within_number_nonzero_datapoints
            df_all_data.at[n_user,'reply_post_count'] = reply_post_count
            df_all_data.at[n_user,'reply_post_number_nonzero_datapoints'] = reply_post_number_nonzero_datapoints

            df_all_data.at[n_user,'Significance_Status_Pre_Reply'] = df_reply['A4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Within_Reply'] = df_reply['B4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Post_Reply'] = df_reply['C4'].iloc[n_user] 
            df_all_data.at[n_user,'SlopeChangePreWithinReply'] = df_reply['B3'].iloc[n_user] - df_reply['A3'].iloc[n_user]
            df_all_data.at[n_user,'SlopeChangeWithinPostReply'] = df_reply['C3'].iloc[n_user] - df_reply['B3'].iloc[n_user]
            # ----------------

            # Quote --------
            df_all_data.at[n_user,'slopepre_quote'] = df_quote['A3'].iloc[n_user]
            df_all_data.at[n_user,'slopewithin_quote'] = df_quote['B3'].iloc[n_user]
            df_all_data.at[n_user,'slopepost_quote'] = df_quote['C3'].iloc[n_user]
                
            df_all_data.at[n_user,'quote_pre_count'] = quote_pre_count
            df_all_data.at[n_user,'quote_pre_number_nonzero_datapoints'] = quote_pre_number_nonzero_datapoints
            df_all_data.at[n_user,'quote_within_count'] = quote_within_count
            df_all_data.at[n_user,'quote_within_number_nonzero_datapoints'] = quote_within_number_nonzero_datapoints
            df_all_data.at[n_user,'quote_post_count'] = quote_post_count
            df_all_data.at[n_user,'quote_post_number_nonzero_datapoints'] = quote_post_number_nonzero_datapoints

            df_all_data.at[n_user,'Significance_Status_Pre_Quote'] = df_quote['A4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Within_Quote'] = df_quote['B4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Post_Quote'] = df_quote['C4'].iloc[n_user] 
            df_all_data.at[n_user,'SlopeChangePreWithinQuote'] = df_quote['B3'].iloc[n_user] - df_quote['A3'].iloc[n_user]
            df_all_data.at[n_user,'SlopeChangeWithinPostQuote'] = df_quote['C3'].iloc[n_user] - df_quote['B3'].iloc[n_user]
            # ----------------

            # Original --------
            df_all_data.at[n_user,'slopepre_original'] = df_original['A3'].iloc[n_user]
            df_all_data.at[n_user,'slopewithin_original'] = df_original['B3'].iloc[n_user]
            df_all_data.at[n_user,'slopepost_original'] = df_original['C3'].iloc[n_user]
                
            df_all_data.at[n_user,'original_pre_count'] = original_pre_count
            df_all_data.at[n_user,'original_pre_number_nonzero_datapoints'] = original_pre_number_nonzero_datapoints
            df_all_data.at[n_user,'original_within_count'] = original_within_count
            df_all_data.at[n_user,'original_within_number_nonzero_datapoints'] = original_within_number_nonzero_datapoints
            df_all_data.at[n_user,'original_post_count'] = original_post_count
            df_all_data.at[n_user,'original_post_number_nonzero_datapoints'] = original_post_number_nonzero_datapoints

            df_all_data.at[n_user,'Significance_Status_Pre_Original'] = df_original['A4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Within_Original'] = df_original['B4'].iloc[n_user] 
            df_all_data.at[n_user,'Significance_Status_Post_Original'] = df_original['C4'].iloc[n_user] 
            df_all_data.at[n_user,'SlopeChangePreWithinOriginal'] = df_original['B3'].iloc[n_user] - df_original['A3'].iloc[n_user]
            df_all_data.at[n_user,'SlopeChangeWithinPostOriginal'] = df_original['C3'].iloc[n_user] - df_original['B3'].iloc[n_user]
            # ----------------
            # ------------------------------------------------

    ##        print (n_user)
            n_user += 1

        df_all_data['id'] = df_all_data['id'].astype('int') # big numbers (bigger than 15 digits) if do not become 'int' before restoring in .csv would turn into a different number for example: 900780202304909312 would saved as 900780202304909184    
        df_all_data.to_csv(address_output + name_output_file, index=False)

#----------------------
def MultiProcessing (grouper, df_all_data, df_retweet, df_reply, df_quote, df_original): #see if we should save each sub dataframe seperately
    i = 0
    Process_List = []
    while i < 17:
        Process_List.append (Process(target = form_the_user_data_chunk_chunk   , args = (grouper, df_all_data, df_retweet, df_reply, df_quote, df_original, i)))
        i += 1

##    print (len (Process_List))
    i = 0
    while i < 17:
        Process_List[i].start()
        i += 1

    i = 0
    while i < 17:
        Process_List[i].join()
        i += 1    

#----------------------
def main():
    
    warnings.filterwarnings("ignore") # to supress warnings
    address_input_1 = '/mnt/tb/amirdata/Merge_All_Waves.zip.hdf5'
    df = Desired_Data_Frame(address_input_1)
    df = Hybrid_to_Pure_Conversion (df)
    grouper = Make_Panda_DataFrame_For_Each_User (df)
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    address_input_2 = '/home/abodaghi/Twitter_Project/Data_Processing/Results/Slope_Level_Merger_Analyzer/'
    df_retweet = read_csv(address_input_2 + 'Retweet_All_Users.csv', encoding="ISO-8859-1", engine='python')
    df_reply = read_csv(address_input_2 + 'Reply_All_Users.csv', encoding="ISO-8859-1", engine='python')
    df_quote = read_csv(address_input_2 + 'Quote_All_Users.csv', encoding="ISO-8859-1", engine='python')
    df_original = read_csv(address_input_2 + 'Original_All_Users.csv', encoding="ISO-8859-1", engine='python')

    Columns_List = ['id','followers_count','friends_count','statuses_count','year','slopepre_retweet','slopewithin_retweet','slopepost_retweet','retweet_pre_count','retweet_pre_number_nonzero_datapoints','retweet_within_count','retweet_within_number_nonzero_datapoints','retweet_post_count','retweet_post_number_nonzero_datapoints','Significance_Status_Pre_Retweet','Significance_Status_Within_Retweet','Significance_Status_Post_Retweet','SlopeChangePreWithinRetweet','SlopeChangeWithinPostRetweet','slopepre_reply','slopewithin_reply','slopepost_reply','reply_pre_count','reply_pre_number_nonzero_datapoints','reply_within_count','reply_within_number_nonzero_datapoints','reply_post_count','reply_post_number_nonzero_datapoints','Significance_Status_Pre_Reply','Significance_Status_Within_Reply','Significance_Status_Post_Reply','SlopeChangePreWithinReply','SlopeChangeWithinPostReply','slopepre_quote','slopewithin_quote','slopepost_quote','quote_pre_count','quote_pre_number_nonzero_datapoints','quote_within_count','quote_within_number_nonzero_datapoints','quote_post_count','quote_post_number_nonzero_datapoints','Significance_Status_Pre_Quote','Significance_Status_Within_Quote','Significance_Status_Post_Quote','SlopeChangePreWithinQuote','SlopeChangeWithinPostQuote','slopepre_original','slopewithin_original','slopepost_original','original_pre_count','original_pre_number_nonzero_datapoints','original_within_count','original_within_number_nonzero_datapoints','original_post_count','original_post_number_nonzero_datapoints','Significance_Status_Pre_Original','Significance_Status_Within_Original','Significance_Status_Post_Original','SlopeChangePreWithinOriginal','SlopeChangeWithinPostOriginal']
    df_all_data = pd.DataFrame(0, index=np.arange(len(grouper)), columns=Columns_List) #making the zero dataframe so to be filled with the slope, level, .. of all users in the future
    df_all_data = df_all_data.astype('float')
##    print (df_all_data.dtypes)
    
##    df_all_data = form_the_user_data (grouper, df_all_data, df_retweet, df_reply, df_quote, df_original)
    MultiProcessing (grouper, df_all_data, df_retweet, df_reply, df_quote, df_original)
##    df_all_data.to_csv("//home//abodaghi//Twitter_Project//Data_Processing//Results//User_Data_Analysis//User_Data.csv", index = False, quoting = csv.QUOTE_NONNUMERIC)
   
#----------------------
if __name__ == '__main__':
        main ()

