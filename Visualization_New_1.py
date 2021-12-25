import vaex as vx
import pandas as pd
import numpy as np
import warnings
from multiprocessing import Process, current_process
from pandas import read_csv
from datetime import date, timedelta
import csv
import pickle
#----------------------
#turn to favorite dataframe
def Desired_Data_Frame(address_input):
    df = vx.open (address_input)
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
def find_significant_users_index ():

    df_original = read_csv('/home/abodaghi/Twitter_Project/Data_Processing/Results/Slope_Level_Merger_Analyzer/Original_All_Users.csv')
    df_Pre_Within_Comparison_o = df_original[(df_original['A4'].isin([2,3,4])) & (df_original['B4'].isin([2,3,4]))]
    df_Within_Post_Comparison_o = df_original[(df_original['B4'].isin([2,3,4])) & (df_original['C4'].isin([2,3,4]))]
    original_users_significant_pre_within = df_Pre_Within_Comparison_o.index.tolist()
    original_users_significant_within_post = df_Within_Post_Comparison_o.index.tolist()

    df_quote = read_csv('/home/abodaghi/Twitter_Project/Data_Processing/Results/Slope_Level_Merger_Analyzer/Quote_All_Users.csv')
    df_Pre_Within_Comparison_q = df_quote[(df_quote['A4'].isin([2,3,4])) & (df_quote['B4'].isin([2,3,4]))]
    df_Within_Post_Comparison_q = df_quote[(df_quote['B4'].isin([2,3,4])) & (df_quote['C4'].isin([2,3,4]))]
    quote_users_significant_pre_within = df_Pre_Within_Comparison_q.index.tolist()
    quote_users_significant_within_post = df_Within_Post_Comparison_q.index.tolist()

    df_reply = read_csv('/home/abodaghi/Twitter_Project/Data_Processing/Results/Slope_Level_Merger_Analyzer/Reply_All_Users.csv')
    df_Pre_Within_Comparison_r = df_reply[(df_reply['A4'].isin([2,3,4])) & (df_reply['B4'].isin([2,3,4]))]
    df_Within_Post_Comparison_r = df_reply[(df_reply['B4'].isin([2,3,4])) & (df_reply['C4'].isin([2,3,4]))]
    reply_users_significant_pre_within = df_Pre_Within_Comparison_r.index.tolist()
    reply_users_significant_within_post = df_Within_Post_Comparison_r.index.tolist()

    df_retweet = read_csv('/home/abodaghi/Twitter_Project/Data_Processing/Results/Slope_Level_Merger_Analyzer/Retweet_All_Users.csv')
    df_Pre_Within_Comparison_re = df_retweet[(df_retweet['A4'].isin([2,3,4])) & (df_retweet['B4'].isin([2,3,4]))]
    df_Within_Post_Comparison_re = df_retweet[(df_retweet['B4'].isin([2,3,4])) & (df_retweet['C4'].isin([2,3,4]))]
    retweet_users_significant_pre_within = df_Pre_Within_Comparison_re.index.tolist()
    retweet_users_significant_within_post = df_Within_Post_Comparison_re.index.tolist()

##    print ("len(original_users_significant_pre_within) : ", len(original_users_significant_pre_within))
##    print ("len(original_users_significant_within_post) : ", len(original_users_significant_within_post))
##
##    print ("len(quote_users_significant_pre_within) : ", len(quote_users_significant_pre_within))
##    print ("len(quote_users_significant_within_post) : ", len(quote_users_significant_within_post))
##
##    print ("len(reply_users_significant_pre_within) : ", len(reply_users_significant_pre_within))
##    print ("len(reply_users_significant_within_post) : ", len(reply_users_significant_within_post))
##
##    print ("len(retweet_users_significant_pre_within) : ", len(retweet_users_significant_pre_within))
##    print ("len(retweet_users_significant_within_post) : ", len(retweet_users_significant_within_post))
    
        
    return original_users_significant_pre_within, original_users_significant_within_post, quote_users_significant_pre_within, quote_users_significant_within_post, reply_users_significant_pre_within, reply_users_significant_within_post, retweet_users_significant_pre_within, retweet_users_significant_within_post
#----------------------
def extract_days (start_date,end_date):
    sdate = date(int(start_date[:4]),int(start_date[5:7]),int(start_date[8:10]))
    edate = date(int(end_date[:4]),int(end_date[5:7]),int(end_date[8:10]))
##    between_days = [i.strftime("%Y/%m/%d, %H:%M:%S") for i in pandas.date_range(sdate,edate-timedelta(days=1),freq='d')]
    between_days = [i.strftime("%Y-%m-%d") for i in pd.date_range(sdate,edate-timedelta(days=1),freq='d')]
    between_days.append (end_date)
    return between_days
#----------------------
def calculate_total_tweettype_per_day_for_significant_users_pre_prewithin_interval (grouper, original_users_significant_pre_within, quote_users_significant_pre_within, reply_users_significant_pre_within,  retweet_users_significant_pre_within):

    #========================= P r e - I n t e r v a l (pre-within) ========================
    print ("pw_p_1")
    pre_interval_start_date = '2019-10-09'
    pre_interval_end_date = '2020-10-08'
    pre_interval_days = extract_days (pre_interval_start_date,pre_interval_end_date)
    print ("pw_p_2")

    Original_PW_P_All = []
    Quote_PW_P_All = []
    Reply_PW_P_All = []
    Retweet_PW_P_All = []
    for day in pre_interval_days:
        print ("================")
        print ("Pre_interval_days")
        print (day)
        print ("================")
        Original_PW_P = []
        Quote_PW_P = []
        Reply_PW_P = []
        Retweet_PW_P = []
        for index in original_users_significant_pre_within:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_pre.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Original_PW_P.append (df_pre.loc[day]['original'])
            else:
                Original_PW_P.append (0)

        for index in quote_users_significant_pre_within:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_pre.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Quote_PW_P.append (df_pre.loc[day]['is_quote_status'])
            else:
                Quote_PW_P.append (0)

        for index in reply_users_significant_pre_within:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_pre.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                    Reply_PW_P.append (df_pre.loc[day]['replyto_userid'])
            else:
                    Reply_PW_P.append (0)
            
            
        for index in retweet_users_significant_pre_within:
                df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
                date_list = [ts.date() for ts in df_pre.index.tolist()]
                date_str_list = [str(date) for date in date_list]
                if day in date_str_list:
                        Retweet_PW_P.append (df_pre.loc[day]['retweeted_userid'])
                else:
                        Retweet_PW_P.append (0)

        Original_PW_P_All.append (Original_PW_P)
        Quote_PW_P_All.append (Quote_PW_P)
        Reply_PW_P_All.append (Reply_PW_P)
        Retweet_PW_P_All.append (Retweet_PW_P)

    PW_P_All = [Original_PW_P_All, Quote_PW_P_All, Reply_PW_P_All, Retweet_PW_P_All]
##    print ("len (PW_P_All) : ", len (PW_P_All))
##    print ("=================================")
##    print ("len (Original_PW_P_All) : ", len (Original_PW_P_All))
##    print ("len (Quote_PW_P_All) : ", len (Quote_PW_P_All))
##    print ("len (Reply_PW_P_All) : ", len (Reply_PW_P_All))
##    print ("len (Retweet_PW_P_All) : ", len (Retweet_PW_P_All))
##    print ("=================================")
##    print ("len (Original_PW_P_All[0]) : ", len (Original_PW_P_All[0]))
##    print ("len (Quote_PW_P_All[0]) : ", len (Quote_PW_P_All[0]))
##    print ("len (Reply_PW_P_All[0]) : ", len (Reply_PW_P_All[0]))
##    print ("len (Retweet_PW_P_All[0]) : ", len (Retweet_PW_P_All[0]))
##    print ("=================================")
##    print ("PW_P_All : ", PW_P_All)

    file_name = "/home/abodaghi/Twitter_Project/Data_Processing/Results/Visualization_New_1/PW_P_All.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(PW_P_All, open_file)
    open_file.close()

#================================================================================================
def calculate_total_tweettype_per_day_for_significant_users_within_prewithin_interval (grouper, original_users_significant_pre_within, quote_users_significant_pre_within, reply_users_significant_pre_within,  retweet_users_significant_pre_within):

        #============================ W i t h i n - I n t e r v a l (pre-within) ========================
    print ("pw_w_1")
    within_interval_start_date = '2020-10-09'
    within_interval_end_date = '2020-12-15'
    within_interval_days = extract_days (within_interval_start_date,within_interval_end_date)

    print ("pw_w_2")
    Original_PW_W_All = []
    Quote_PW_W_All = []
    Reply_PW_W_All = []
    Retweet_PW_W_All = []
    for day in within_interval_days:
        print ("================")
        print ("within_interval_days_pw")
        print (day)
        print ("================")
        Original_PW_W = []
        Quote_PW_W = []
        Reply_PW_W = []
        Retweet_PW_W = []
        for index in original_users_significant_pre_within:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_within.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Original_PW_W.append (df_within.loc[day]['original'])
            else:
                Original_PW_W.append (0)

        for index in quote_users_significant_pre_within:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_within.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Quote_PW_W.append (df_within.loc[day]['is_quote_status'])
            else:
                Quote_PW_W.append (0)

        for index in reply_users_significant_pre_within:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_within.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Reply_PW_W.append (df_within.loc[day]['replyto_userid'])
            else:
                Reply_PW_W.append (0)
                
                
        for index in retweet_users_significant_pre_within:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_within.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Retweet_PW_W.append (df_within.loc[day]['retweeted_userid'])
            else:
                Retweet_PW_W.append (0)

        Original_PW_W_All.append(Original_PW_W)       
        Quote_PW_W_All.append(Quote_PW_W)        
        Reply_PW_W_All.append(Reply_PW_W)        
        Retweet_PW_W_All.append(Retweet_PW_W)

    PW_W_All = [Original_PW_W_All, Quote_PW_W_All, Reply_PW_W_All, Retweet_PW_W_All]
##    print ("len (PW_W_All) : ", len (PW_W_All))
##    print ("=================================")
##    print ("len (Original_PW_W_All) : ", len (Original_PW_W_All))
##    print ("len (Quote_PW_W_All) : ", len (Quote_PW_W_All))
##    print ("len (Reply_PW_W_All) : ", len (Reply_PW_W_All))
##    print ("len (Retweet_PW_W_All) : ", len (Retweet_PW_W_All))
##    print ("=================================")
##    print ("len (Original_PW_W_All[0]) : ", len (Original_PW_W_All[0]))
##    print ("len (Quote_PW_W_All[0]) : ", len (Quote_PW_W_All[0]))
##    print ("len (Reply_PW_W_All[0]) : ", len (Reply_PW_W_All[0]))
##    print ("len (Retweet_PW_W_All[0]) : ", len (Retweet_PW_W_All[0]))
##    print ("=================================")
##    print ("len (Original_PW_W_All[1]) : ", len (Original_PW_W_All[1]))
##    print ("len (Quote_PW_W_All[1]) : ", len (Quote_PW_W_All[1]))
##    print ("len (Reply_PW_W_All[1]) : ", len (Reply_PW_W_All[1]))
##    print ("len (Retweet_PW_W_All[1]) : ", len (Retweet_PW_W_All[1]))
##    print ("PW_W_All : ", PW_W_All)

    file_name = "/home/abodaghi/Twitter_Project/Data_Processing/Results/Visualization_New_1/PW_W_All.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(PW_W_All, open_file)
    open_file.close()
#================================================================================
    
def calculate_total_tweettype_per_day_for_significant_users_within_withinpost_interval (grouper, original_users_significant_within_post, quote_users_significant_within_post, reply_users_significant_within_post, retweet_users_significant_within_post):

        #============================ W i t h i n - I n t e r v a l (within-post) ====================================================
    print ("wp_w_1")
    within_interval_start_date = '2020-10-09'
    within_interval_end_date = '2020-12-15'
    within_interval_days = extract_days (within_interval_start_date,within_interval_end_date)

    print ("wp_w_2")
    Original_WP_W_All = []
    Quote_WP_W_All = []
    Reply_WP_W_All = []
    Retweet_WP_W_All = []
    for day in within_interval_days:
        print ("================")
        print ("within_interval_days_wp")
        print (day)
        print ("================")
        Original_WP_W = []
        Quote_WP_W = []
        Reply_WP_W = []
        Retweet_WP_W = []
        for index in original_users_significant_within_post:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_within.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Original_WP_W.append (df_within.loc[day]['original'])
            else:
                Original_WP_W.append (0)

        for index in quote_users_significant_within_post:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_within.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Quote_WP_W.append (df_within.loc[day]['is_quote_status'])
            else:
                Quote_WP_W.append (0)

        for index in reply_users_significant_within_post:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_within.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Reply_WP_W.append (df_within.loc[day]['replyto_userid'])
            else:
                Reply_WP_W.append (0)
                
                
        for index in retweet_users_significant_within_post:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_within.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Retweet_WP_W.append (df_within.loc[day]['retweeted_userid'])
            else:
                Retweet_WP_W.append (0)

        Original_WP_W_All.append (Original_WP_W)
        Quote_WP_W_All.append (Quote_WP_W)
        Reply_WP_W_All.append (Reply_WP_W)
        Retweet_WP_W_All.append (Retweet_WP_W)

    WP_W_All = [Original_WP_W_All, Quote_WP_W_All, Reply_WP_W_All, Retweet_WP_W_All]
##    print ("len (WP_W_All) : ", len (WP_W_All))
##    print ("=================================")
##    print ("len (Original_WP_W_All) : ", len (Original_WP_W_All))
##    print ("len (Quote_WP_W_All) : ", len (Quote_WP_W_All))
##    print ("len (Reply_WP_W_All) : ", len (Reply_WP_W_All))
##    print ("len (Retweet_WP_W_All) : ", len (Retweet_WP_W_All))
##    print ("=================================")
##    print ("len (Original_WP_W_All[0]) : ", len (Original_WP_W_All[0]))
##    print ("len (Quote_WP_W_All[0]) : ", len (Quote_WP_W_All[0]))
##    print ("len (Reply_WP_W_All[0]) : ", len (Reply_WP_W_All[0]))
##    print ("len (Retweet_WP_W_All[0]) : ", len (Retweet_WP_W_All[0]))
##    print ("=================================")
##    print ("len (Original_WP_W_All[1]) : ", len (Original_WP_W_All[1]))
##    print ("len (Quote_WP_W_All[1]) : ", len (Quote_WP_W_All[1]))
##    print ("len (Reply_WP_W_All[1]) : ", len (Reply_WP_W_All[1]))
##    print ("len (Retweet_WP_W_All[1]) : ", len (Retweet_WP_W_All[1]))
##    print ("WP_W_All : ", WP_W_All)

    file_name = "/home/abodaghi/Twitter_Project/Data_Processing/Results/Visualization_New_1/WP_W_All.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(WP_W_All, open_file)
    open_file.close()
    
#================================================================================
def calculate_total_tweettype_per_day_for_significant_users_post_withinpost_interval (grouper, original_users_significant_within_post, quote_users_significant_within_post, reply_users_significant_within_post, retweet_users_significant_within_post):
    #========================= P o s t - I n t e r v a l (within-post) ======================================================
    print ("wp_p_1")
    post_interval_start_date = '2020-12-16'
    post_interval_end_date = '2021-02-02'
    post_interval_days = extract_days (post_interval_start_date,post_interval_end_date)

    print ("wp_p_2")
    Original_WP_P_All = []
    Quote_WP_P_All = []
    Reply_WP_P_All = []
    Retweet_WP_P_All = []
    for day in post_interval_days:
        print ("================")
        print ("post_interval_days")
        print (day)
        print ("================")
        Original_WP_P = []
        Quote_WP_P = []
        Reply_WP_P = []
        Retweet_WP_P = []
        for index in original_users_significant_within_post:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_post.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Original_WP_P.append (df_post.loc[day]['original'])
            else:
                Original_WP_P.append (0)

        for index in quote_users_significant_within_post:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_post.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Quote_WP_P.append (df_post.loc[day]['is_quote_status'])
            else:
                Quote_WP_P.append (0)

        for index in reply_users_significant_within_post:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_post.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Reply_WP_P.append (df_post.loc[day]['replyto_userid'])
            else:
                Reply_WP_P.append (0)
                
                
        for index in retweet_users_significant_within_post:
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_post.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Retweet_WP_P.append (df_post.loc[day]['retweeted_userid'])
            else:
                Retweet_WP_P.append (0)
                
        Original_WP_P_All.append (Original_WP_P)
        Quote_WP_P_All.append (Quote_WP_P)
        Reply_WP_P_All.append (Reply_WP_P)
        Retweet_WP_P_All.append (Retweet_WP_P)

    WP_P_All = [Original_WP_P_All, Quote_WP_P_All, Reply_WP_P_All, Retweet_WP_P_All]
    
##    print ("len (WP_P_All) : ", len (WP_P_All))
##    print ("=================================")
##    print ("len (Original_WP_P_All) : ", len (Original_WP_P_All))
##    print ("len (Quote_WP_P_All) : ", len (Quote_WP_P_All))
##    print ("len (Reply_WP_P_All) : ", len (Reply_WP_P_All))
##    print ("len (Retweet_WP_P_All) : ", len (Retweet_WP_P_All))
##    print ("=================================")
##    print ("len (Original_WP_P_All[0]) : ", len (Original_WP_P_All[0]))
##    print ("len (Quote_WP_P_All[0]) : ", len (Quote_WP_P_All[0]))
##    print ("len (Reply_WP_P_All[0]) : ", len (Reply_WP_P_All[0]))
##    print ("len (Retweet_WP_P_All[0]) : ", len (Retweet_WP_P_All[0]))

    
    file_name = "/home/abodaghi/Twitter_Project/Data_Processing/Results/Visualization_New_1/WP_P_All.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(WP_P_All, open_file)
    open_file.close()

#--------------------------------------------------------------------------
def main():
    
    #Reading and Grouping Data into each user dataframe-------------
    warnings.filterwarnings("ignore") # to supress warnings
    address_input = '/mnt/tb/amirdata/Merge_All_Waves.zip.hdf5'
    df = Desired_Data_Frame(address_input)
    print ("1")
    df = Hybrid_to_Pure_Conversion (df)
    print ("2")
    grouper = Make_Panda_DataFrame_For_Each_User (df)
    print ("3")
    #---------------------------------------------------------------

    #Calculating total tweets in any type for significant users in each day of each interval (pre, within, and post). Because users who are significant in pre-within are not nessasarily those who are significant in within-post, then we have two within interval calculation (one for pre-within and the other for within-post)
    original_users_significant_pre_within, original_users_significant_within_post, quote_users_significant_pre_within, quote_users_significant_within_post, reply_users_significant_pre_within, reply_users_significant_within_post, retweet_users_significant_pre_within, retweet_users_significant_within_post = find_significant_users_index ()
    print ("4")
    
##    calculate_total_tweettype_per_day_for_significant_users_pre_prewithin_interval (grouper, original_users_significant_pre_within, quote_users_significant_pre_within, reply_users_significant_pre_within,  retweet_users_significant_pre_within)
##
##    calculate_total_tweettype_per_day_for_significant_users_within_prewithin_interval (grouper, original_users_significant_pre_within, quote_users_significant_pre_within, reply_users_significant_pre_within,  retweet_users_significant_pre_within) 
##
##    calculate_total_tweettype_per_day_for_significant_users_within_withinpost_interval (grouper, original_users_significant_within_post, quote_users_significant_within_post, reply_users_significant_within_post, retweet_users_significant_within_post)
##    calculate_total_tweettype_per_day_for_significant_users_post_withinpost_interval (grouper, original_users_significant_within_post, quote_users_significant_within_post, reply_users_significant_within_post, retweet_users_significant_within_post)
    #--------------------------------------------------------

    #Multi Processing --------
    process_1 = Process(target = calculate_total_tweettype_per_day_for_significant_users_pre_prewithin_interval   , args = (grouper, original_users_significant_pre_within, quote_users_significant_pre_within, reply_users_significant_pre_within,  retweet_users_significant_pre_within))
    print ("5")
    process_2 = Process(target = calculate_total_tweettype_per_day_for_significant_users_within_prewithin_interval   , args = (grouper, original_users_significant_pre_within, quote_users_significant_pre_within, reply_users_significant_pre_within,  retweet_users_significant_pre_within))
    print ("6")
    process_3 = Process(target = calculate_total_tweettype_per_day_for_significant_users_within_withinpost_interval   , args = (grouper, original_users_significant_within_post, quote_users_significant_within_post, reply_users_significant_within_post, retweet_users_significant_within_post))
    print ("7")
    process_4 = Process(target = calculate_total_tweettype_per_day_for_significant_users_post_withinpost_interval   , args = (grouper, original_users_significant_within_post, quote_users_significant_within_post, reply_users_significant_within_post, retweet_users_significant_within_post))
    print ("8")

    process_1.start()
    print ("9")
    process_2.start()
    print ("10")
    process_3.start()
    print ("11")
    process_4.start()
    print ("12")

    process_1.join()
    print ("13")
    process_2.join()
    print ("14")
    process_3.join()
    print ("15")
    process_4.join()
    print ("16")
    #-------------------------
#----------------------
if __name__ == '__main__':
        main ()
