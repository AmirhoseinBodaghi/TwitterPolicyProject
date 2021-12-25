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
def extract_days (start_date,end_date):
    sdate = date(int(start_date[:4]),int(start_date[5:7]),int(start_date[8:10]))
    edate = date(int(end_date[:4]),int(end_date[5:7]),int(end_date[8:10]))
##    between_days = [i.strftime("%Y/%m/%d, %H:%M:%S") for i in pandas.date_range(sdate,edate-timedelta(days=1),freq='d')]
    between_days = [i.strftime("%Y-%m-%d") for i in pd.date_range(sdate,edate-timedelta(days=1),freq='d')]
    between_days.append (end_date)
    return between_days
#----------------------
def calculate_total_tweettype_per_day_for_users_pre_interval (grouper):

    #========================= P r e - I n t e r v a l (pre-within) ========================
    print ("pre_1")
    pre_interval_start_date = '2019-10-09'
    pre_interval_end_date = '2020-10-08'
    pre_interval_days = extract_days (pre_interval_start_date,pre_interval_end_date)
    print ("pre_2")

    Original_P_All = []
    Quote_P_All = []
    Reply_P_All = []
    Retweet_P_All = []
    for day in pre_interval_days:
        print ("================")
        print ("Pre_interval_days")
        print (day)
        print ("================")
        Original_P = []
        Quote_P = []
        Reply_P = []
        Retweet_P = []
        index = 0
        while index < len (grouper):
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_pre.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Original_P.append (df_pre.loc[day]['original'])
                Quote_P.append (df_pre.loc[day]['is_quote_status'])
                Reply_P.append (df_pre.loc[day]['replyto_userid'])
                Retweet_P.append (df_pre.loc[day]['retweeted_userid'])
            else:
                Original_P.append (0)
                Quote_P.append (0)
                Reply_P.append (0)
                Retweet_P.append (0)
            index += 1

        Original_P_All.append (Original_P)
        Quote_P_All.append (Quote_P)
        Reply_P_All.append (Reply_P)
        Retweet_P_All.append (Retweet_P)

    Pre_All = [Original_P_All, Quote_P_All, Reply_P_All, Retweet_P_All]

    file_name = "/home/abodaghi/Twitter_Project/Data_Processing/Results/Visualization_New_all/Pre_All.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(Pre_All, open_file)
    open_file.close()

#================================================================================================
def calculate_total_tweettype_per_day_for_users_within_interval (grouper):

        #============================ W i t h i n - I n t e r v a l (pre-within) ========================
    print ("w_1")
    within_interval_start_date = '2020-10-09'
    within_interval_end_date = '2020-12-15'
    within_interval_days = extract_days (within_interval_start_date,within_interval_end_date)

    print ("w_2")
    Original_W_All = []
    Quote_W_All = []
    Reply_W_All = []
    Retweet_W_All = []
    for day in within_interval_days:
        print ("================")
        print ("within_interval_days")
        print (day)
        print ("================")
        Original_W = []
        Quote_W = []
        Reply_W = []
        Retweet_W = []
        index = 0
        while index < len (grouper):
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_within.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Original_W.append (df_within.loc[day]['original'])
                Quote_W.append (df_within.loc[day]['is_quote_status'])
                Reply_W.append (df_within.loc[day]['replyto_userid'])
                Retweet_W.append (df_within.loc[day]['retweeted_userid'])
            else:
                Original_W.append (0)
                Quote_W.append (0)
                Reply_W.append (0)
                Retweet_W.append (0)

            index += 1

        Original_W_All.append(Original_W)       
        Quote_W_All.append(Quote_W)        
        Reply_W_All.append(Reply_W)        
        Retweet_W_All.append(Retweet_W)

    Within_All = [Original_W_All, Quote_W_All, Reply_W_All, Retweet_W_All]

    file_name = "/home/abodaghi/Twitter_Project/Data_Processing/Results/Visualization_New_all/Within_All.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(Within_All, open_file)
    open_file.close()
    
#================================================================================
def calculate_total_tweettype_per_day_for_users_post_interval (grouper):
    #========================= P o s t - I n t e r v a l (within-post) ======================================================
    print ("wp_p_1")
    post_interval_start_date = '2020-12-16'
    post_interval_end_date = '2021-02-02'
    post_interval_days = extract_days (post_interval_start_date,post_interval_end_date)

    print ("wp_p_2")
    Original_P_All = []
    Quote_P_All = []
    Reply_P_All = []
    Retweet_P_All = []
    for day in post_interval_days:
        print ("================")
        print ("post_interval_days")
        print (day)
        print ("================")
        Original_P = []
        Quote_P = []
        Reply_P = []
        Retweet_P = []
        index = 0
        while index < len (grouper):
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
            date_list = [ts.date() for ts in df_post.index.tolist()]
            date_str_list = [str(date) for date in date_list]
            if day in date_str_list:
                Original_P.append (df_post.loc[day]['original'])
                Quote_P.append (df_post.loc[day]['is_quote_status'])
                Reply_P.append (df_post.loc[day]['replyto_userid'])
                Retweet_P.append (df_post.loc[day]['retweeted_userid'])                
            else:
                Original_P.append (0)
                Quote_P.append (0)
                Reply_P.append (0)
                Retweet_P.append (0)

            index += 1

        Original_P_All.append (Original_P)
        Quote_P_All.append (Quote_P)
        Reply_P_All.append (Reply_P)
        Retweet_P_All.append (Retweet_P)

    Post_All = [Original_P_All, Quote_P_All, Reply_P_All, Retweet_P_All]
    
    file_name = "/home/abodaghi/Twitter_Project/Data_Processing/Results/Visualization_New_all/Post_All.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(Post_All, open_file)
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
    
##    print (grouper[0])
##    df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[0])
##    print (df_pre)
    #---------------------------------------------------------------
    #Multi Processing --------
##    x = 'AmirBala' #this is just a non_sense args to be imported with grouper, otherwise the function detects the grouper as 86334 args instead of 1
    process_1 = Process(target = calculate_total_tweettype_per_day_for_users_pre_interval   , args = (grouper,))
    print ("4")
    process_2 = Process(target = calculate_total_tweettype_per_day_for_users_within_interval   , args = (grouper,))
    print ("5")
    process_3 = Process(target = calculate_total_tweettype_per_day_for_users_post_interval   , args = (grouper,))
    print ("6")

    process_1.start()
    print ("7")
    process_2.start()
    print ("8")
    process_3.start()
    print ("9")


    process_1.join()
    print ("10")
    process_2.join()
    print ("11")
    process_3.join()
    print ("12")
    #-------------------------
#----------------------
if __name__ == '__main__':
        main ()
