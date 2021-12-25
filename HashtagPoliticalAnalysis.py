import xlsxwriter
import pickle
import os
import warnings
import xlrd
import vaex as vx
import pandas as pd
import numpy as np
from datetime import date, timedelta
from collections import Counter
from multiprocessing import Process, current_process
#----------------------
def Political_Hashtags (address_input_1,address_output):
    # Reading input data
    workbook_input = xlrd.open_workbook(address_input_1 + 'MostPopularHashtagsFinal.xlsx')
    worksheet1_input = workbook_input.sheet_by_name('Sheet1')

    row_count = worksheet1_input.nrows
##    print ("row_count : ", row_count)
    tweets_all = []
    row_input = 1
    Political_Hashtag_List = []
    while row_input < row_count:
        Annotation = worksheet1_input.cell(row_input, 2).value
        if Annotation == 1:
            Political_Hashtag_List.append (worksheet1_input.cell(row_input, 1).value)
        
##        print ("row_input : ", row_input)
##        print ("Annotation : ", Annotation)
##        print ("=========================")
        row_input += 1
    return Political_Hashtag_List    
#----------------------
#turn to favorite dataframe
def Desired_Data_Frame(address_input_2):
    df = vx.open (address_input_2)
    cols = ['created_at', 'id', 'tweetid', 'hashtags', 'retweeted_userid', 'is_quote_status', 'replyto_userid']
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
    df = user_dataframe.groupby([user_dataframe.index.date]).agg({'hashtags': lambda x: '\t'.join(x[x.notna()]), 'retweeted_userid':sum, 'is_quote_status':sum, 'replyto_userid':sum, 'original':sum,})
    index = pd.DatetimeIndex(df.index)
    df.set_index(index, inplace=True) # turning the index to timeseries index
    df = df.asfreq(freq = 'D', fill_value=0) # setting the frequency as daily and filling the gaps with days with 0 value for all the columns

    df_pre = df['2020-09-01':'2020-10-08'] # we only consider from september first
    df_pre = df_pre.copy()
    df_pre['row_number'] = np.arange(df_pre.shape[0]) #this column is useful for regrerssion
    
    df_within = df['2020-10-09':'2020-12-15']
    df_within = df_within.copy()
    df_within['row_number'] = np.arange(df_within.shape[0]) #this column is useful for regrerssion
    
    df_post = df['2020-12-16':'2021-02-01'] # we deleted the last day of the dataset in post interval (2021-02-02) due the the end of crawling at 18pm of this day
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
def calculate_total_politicaltweets_per_day_for_users_pre_interval (grouper, Political_Hashtag_List, i, j):
    pre_interval_start_date = '2020-09-01'
    pre_interval_end_date = '2020-10-08'
    pre_interval_days = extract_days (pre_interval_start_date,pre_interval_end_date)

    all_users_pre = []    
    for day in pre_interval_days:
        index = j

        all_pre_user_day_number_political_tweet = []
        if index != 80000:            
            while index < 5000*(i+1):
##                print ("index : ", index)
                pre_user_day_number_political_tweet = 0
                df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
                date_list = [ts.date() for ts in df_pre.index.tolist()]
                date_str_list = [str(date) for date in date_list]
                if day in date_str_list:
                    if df_pre.loc[day]['hashtags']:
                        hashtags = df_pre.loc[day]['hashtags'].split("\t") #because tweets with more than one hashtags have their hashtags seperated with a tab which is marked with a '\t' : for a example a tweet with three hashtags #Amir and #Roya and #Reza has it as 'Amir\tRoya\tReza'
##                        print (hashtags)
                        for hashtag in hashtags:
                            if hashtag in Political_Hashtag_List:
                                pre_user_day_number_political_tweet += 1

##                print ("pre_user_day_number_political_tweet : ", pre_user_day_number_political_tweet)
##                print ("================")
                all_pre_user_day_number_political_tweet.append (pre_user_day_number_political_tweet)
                index += 1
                
##            print ("all_pre_user_day_number_political_tweet : ", all_pre_user_day_number_political_tweet)
            all_users_pre.append (all_pre_user_day_number_political_tweet)
        else:
            while index < len(grouper):
##                print ("index : ", index)
                pre_user_day_number_political_tweet = 0
                df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
                date_list = [ts.date() for ts in df_pre.index.tolist()]
                date_str_list = [str(date) for date in date_list]
                if day in date_str_list:
                    if df_pre.loc[day]['hashtags']:
                        hashtags = df_pre.loc[day]['hashtags'].split("\t") #because tweets with more than one hashtags have their hashtags seperated with a tab which is marked with a '\t' : for a example a tweet with three hashtags #Amir and #Roya and #Reza has it as 'Amir\tRoya\tReza'
##                        print (hashtags)
                        for hashtag in hashtags:
                            if hashtag in Political_Hashtag_List:
                                pre_user_day_number_political_tweet += 1

##                print ("pre_user_day_number_political_tweet : ", pre_user_day_number_political_tweet)
##                print ("================")
                all_pre_user_day_number_political_tweet.append (pre_user_day_number_political_tweet)
                index += 1
                
##            print ("all_pre_user_day_number_political_tweet : ", all_pre_user_day_number_political_tweet)
            all_users_pre.append (all_pre_user_day_number_political_tweet)
        
##    print ("all_users_pre : ", all_users_pre)
    return all_users_pre
#----------------------
def calculate_total_politicaltweets_per_day_for_users_within_interval (grouper, Political_Hashtag_List, i, j):
    within_interval_start_date = '2020-10-09'
    within_interval_end_date = '2020-12-15'
    within_interval_days = extract_days (within_interval_start_date,within_interval_end_date)

    all_users_within = []    
    for day in within_interval_days:
        index = j

        all_within_user_day_number_political_tweet = []
        if index != 80000:            
            while index < 5000*(i+1):
##                print ("index : ", index)
                within_user_day_number_political_tweet = 0
                df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
                date_list = [ts.date() for ts in df_within.index.tolist()]
                date_str_list = [str(date) for date in date_list]
                if day in date_str_list:
                    if df_within.loc[day]['hashtags']:
                        hashtags = df_within.loc[day]['hashtags'].split("\t") #because tweets with more than one hashtags have their hashtags seperated with a tab which is marked with a '\t' : for a example a tweet with three hashtags #Amir and #Roya and #Reza has it as 'Amir\tRoya\tReza'
##                        print (hashtags)
                        for hashtag in hashtags:
                            if hashtag in Political_Hashtag_List:
                                within_user_day_number_political_tweet += 1

##                print ("within_user_day_number_political_tweet : ", within_user_day_number_political_tweet)
##                print ("================")
                all_within_user_day_number_political_tweet.append (within_user_day_number_political_tweet)
                index += 1
                
##            print ("all_within_user_day_number_political_tweet : ", all_within_user_day_number_political_tweet)
            all_users_within.append (all_within_user_day_number_political_tweet)
        else:
            while index < len(grouper):
##                print ("index : ", index)
                within_user_day_number_political_tweet = 0
                df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
                date_list = [ts.date() for ts in df_within.index.tolist()]
                date_str_list = [str(date) for date in date_list]
                if day in date_str_list:
                    if df_within.loc[day]['hashtags']:
                        hashtags = df_within.loc[day]['hashtags'].split("\t") #because tweets with more than one hashtags have their hashtags seperated with a tab which is marked with a '\t' : for a example a tweet with three hashtags #Amir and #Roya and #Reza has it as 'Amir\tRoya\tReza'
##                        print (hashtags)
                        for hashtag in hashtags:
                            if hashtag in Political_Hashtag_List:
                                within_user_day_number_political_tweet += 1

##                print ("within_user_day_number_political_tweet : ", within_user_day_number_political_tweet)
##                print ("================")
                all_within_user_day_number_political_tweet.append (within_user_day_number_political_tweet)
                index += 1
                
##            print ("all_within_user_day_number_political_tweet : ", all_within_user_day_number_political_tweet)
            all_users_within.append (all_within_user_day_number_political_tweet)
        
##    print ("all_users_within : ", all_users_within)
    return all_users_within
#----------------------
def calculate_total_politicaltweets_per_day_for_users_post_interval (grouper, Political_Hashtag_List, i, j):
    post_interval_start_date = '2020-12-16'
    post_interval_end_date = '2021-02-01'
    post_interval_days = extract_days (post_interval_start_date,post_interval_end_date)

    all_users_post = []    
    for day in post_interval_days:
        index = j

        all_post_user_day_number_political_tweet = []
        if index != 80000:            
            while index < 5000*(i+1):
##                print ("index : ", index)
                post_user_day_number_political_tweet = 0
                df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
                date_list = [ts.date() for ts in df_post.index.tolist()]
                date_str_list = [str(date) for date in date_list]
                if day in date_str_list:
                    if df_post.loc[day]['hashtags']:
                        hashtags = df_post.loc[day]['hashtags'].split("\t") #because tweets with more than one hashtags have their hashtags seperated with a tab which is marked with a '\t' : for a example a tweet with three hashtags #Amir and #Roya and #Reza has it as 'Amir\tRoya\tReza'
##                        print (hashtags)
                        for hashtag in hashtags:
                            if hashtag in Political_Hashtag_List:
                                post_user_day_number_political_tweet += 1

##                print ("post_user_day_number_political_tweet : ", post_user_day_number_political_tweet)
##                print ("================")
                all_post_user_day_number_political_tweet.append (post_user_day_number_political_tweet)
                index += 1
                
##            print ("all_post_user_day_number_political_tweet : ", all_post_user_day_number_political_tweet)
            all_users_post.append (all_post_user_day_number_political_tweet)
        else:
            while index < len(grouper):
##                print ("index : ", index)
                post_user_day_number_political_tweet = 0
                df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[index])
                date_list = [ts.date() for ts in df_post.index.tolist()]
                date_str_list = [str(date) for date in date_list]
                if day in date_str_list:
                    if df_post.loc[day]['hashtags']:
                        hashtags = df_post.loc[day]['hashtags'].split("\t") #because tweets with more than one hashtags have their hashtags seperated with a tab which is marked with a '\t' : for a example a tweet with three hashtags #Amir and #Roya and #Reza has it as 'Amir\tRoya\tReza'
##                        print (hashtags)
                        for hashtag in hashtags:
                            if hashtag in Political_Hashtag_List:
                                post_user_day_number_political_tweet += 1

##                print ("post_user_day_number_political_tweet : ", post_user_day_number_political_tweet)
##                print ("================")
                all_post_user_day_number_political_tweet.append (post_user_day_number_political_tweet)
                index += 1
                
##            print ("all_post_user_day_number_political_tweet : ", all_post_user_day_number_political_tweet)
            all_users_post.append (all_post_user_day_number_political_tweet)
        
##    print ("all_users_post : ", all_users_post)
    return all_users_post
#----------------------
def calculate_total_politicaltweets_per_day_for_users_all_intervals(grouper, Political_Hashtag_List, address_output, i,  j):
    
    all_users_pre = calculate_total_politicaltweets_per_day_for_users_pre_interval (grouper, Political_Hashtag_List, i, j)
    all_users_within = calculate_total_politicaltweets_per_day_for_users_within_interval (grouper, Political_Hashtag_List, i, j)
    all_users_post = calculate_total_politicaltweets_per_day_for_users_post_interval (grouper, Political_Hashtag_List, i, j)

    all_users = [all_users_pre] + [all_users_within] + [all_users_post]

    file_name = address_output + str(i) + ".pkl"
    open_file = open(file_name, "wb")
    pickle.dump(all_users, open_file)
    open_file.close()
#----------------------
def Chunk_Chunk_Analysis (grouper, Political_Hashtag_List, i, address_output):
    j = 5000*(i)
    calculate_total_politicaltweets_per_day_for_users_all_intervals(grouper, Political_Hashtag_List, address_output, i,  j)
    print ("This chunk is finished : " + str(i))
#----------------------    
def MultiProcessing (grouper, Political_Hashtag_List, address_output): #see if we should save each sub dataframe seperately
    i = 0
    Process_List = []
    while i < 17:
        Process_List.append (Process(target = Chunk_Chunk_Analysis   , args = (grouper, Political_Hashtag_List, i, address_output)))
        i += 1

    print ("len (Process_List) : ", len (Process_List))
    i = 0
    while i < 17:
        Process_List[i].start()
        i += 1

    i = 0
    while i < 17:
        Process_List[i].join()
        i += 1    
###----------------------
def main():        
    warnings.filterwarnings("ignore") # to supress warnings    
    address_input_1 = '/home/abodaghi/Twitter_Project/Data_Processing/Results/HashtagMostPopular/'
    address_input_2 = '/mnt/tb/amirdata/Merge_All_Waves.zip.hdf5'
    address_output = '/home/abodaghi/Twitter_Project/Data_Processing/Results/HashtagPoliticalAnalysis/'

    Political_Hashtag_List = Political_Hashtags (address_input_1,address_output)
##    print (Political_Hashtag_List)
##    print (len (Political_Hashtag_List))

    df = Desired_Data_Frame(address_input_2)
    df = Hybrid_to_Pure_Conversion (df)
    grouper = Make_Panda_DataFrame_For_Each_User (df)

    MultiProcessing (grouper, Political_Hashtag_List, address_output)
#----------------------
if __name__ == '__main__':
        main ()
