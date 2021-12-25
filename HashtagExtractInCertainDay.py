import vaex as vx
from datetime import datetime
import warnings
from multiprocessing import Process, current_process, Queue
from collections import Counter
import xlsxwriter
import pandas as pd
import numpy as np
import pickle
import os
#----------------------
#turn to favorite dataframe
def Desired_Data_Frame(address_input):
    df = vx.open (address_input)
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
def hashtag_finder (df_day):
    
    hashtag_list_day = []

    i = 0
    while i < df_day.shape[0]:
        if df_day['hashtags'].iloc[i]:
            hashtags = df_day['hashtags'].iloc[i].split("\t") #because tweets with more than one hashtags have their hashtags seperated with a tab which is marked with a '\t' : for a example a tweet with three hashtags #Amir and #Roya and #Reza has it as 'Amir\tRoya\tReza'
            for hashtag in hashtags:
                hashtag_list_day.append (hashtag)
        i+=1

    
    return hashtag_list_day
#----------------------
def Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (user_dataframe):
    df = user_dataframe.groupby([user_dataframe.index.date]).agg({'hashtags': lambda x: '\t'.join(x[x.notna()]), 'retweeted_userid':sum, 'is_quote_status':sum, 'replyto_userid':sum, 'original':sum,})
    index = pd.DatetimeIndex(df.index)
    df.set_index(index, inplace=True) # turning the index to timeseries index
    df = df.asfreq(freq = 'D', fill_value=0) # setting the frequency as daily and filling the gaps with days with 0 value for all the columns

    df_day = df['2020-11-07':'2020-11-07'] # we only consider from september first
    df_day = df_day.copy()
    df_day['row_number'] = np.arange(df_day.shape[0]) #this column is useful for regrerssion
    
    return df_day
#----------------------
def Chunk_Chunk_Analysis (grouper, i):
    j = 5000*(i)
    hashtag_AllUsers_day = []

    if j != 80000: # Early chunks contains 5k users each
        while j < 5000*(i+1):
            df_day = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[j])
            hashtag_list_day = hashtag_finder (df_day)
            hashtag_AllUsers_day += hashtag_list_day 

            j += 1        

    else :   # the last chunk goes from user number 80k to the last user in the dataset (+86k)   
        while j < len(grouper):
            df_day = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[j])
            hashtag_list_day = hashtag_finder (df_day)
            hashtag_AllUsers_day += hashtag_list_day 

            j += 1            

    hashtag_user_all = [hashtag_AllUsers_day]
    print ("This chunk is finished : " + str(i))

    file_name = "/home/abodaghi/Twitter_Project/Data_Processing/Results/HashtagExtractInCertainDay/" + str(i) + ".pkl"
    open_file = open(file_name, "wb")
    pickle.dump(hashtag_user_all, open_file)
    open_file.close()
    hashtag_user_all = []
#----------------------
def MultiProcessing (grouper): #see if we should save each sub dataframe seperately
    ResultsAll = []
    Process_List = []
    i = 0
    while i < 17: # 16*5000 + 1*6336 = 86336 (number of users in dataset)
        Process_List.append (Process(target = Chunk_Chunk_Analysis , args = (grouper, i)))
        i += 1

##    print (len (Process_List))
    i = 0
    while i < len (Process_List):
        Process_List[i].start()
        i += 1

    i = 0
    while i < len (Process_List):
        Process_List[i].join()
        i += 1

#--------------------
def main():        
    warnings.filterwarnings("ignore") # to supress warnings
    address_input = '/mnt/tb/amirdata/Merge_All_Waves.zip.hdf5'
    df = Desired_Data_Frame(address_input)
    df = Hybrid_to_Pure_Conversion (df)
    grouper = Make_Panda_DataFrame_For_Each_User (df)
    MultiProcessing (grouper)  
#----------------------
if __name__ == '__main__':
        main ()
