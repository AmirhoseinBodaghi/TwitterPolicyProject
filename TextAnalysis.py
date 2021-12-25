import vaex as vx
from nltk.sentiment import SentimentIntensityAnalyzer
from multiprocessing import Process, current_process, Queue
import nltk
import re
import pandas as pd
import numpy as np
from pandas import read_csv
import sys
#--------------------
def Desired_Data_Frame(address_input):
    df = vx.open (address_input)
##    df['txt_letters_numbers']=df['txt'].apply(number_letters_finders)
##    df['txt_sentiment_score']=df['txt'].apply(sentiment_score_finders)
    
    cols = ['created_at', 'id', 'tweetid', 'retweeted_userid', 'is_quote_status', 'replyto_userid', 'quoted_txt', 'txt', 'txt_letters_numbers', 'txt_sentiment_score', 'txt_letters_numbers_without_HashtagMention', 'txt_sentiment_score_without_HashtagMention', 'retweet_count']
    df = df[cols]
    df['retweeted_userid'] = df['retweeted_userid'].fillna(0) #after we found that all the three feature of the above are same, we choose one of them ie. 'retweeted_userid' to be the representative of the being retweet
    df['retweeted_userid'] = df.func.where(df.retweeted_userid != 0, 1, df.retweeted_userid)
    df['is_quote_status'] = df.func.where(df.is_quote_status == False, 0, df.is_quote_status) #automatically True truns to 1
    df['replyto_userid'] = df['replyto_userid'].fillna(0)
    df['replyto_userid'] = df.func.where(df.replyto_userid != 0, 1, df.replyto_userid)

    df['tweetid']=df.tweetid.values.astype('int')
    df['retweeted_userid']=df.retweeted_userid.values.astype('int')
    df['replyto_userid']=df.replyto_userid.values.astype('int')
    df['retweet_count']=df.retweet_count.values.astype('int')

    df['original'] = df['retweeted_userid'] + df['replyto_userid'] + df['is_quote_status']
    df['original'] = df['original'].where(df['original'] == 0, 1) #automatically turns 0 all values of df['original'] != 0

    df['modes'] = 1000*df.retweeted_userid + 100*df.is_quote_status + 10*df.replyto_userid + df.original 


    df['created_at'] = df.created_at.astype('datetime64') #turn the created_at column to timestamp

    return df
#--------------------
def Hybrid_to_Pure_Conversion (df):
    df ['is_quote_status'] = df.func.where(df.retweeted_userid == 1, 0, df.is_quote_status) # turning 1100 mode ---> 1000 mode 
    df ['replyto_userid'] = df.func.where(df.is_quote_status == 1, 0, df.replyto_userid) # turning 110 mode ---> 100 mode
    df ['modes'] = 1000*df.retweeted_userid + 100*df.is_quote_status + 10*df.replyto_userid + df.original # now we don't have hybrid modes in the dataset anymore
    return df
#--------------------
def Make_Panda_DataFrame_For_Each_User (df):
    df_pandas = df.to_pandas_df()
    df_pandas['created_at'] = pd.to_datetime(df_pandas['created_at'])
    df_pandas.set_index('created_at', inplace=True)
    grouper = [g[1] for g in df_pandas.groupby('id')]
    
    return grouper
#--------------------
# This Function gives three outputs:
    # 1- Type of quoter
    # 2- The change in average of quote text number of letters between within and pre intervals
    # 3- The change in average of quote text sentiment between within and pre intervals
def Quote_Analyzer (UserDataframe, threshold): 
    UserDataframePre = UserDataframe['2019-10-09':'2020-10-08']
    UserDataframePre = UserDataframePre.copy()
    UserDataframeWithin = UserDataframe['2020-10-09':'2020-12-15']
    UserDataframeWithin = UserDataframeWithin.copy()
    UserDataframePost = UserDataframe['2020-12-16':]
    UserDataframePost = UserDataframePost.copy()

    UserDataframePre['txt_letters_numbers_quote'] = UserDataframePre['txt_letters_numbers'] * UserDataframePre['is_quote_status']
    UserDataframePre['txt_sentiment_score_quote'] = UserDataframePre['txt_sentiment_score'] * UserDataframePre['is_quote_status']

    UserDataframeWithin['txt_letters_numbers_quote'] = UserDataframeWithin['txt_letters_numbers'] * UserDataframeWithin['is_quote_status']
    UserDataframeWithin['txt_sentiment_score_quote'] = UserDataframeWithin['txt_sentiment_score'] * UserDataframeWithin['is_quote_status']

##    print (UserDataframe.columns)
##    print ("number non zero in pre quote : " , np.count_nonzero(UserDataframePre['is_quote_status']))
    UserDataframePre['txt_letters_numbers_quote_without_HashtagMention'] = UserDataframePre['txt_letters_numbers_without_HashtagMention'] * UserDataframePre['is_quote_status']
##    print ("number of non zero sentiment in pre txt_sentiment_score_without_HashtagMention : ", np.count_nonzero(UserDataframePre['txt_sentiment_score_without_HashtagMention']))
    UserDataframePre['txt_sentiment_score_quote_without_HashtagMention'] = UserDataframePre['txt_sentiment_score_without_HashtagMention'] * UserDataframePre['is_quote_status']
##    print ("number of non zero sentiment in pre quote txt_sentiment_score_quote_without_HashtagMention : ", np.count_nonzero(UserDataframePre['txt_sentiment_score_quote_without_HashtagMention']))
    UserDataframePre['txt_retweet_count_quote'] = UserDataframePre['retweet_count'] * UserDataframePre['is_quote_status']
    

##    print ("number non zero in within quote : " , np.count_nonzero(UserDataframeWithin['is_quote_status']))
    UserDataframeWithin['txt_letters_numbers_quote_without_HashtagMention'] = UserDataframeWithin['txt_letters_numbers_without_HashtagMention'] * UserDataframeWithin['is_quote_status']
##    print ("number of non zero sentiment in within txt_sentiment_score_without_HashtagMention : ", np.count_nonzero(UserDataframeWithin['txt_sentiment_score_without_HashtagMention']))
    UserDataframeWithin['txt_sentiment_score_quote_without_HashtagMention'] = UserDataframeWithin['txt_sentiment_score_without_HashtagMention'] * UserDataframeWithin['is_quote_status']    
##    print ("number of non zero sentiment in within quote txt_sentiment_score_quote_without_HashtagMention : ", np.count_nonzero(UserDataframeWithin['txt_sentiment_score_quote_without_HashtagMention']))
    UserDataframeWithin['txt_retweet_count_quote'] = UserDataframeWithin['retweet_count'] * UserDataframeWithin['is_quote_status']
     
    pd.set_option("display.max_rows", None, "display.max_columns", None)
##    print ("########################################")
##    print (UserDataframePre[(UserDataframePre['is_quote_status']!=0) & (UserDataframePre['txt_sentiment_score_without_HashtagMention']!=0)])
##    print (UserDataframePre[(UserDataframePre['txt_sentiment_score_quote_without_HashtagMention']!=0)])
    #---------------------
    if UserDataframePre.shape[0]!=0:
        NumberTweetPre = UserDataframePre.shape[0]
        NumberQuotePre = UserDataframePre['is_quote_status'].sum()         
        AverageQuotePre = NumberQuotePre/NumberTweetPre
        
        if AverageQuotePre > threshold: #threshold = 0.075 is the average rate of number of quotes to number of all tweets (i.e. each user in dataset in avaerage, has 7.5 quotes in 100 tweets of her own) 
            Quoter_Type_Pre = 1
        else :
            Quoter_Type_Pre = 0

        NumberRetweetQuotePre = UserDataframePre['txt_retweet_count_quote'].sum()

        LettersQuotePre = UserDataframePre['txt_letters_numbers_quote'].sum()
        SentimentQuotePre = UserDataframePre['txt_sentiment_score_quote'].sum()
        LettersQuotePreWithoutHashtagMention = UserDataframePre['txt_letters_numbers_quote_without_HashtagMention'].sum()
        SentimentQuotePreWithoutHashtagMention = UserDataframePre['txt_sentiment_score_quote_without_HashtagMention'].sum()
        
        if NumberQuotePre != 0:
            LettersAverageQuotePre = LettersQuotePre/NumberQuotePre
            SentimentAverageQuotePre = SentimentQuotePre/NumberQuotePre
            LettersAverageQuotePreWithoutHashtagMention = LettersQuotePreWithoutHashtagMention/NumberQuotePre
##            print ("SentimentQuotePreWithoutHashtagMention : ",SentimentQuotePreWithoutHashtagMention) #################
            SentimentAverageQuotePreWithoutHashtagMention = SentimentQuotePreWithoutHashtagMention/NumberQuotePre
            RetweetAverageQuotePre = NumberRetweetQuotePre/NumberQuotePre
        else:
            LettersAverageQuotePre = 0
            SentimentAverageQuotePre = 0
            LettersAverageQuotePreWithoutHashtagMention = 0
            SentimentAverageQuotePreWithoutHashtagMention = 0
            RetweetAverageQuotePre = 0
    else:
        Quoter_Type_Pre = 3
        LettersAverageQuotePre = 0
        SentimentAverageQuotePre = 0
        LettersAverageQuotePreWithoutHashtagMention = 0
        SentimentAverageQuotePreWithoutHashtagMention = 0
        RetweetAverageQuotePre = 0
    #---------------------

        
    #---------------------
    if UserDataframeWithin.shape[0]!=0:
        NumberTweetWithin = UserDataframeWithin.shape[0]
        NumberQuoteWithin = UserDataframeWithin['is_quote_status'].sum()        
        AverageQuoteWithin = NumberQuoteWithin/NumberTweetWithin
        
        if AverageQuoteWithin > threshold: # threshold = 0.075 is the average rate of number of quotes to number of all tweets (i.e. each user in dataset in avaerage, has 7.5 quotes in 100 tweets of her own) 
            Quoter_Type_Within = 1
        else :
            Quoter_Type_Within = 0

        NumberRetweetQuoteWithin = UserDataframeWithin['txt_retweet_count_quote'].sum()
        
        LettersQuoteWithin = UserDataframeWithin['txt_letters_numbers_quote'].sum()
        SentimentQuoteWithin = UserDataframeWithin['txt_sentiment_score_quote'].sum()
        LettersQuoteWithinWithoutHashtagMention = UserDataframeWithin['txt_letters_numbers_quote_without_HashtagMention'].sum()
        SentimentQuoteWithinWithoutHashtagMention = UserDataframeWithin['txt_sentiment_score_quote_without_HashtagMention'].sum()
        
        if NumberQuoteWithin != 0:
            LettersAverageQuoteWithin = LettersQuoteWithin/NumberQuoteWithin
            SentimentAverageQuoteWithin = SentimentQuoteWithin/NumberQuoteWithin
            LettersAverageQuoteWithinWithoutHashtagMention = LettersQuoteWithinWithoutHashtagMention/NumberQuoteWithin
##            print ("SentimentQuoteWithinWithoutHashtagMention : ",SentimentQuoteWithinWithoutHashtagMention) #################
            SentimentAverageQuoteWithinWithoutHashtagMention = SentimentQuoteWithinWithoutHashtagMention/NumberQuoteWithin
            RetweetAverageQuoteWithin = NumberRetweetQuoteWithin/NumberQuoteWithin
        else:
            LettersAverageQuoteWithin = 0
            SentimentAverageQuoteWithin = 0
            LettersAverageQuoteWithinWithoutHashtagMention = 0
            SentimentAverageQuoteWithinWithoutHashtagMention = 0
            RetweetAverageQuoteWithin = 0
    else:
        Quoter_Type_Within = 3
        LettersAverageQuoteWithin = 0
        SentimentAverageQuoteWithin = 0
        LettersAverageQuoteWithinWithoutHashtagMention = 0
        SentimentAverageQuoteWithinWithoutHashtagMention = 0
        RetweetAverageQuoteWithin = 0
    #---------------------
        
    #---------------------
    if (Quoter_Type_Pre == 1 and Quoter_Type_Within == 1):
        Quoter_Type = 1 # Long Term Quoter
        LettersAverageQuoteChange = LettersAverageQuoteWithin - LettersAverageQuotePre
        SentimentAverageQuoteChange = SentimentAverageQuoteWithin - SentimentAverageQuotePre
        LettersAverageQuoteChangeWithoutHashtagMention = LettersAverageQuoteWithinWithoutHashtagMention - LettersAverageQuotePreWithoutHashtagMention
        SentimentAverageQuoteChangeWithoutHashtagMention = SentimentAverageQuoteWithinWithoutHashtagMention - SentimentAverageQuotePreWithoutHashtagMention
        RetweetAverageQuoteChange = RetweetAverageQuoteWithin - RetweetAverageQuotePre
        
    elif (Quoter_Type_Pre == 0 and Quoter_Type_Within == 1):
        Quoter_Type = 0 # Short Term Quoter
        LettersAverageQuoteChange = LettersAverageQuoteWithin - LettersAverageQuotePre
        SentimentAverageQuoteChange = SentimentAverageQuoteWithin - SentimentAverageQuotePre
        LettersAverageQuoteChangeWithoutHashtagMention = LettersAverageQuoteWithinWithoutHashtagMention - LettersAverageQuotePreWithoutHashtagMention
        SentimentAverageQuoteChangeWithoutHashtagMention = SentimentAverageQuoteWithinWithoutHashtagMention - SentimentAverageQuotePreWithoutHashtagMention
        RetweetAverageQuoteChange = RetweetAverageQuoteWithin - RetweetAverageQuotePre
        
    else:
        Quoter_Type = 2 # Not Interested in the study
        LettersAverageQuoteChange = 0
        SentimentAverageQuoteChange = 0
        LettersAverageQuoteChangeWithoutHashtagMention = 0
        SentimentAverageQuoteChangeWithoutHashtagMention = 0
        RetweetAverageQuoteChange = 0
    #---------------------

    UserResults = [Quoter_Type,LettersAverageQuoteChange,SentimentAverageQuoteChange,LettersAverageQuoteChangeWithoutHashtagMention,SentimentAverageQuoteChangeWithoutHashtagMention, RetweetAverageQuoteChange]
##    print (UserResults)  #########################
    return UserResults                    
#--------------------
def Extracting_Quote_Analysis_Data (grouper, threshold, fout):
    UserResultsAll = []
    j = 0 #########################
    for UserDataframe in grouper:
##        print (j) #########################
        UserResults = Quote_Analyzer (UserDataframe, threshold)
##        print ("==========================================") #########################
        UserResultsAll.append (UserResults)
        j += 1 #########################
        

    ShortTermUsersCount = 0
    ShortTermLetterChange = []
    ShortTermSentimentChange = []
    ShortTermLetterChangeWithoutHashtagMention = []
    ShortTermSentimentChangeWithoutHashtagMention = []
    ShoreTermRetweetChange = []
    LongTermUsersCount = 0
    LongTermLetterChange = []
    LongTermSentimentChange = []
    LongTermLetterChangeWithoutHashtagMention = []
    LongTermSentimentChangeWithoutHashtagMention = []
    LongTermRetweetChange = []
    for user in UserResultsAll:
        if user[0]==0: #ShortTerm Quoter
            ShortTermLetterChange.append (user[1])
            ShortTermSentimentChange.append (user[2])
            ShortTermLetterChangeWithoutHashtagMention.append(user[3])
            ShortTermSentimentChangeWithoutHashtagMention.append(user[4])
            ShoreTermRetweetChange.append(user[5])
            ShortTermUsersCount += 1
        elif user[0]==1: #LongTerm Quoter
            LongTermLetterChange.append (user[1])
            LongTermSentimentChange.append (user[2])
            LongTermLetterChangeWithoutHashtagMention.append(user[3])
            LongTermSentimentChangeWithoutHashtagMention.append(user[4])
            LongTermRetweetChange.append(user[5])
            LongTermUsersCount += 1
        else:
            None

    fout.write(" - - -  With Threshold of " + str(threshold) + " - - - ")
    fout.write("\n")
    fout.write("ShortTermUsersCount : " + str(ShortTermUsersCount))
    fout.write("\n")
    fout.write("LongTermUsersCount : " + str(LongTermUsersCount))
    fout.write("\n")
    fout.write("ShortTermLetterChangeMean : " + str(np.mean (ShortTermLetterChange)))
    fout.write("\n")
    fout.write("ShortTermSentimentChangeMean : " + str(np.mean (ShortTermSentimentChange)))
    fout.write("\n")
    fout.write("LongTermLetterChangeMean : " + str(np.mean (LongTermLetterChange)))
    fout.write("\n")
    fout.write("LongTermSentimentChangeMean : " + str(np.mean (LongTermSentimentChange)))
    fout.write("\n")
    fout.write("ShortTermLetterChangeMeanWithoutHashtagMention : " + str(np.mean (ShortTermLetterChangeWithoutHashtagMention)))
    fout.write("\n")
    fout.write("ShortTermSentimentChangeMeanWithoutHashtagMention : " + str(np.mean (ShortTermSentimentChangeWithoutHashtagMention)))
    fout.write("\n")
    fout.write("LongTermLetterChangeMeanWithoutHashtagMention : " + str(np.mean (LongTermLetterChangeWithoutHashtagMention)))
    fout.write("\n")
    fout.write("LongTermSentimentChangeMeanWithoutHashtagMention : " + str(np.mean (LongTermSentimentChangeWithoutHashtagMention)))
    fout.write("\n")
    fout.write("ShoreTermRetweetChangeMean : " + str(np.mean (ShoreTermRetweetChange)))
    fout.write("\n")
    fout.write("LongTermRetweetChangeMean : " + str(np.mean (LongTermRetweetChange)))
    fout.write("\n")
    fout.write("- - - - - - - - - - - - - - - - - - -")
    fout.write("\n")
    fout.write("- - - - - - - - - - - - - - - - - - -")
    fout.write("\n")

    return fout    
#--------------------
def main():
##    address_input_1 = '/mnt/tb/amirdata/Merge_All_Waves.zip.hdf5'
    address_input = '/mnt/tb/amirdata/Merge_All_Waves_WithTextDataAnalysis.hdf5'
    address_output = '/home/abodaghi/Twitter_Project/Data_Processing/Results/Text_Analysis/'

    
    df = Desired_Data_Frame(address_input)
    df = Hybrid_to_Pure_Conversion (df)
    grouper = Make_Panda_DataFrame_For_Each_User (df)

    fout = open (address_output + "FinalResult.txt","w")
    threshold_list = [0.075,0.05,0.01,0]
    for threshold in threshold_list:
        fout = Extracting_Quote_Analysis_Data (grouper, threshold, fout)

    fout.close()
    
#--------------------
if __name__ == '__main__':
        main ()
