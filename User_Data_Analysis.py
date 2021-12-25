from pandas import read_csv
import vaex as vx
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
#----------------------
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
def Checking_MisMatch (address_integrated, address_input_1): #Check if there is any mismatch between the integrated user data (df_integrated) and the early dataset of users (grouper) in terms of user ID  (the sorting of ID from first user to the last user Must be same)    
    # To achieve df_integrated -----
    df_integrated = read_csv(address_integrated, encoding="ISO-8859-1", low_memory=False)
    #-------------------------------
    # To achieve grouper -----------
    df = Desired_Data_Frame(address_input_1)
    df = Hybrid_to_Pure_Conversion (df)
    grouper = Make_Panda_DataFrame_For_Each_User (df)
    #-------------------------------
    There_Is_Mismatch =  False
    i = 0
    while i < 86334: # 86334 = number of users in dataset 
        if (df_integrated['id'].iloc[i]) != (grouper[i]['id'].iloc[0]):
            There_Is_Mismatch = True
        i += 1

    return There_Is_Mismatch, df_integrated       
#----------------------
def Finding_Number_Users_With_More_Than_7_Datapoints (df_integrated, fout):
    ### note that number of users who have less than 7 datapoints in the intervals in each type is higher than those who have less number of tweets in the same,
    ### because for example a user can have only one non-zero datapoints and within that day published more than 7 tweets.
    fout.write("==================================================================================================" + "\n")
    fout.write("---  Number of Users With More Than 7 Datapoints in Each Tweet Type in Each Interval  ---" + "\n")
    TweetType_Interval_list = ['retweet_pre_number_nonzero_datapoints', 'retweet_within_number_nonzero_datapoints', 'retweet_post_number_nonzero_datapoints', 'reply_pre_number_nonzero_datapoints', 'reply_within_number_nonzero_datapoints', 'reply_post_number_nonzero_datapoints', 'quote_pre_number_nonzero_datapoints', 'quote_within_number_nonzero_datapoints', 'quote_post_number_nonzero_datapoints', 'original_pre_number_nonzero_datapoints', 'original_within_number_nonzero_datapoints', 'original_post_number_nonzero_datapoints']
    for TweetType_Interval in TweetType_Interval_list:
        dfti = df_integrated[df_integrated[TweetType_Interval] > 7]
        fout.write ("Number of users with more than 7 datapoints in " + TweetType_Interval + " : " + str (dfti.shape[0]) + "\n")	
    return fout
#----------------------
def Number_Significant_Users(df_integrated, fout):
    fout.write ("=========================================================================================================================" + "\n")
    fout.write ("---  Number of Users With More Than 7 Datapoints and Significant Model in Each Tweet Type in Each Interval  ---" + "\n")
    # --- Retweet ----
    df_pre_within_retweet_acceptable = df_integrated[(df_integrated['Significance_Status_Pre_Retweet'].isin ([2,3,4])) & (df_integrated['Significance_Status_Within_Retweet'].isin ([2,3,4]))]
    fout.write ("Number of users with more than 7 datapoints and significant model for pre and within intervals in retweet : " + str (df_pre_within_retweet_acceptable.shape[0]) + "\n")
    df_within_post_retweet_acceptable = df_integrated[(df_integrated['Significance_Status_Within_Retweet'].isin ([2,3,4])) & (df_integrated['Significance_Status_Post_Retweet'].isin ([2,3,4]))]
    fout.write ("Number of users with more than 7 datapoints and significant model for within and post intervals in retweet : " + str (df_within_post_retweet_acceptable.shape[0]) + "\n")

    # --- Reply ----
    df_pre_within_reply_acceptable = df_integrated[(df_integrated['Significance_Status_Pre_Reply'].isin ([2,3,4])) & (df_integrated['Significance_Status_Within_Reply'].isin ([2,3,4]))]
    fout.write ("Number of users with more than 7 datapoints and significant model for pre and within intervals in reply : " + str (df_pre_within_reply_acceptable.shape[0]) + "\n")
    df_within_post_reply_acceptable = df_integrated[(df_integrated['Significance_Status_Within_Reply'].isin ([2,3,4])) & (df_integrated['Significance_Status_Post_Reply'].isin ([2,3,4]))]
    fout.write ("Number of users with more than 7 datapoints and significant model for within and post intervals in reply : " + str (df_within_post_reply_acceptable.shape[0]) + "\n")

    # --- Quote ----
    df_pre_within_quote_acceptable = df_integrated[(df_integrated['Significance_Status_Pre_Quote'].isin ([2,3,4])) & (df_integrated['Significance_Status_Within_Quote'].isin ([2,3,4]))]
    fout.write ("Number of users with more than 7 datapoints and significant model for pre and within intervals in quote : " + str (df_pre_within_quote_acceptable.shape[0]) + "\n")
    df_within_post_quote_acceptable = df_integrated[(df_integrated['Significance_Status_Within_Quote'].isin ([2,3,4])) & (df_integrated['Significance_Status_Post_Quote'].isin ([2,3,4]))]
    fout.write ("Number of users with more than 7 datapoints and significant model for within and post intervals in quote : " + str (df_within_post_quote_acceptable.shape[0]) + "\n")

    # --- Original ----
    df_pre_within_original_acceptable = df_integrated[(df_integrated['Significance_Status_Pre_Original'].isin ([2,3,4])) & (df_integrated['Significance_Status_Within_Original'].isin ([2,3,4]))]
    fout.write ("Number of users with more than 7 datapoints and significant model for pre and within intervals in original : " + str (df_pre_within_original_acceptable.shape[0]) + "\n")
    df_within_post_original_acceptable = df_integrated[(df_integrated['Significance_Status_Within_Original'].isin ([2,3,4])) & (df_integrated['Significance_Status_Post_Original'].isin ([2,3,4]))]
    fout.write ("Number of users with more than 7 datapoints and significant model for within and post intervals in original : " + str (df_within_post_original_acceptable.shape[0]) + "\n")

#----------------------
def CorrPval_Calculator (df):
    dfc = df.corr()
    pval = df.corr(method=lambda x, y: pearsonr(x, y)[1])-np.eye(dfc.shape[0])
    p = pval.applymap(lambda x: ''.join(['*' for t in [0.01,0.05,0.1] if x<=t]))
    df_CorrPval = dfc.round(3).astype(str) + p
    return df_CorrPval
#----------------------
def CorrPval_Calculator_TweetType_Interval (df, address_output_1):
    # Pre -  Within 
    df_Retweet_pre_within_significant = df[(df['Significance_Status_Pre_Retweet'].isin([2,3,4])) & (df['Significance_Status_Within_Retweet'].isin([2,3,4]))]
    df_Retweet_pre_within_significant_CorrPval = CorrPval_Calculator (df_Retweet_pre_within_significant)
##    df_Retweet_pre_within_significant_CorrPval ['SlopeChangePreWithinRetweet']

    df_Reply_pre_within_significant = df[(df['Significance_Status_Pre_Reply'].isin([2,3,4])) & (df['Significance_Status_Within_Reply'].isin([2,3,4]))]
    df_Reply_pre_within_significant_CorrPval = CorrPval_Calculator (df_Reply_pre_within_significant)
##    df_Reply_pre_within_significant_CorrPval ['SlopeChangePreWithinReply']

    df_Quote_pre_within_significant = df[(df['Significance_Status_Pre_Quote'].isin([2,3,4])) & (df['Significance_Status_Within_Quote'].isin([2,3,4]))]
    df_Quote_pre_within_significant_CorrPval = CorrPval_Calculator (df_Quote_pre_within_significant)
##    df_Quote_pre_within_significant_CorrPval ['SlopeChangePreWithinQuote']

    df_Original_pre_within_significant = df[(df['Significance_Status_Pre_Original'].isin([2,3,4])) & (df['Significance_Status_Within_Original'].isin([2,3,4]))]
    df_Original_pre_within_significant_CorrPval = CorrPval_Calculator (df_Original_pre_within_significant)
##    df_Original_pre_within_significant_CorrPval ['SlopeChangePreWithinOriginal']

    # Within - Post
    df_Retweet_within_post_significant = df[(df['Significance_Status_Within_Retweet'].isin([2,3,4])) & (df['Significance_Status_Post_Retweet'].isin([2,3,4]))]
    df_Retweet_within_post_significant_CorrPval = CorrPval_Calculator (df_Retweet_within_post_significant)
##    df_Retweet_within_post_significant_CorrPval ['SlopeChangeWithinPostRetweet']

    df_Reply_within_post_significant = df[(df['Significance_Status_Within_Reply'].isin([2,3,4])) & (df['Significance_Status_Post_Reply'].isin([2,3,4]))]
    df_Reply_within_post_significant_CorrPval = CorrPval_Calculator (df_Reply_within_post_significant)
##    df_Reply_within_post_significant_CorrPval ['SlopeChangeWithinPostReply']

    df_Quote_within_post_significant = df[(df['Significance_Status_Within_Quote'].isin([2,3,4])) & (df['Significance_Status_Post_Quote'].isin([2,3,4]))]
    df_Quote_within_post_significant_CorrPval = CorrPval_Calculator (df_Quote_within_post_significant)
##    df_Quote_within_post_significant_CorrPval ['SlopeChangeWithinPostQuote']

    df_Original_within_post_significant = df[(df['Significance_Status_Within_Original'].isin([2,3,4])) & (df['Significance_Status_Post_Original'].isin([2,3,4]))]
    df_Original_within_post_significant_CorrPval = CorrPval_Calculator (df_Original_within_post_significant)
##    df_Original_within_post_significant_CorrPval ['SlopeChangeWithinPostOriginal']

    data = [df_Retweet_pre_within_significant_CorrPval ['SlopeChangePreWithinRetweet'],df_Reply_pre_within_significant_CorrPval ['SlopeChangePreWithinReply'],df_Quote_pre_within_significant_CorrPval ['SlopeChangePreWithinQuote'],df_Original_pre_within_significant_CorrPval ['SlopeChangePreWithinOriginal'],df_Retweet_within_post_significant_CorrPval ['SlopeChangeWithinPostRetweet'],df_Reply_within_post_significant_CorrPval ['SlopeChangeWithinPostReply'],df_Quote_within_post_significant_CorrPval ['SlopeChangeWithinPostQuote'],df_Original_within_post_significant_CorrPval ['SlopeChangeWithinPostOriginal']]
    headers = ['PreWithinRetweet','PreWithinReply','PreWithinQuote','PreWithinOriginal','WithinPostRetweet','WithinPostReply','WithinPostQuote','WithinPostOriginal']
    df_corrpval_results = pd.concat(data, axis=1, keys=headers)
##    df_corrpval_results.to_csv(address_output_1, index=False)
    df_corrpval_results.to_csv(address_output_1)

#----------------------
def main():
    address_integrated = "/home/abodaghi/Twitter_Project/Data_Processing/Results/User_Data_Analysis/User_Data_File_Integrated/UDFI.csv"
    address_input_1 = '/mnt/tb/amirdata/Merge_All_Waves.zip.hdf5'
    address_output = "/home/abodaghi/Twitter_Project/Data_Processing/Results/User_Data_Analysis/User_Data_File_Integrated_Analysis/UDFIA.txt"
    address_output_1 = "/home/abodaghi/Twitter_Project/Data_Processing/Results/User_Data_Analysis/User_Data_File_Integrated_Analysis/UDFIA.csv"
    There_Is_Mismatch, df_integrated = Checking_MisMatch (address_integrated, address_input_1)
    if There_Is_Mismatch:
        print ("Sorry First you need to solve the mismatch") #usually it happened when in pandas dataframe, after saving to .csv file, big numbers after 15 digits changed to another number while type was 'float', but after changing the 'id' column to 'int' before saveing to .csv the issue solved.
    else:
        fout = open (address_output,"w")
        fout = Finding_Number_Users_With_More_Than_7_Datapoints (df_integrated, fout)
        Number_Significant_Users (df_integrated, fout)
        CorrPval_Calculator_TweetType_Interval (df_integrated, address_output_1)
        fout.close()
#----------------------
if __name__ == '__main__':
        main ()
