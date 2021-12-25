from pandas import read_csv
#==========================
def main():
    for i in range (8): #There are 8 waves in //mnt//zhudata//CSV//tweets_csv
        j = str (i + 1)        
        print ("Wave" + j + " Info")
        if int(j) < 7 : #Waves 7 and 8 have different formats
            df = read_csv("//mnt//zhudata//CSV//tweets_csv//tweets_wave" + j + "_csv.zip", compression='zip', encoding="ISO-8859-1", low_memory=False)
        else :
            df = read_csv("//mnt//zhudata//CSV//tweets_csv//tweets_wave" + j + "_csv.gz", compression='gzip', encoding="ISO-8859-1", low_memory=False)
            
        df = df.sort_values(by='created_at').reset_index(drop=True)
        print (df) #To find the structure of the dataframe
        print (df['created_at']) #To find the first and last tweet in the dataset
        print (df.columns.values) #To find the column labels
        print ("Duplicate in rows: ", df['tweetid'].duplicated().any()) #To see if there is a duplicate in tweetids
        #---- To find the number of unique users involved in the wave ------
        Size_Frame = df.groupby('id').size() 
        print (Size_Frame)
        print ("Number of Unique Users Involved : ", len(Size_Frame.index))
        #-------------------------------------------------------------------
        print ("===============================")
#==========================
main()
input ("Press Enter to end the program!")
