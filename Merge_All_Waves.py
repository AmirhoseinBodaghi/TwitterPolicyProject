from pandas import read_csv
from pandas import to_datetime
from pandas import concat
import csv
import time
#==========================
def convert_datetime (value): #designed for wave 1 to turn the format of its values at the column 'created_at' to the standard format 'date time' (like the other waves)
    return str(to_datetime(value))[:-6]
#==========================
def main():
    f = open ('/mnt/tb/amirdata/Merge_All_Waves_Info.txt', 'w')

    #----------------------------------------------------
    # Task1: Reading and Reforming Wave1
    f.write("reading data from Wave 1 and droping empty columns and turning the values in the column of 'created_at' into standard format of 'date time' and sorting the dataframe based on 'created_at' ... \r\n")
    t0 = time.perf_counter()    
    df1 = read_csv("//mnt//zhudata//CSV//tweets_csv//tweets_wave1_csv.zip", compression='zip', encoding="ISO-8859-1", low_memory=False)
    df1 = df1.drop(columns = ['screen_name','name','coordinates','geo', 'Unnamed: 0'])
    df1 = df1.drop_duplicates('tweetid').reset_index(drop=True)
    df1['created_at']=df1['created_at'].apply(convert_datetime)
    df1 = df1.sort_values(by='created_at').reset_index(drop=True)
    f.write("The first timestamp after sorting : " + str(df1['created_at'].iloc[[0]].values)+'\r\n') #getting the first timestamp
    f.write("The last timestamp after sorting: " + str(df1['created_at'].iloc[[-1]].values)+'\r\n') #getting the last timestamp
    f.write("Data Shape : " + str(df1.shape) + '\r\n') #getting the data shape
    f.write("Duplicate Tweetid : " + str(df1['tweetid'].duplicated().any()) + '\r\n') #to see if there is a tweetid duplicate in the data yet!
    f.write("Columns Labels : " + str(df1.columns.values) + '\r\n') #to get the column labels
    Size_Frame = df1.groupby('id').size()
    number_unique_users = len(Size_Frame.index)
    f.write("Number of Unique Users : " + str(number_unique_users) + '\r\n') #to get the number of unique users in the final dataframe 
    t00 = time.perf_counter()
    f.write(str(t00-t0) + '\r\n')

    #----------------------------------------------------
    # Task2: Reading and Reforming Wave2
    f.write("reading data from Wave 2 and droping empty columns ... \r\n")
    t1 = time.perf_counter()    
    df2 = read_csv("//mnt//zhudata//CSV//tweets_csv//tweets_wave2_csv.zip", compression='zip', encoding="ISO-8859-1", low_memory=False)
    df2 = df2.drop(columns = ['screen_name','name','coordinates','geo', 'Unnamed: 0'])
    df2 = df2.drop_duplicates('tweetid').reset_index(drop=True)
    df2 = df2.sort_values(by='created_at').reset_index(drop=True)
    f.write("The first timestamp after sorting : " + str(df2['created_at'].iloc[[0]].values)+'\r\n') #getting the first timestamp
    f.write("The last timestamp after sorting: " + str(df2['created_at'].iloc[[-1]].values)+'\r\n') #getting the last timestamp
    f.write("Data Shape : " + str(df2.shape) + '\r\n') #getting the data shape
    f.write("Duplicate Tweetid : " + str(df2['tweetid'].duplicated().any()) + '\r\n') #to see if there is a tweetid duplicate in the data yet!
    f.write("Columns Labels : " + str(df2.columns.values) + '\r\n') #to get the column labels
    Size_Frame = df2.groupby('id').size()
    number_unique_users = len(Size_Frame.index)
    f.write("Number of Unique Users : " + str(number_unique_users) + '\r\n') #to get the number of unique users in the final dataframe 
    t2 = time.perf_counter()
    f.write(str(t2-t1) + '\r\n')

    #----------------------------------------------------
    # Task3: Reading and Reforming Wave3
    f.write("reading data from Wave 3 and droping empty columns ... \r\n")
    t3 = time.perf_counter()
    df3 = read_csv("//mnt//zhudata//CSV//tweets_csv//tweets_wave3_csv.zip", compression='zip', encoding="ISO-8859-1", low_memory=False)
    df3 = df3.drop(columns = ['screen_name','name','coordinates','geo', 'Unnamed: 0'])
    df3 = df3.drop_duplicates('tweetid').reset_index(drop=True)
    df3 = df3.sort_values(by='created_at').reset_index(drop=True)
    f.write("The first timestamp after sorting : " + str(df3['created_at'].iloc[[0]].values)+'\r\n') #getting the first timestamp
    f.write("The last timestamp after sorting: " + str(df3['created_at'].iloc[[-1]].values)+'\r\n') #getting the last timestamp
    f.write("Data Shape : " + str(df3.shape) + '\r\n') #getting the data shape
    f.write("Duplicate Tweetid : " + str(df3['tweetid'].duplicated().any()) + '\r\n') #to see if there is a tweetid duplicate in the data yet!
    f.write("Columns Labels : " + str(df3.columns.values) + '\r\n') #to get the column labels
    Size_Frame = df3.groupby('id').size()
    number_unique_users = len(Size_Frame.index)
    f.write("Number of Unique Users : " + str(number_unique_users) + '\r\n') #to get the number of unique users in the final dataframe 
    t4 = time.perf_counter()
    f.write(str(t4-t3) + '\r\n')

    #----------------------------------------------------
    # Task4: Reading and Reforming Wave8
    f.write("reading data from Wave 8 and droping empty columns and reversing the dataframe upside down ... \r\n")
    t7 = time.perf_counter()
    df8 = read_csv("//mnt//zhudata//CSV//tweets_csv//tweets_wave8_csv.gz", compression='gzip', encoding="ISO-8859-1", low_memory=False)
    df8 = df8.drop(columns = ['screen_name','name','coordinates','geo'])
    df8 = df8.drop_duplicates('tweetid').reset_index(drop=True)       
    df8 = df8.sort_values(by='created_at').reset_index(drop=True) #to make sure sorting has no effect because the dataframe had been sorted at first
    f.write("The first timestamp after sorting : " + str(df8['created_at'].iloc[[0]].values)+'\r\n') #getting the first timestamp
    f.write("The last timestamp after sorting: " + str(df8['created_at'].iloc[[-1]].values)+'\r\n') #getting the last timestamp
    f.write("Data Shape : " + str(df8.shape) + '\r\n') #getting the data shape
    f.write("Duplicate Tweetid : " + str(df8['tweetid'].duplicated().any()) + '\r\n') #to see if there is a tweetid duplicate in the data yet!
    f.write("Columns Labels : " + str(df8.columns.values) + '\r\n') #to get the column labels
    Size_Frame = df8.groupby('id').size()
    number_unique_users = len(Size_Frame.index)
    f.write("Number of Unique Users : " + str(number_unique_users) + '\r\n') #to get the number of unique users in the final dataframe 
    t8 = time.perf_counter()
    f.write(str(t8-t7) + '\r\n')
    
    #----------------------------------------------------
    # Task5: Concatenation and Writing the Final Dataframe to Output .zip file
    f.write("concatenation all waves into one and then droping the all duplicates['tweetid'] (all rows whose 'tweetid' value has the same with another row will be deleted except the first occurance) and then getting the info of the dataframe and finally writing it to a csv compressed zip file ... \r\n")
    t9 = time.perf_counter()

##    tio = df2.loc[df2['tweetid'].isin(df1['tweetid'])]
##    tio = tio.sort_values(by='created_at').reset_index(drop=True) #to make sure sorting has no effect because the dataframe had been sorted at first
##    f.write("The first timestamp after sorting tio : " + str(tio['created_at'].iloc[[0]].values)+'\r\n') #getting the first timestamp
##    f.write("The last timestamp after sorting tio : " + str(tio['created_at'].iloc[[-1]].values)+'\r\n') #getting the last timestamp
##    f.write("Data Shape tio: " + str(tio.shape) + '\r\n') #getting the data shape
##    f.write("Duplicate Tweetid tio : " + str(tio['tweetid'].duplicated().any()) + '\r\n') #to see if there is a tweetid duplicate in the data yet!
##    f.write("Columns Labels tio: " + str(tio.columns.values) + '\r\n') #to get the column labels
##    Size_Frame = tio.groupby('id').size()
##    number_unique_users = len(Size_Frame.index)
##    f.write("Number of Unique Users tio: " + str(number_unique_users) + '\r\n') #to get the number of unique users in the final dataframe    
    
    df = concat([df1,df2,df3,df8], ignore_index = True)
    df = df.drop_duplicates('tweetid').reset_index(drop=True)       
    df = df.sort_values(by='created_at').reset_index(drop=True) #to make sure sorting has no effect because the dataframe had been sorted at first
    f.write("The first timestamp after sorting : " + str(df['created_at'].iloc[[0]].values)+'\r\n') #getting the first timestamp
    f.write("The last timestamp after sorting: " + str(df['created_at'].iloc[[-1]].values)+'\r\n') #getting the last timestamp
    f.write("Data Shape : " + str(df.shape) + '\r\n') #getting the data shape
    f.write("Duplicate Tweetid : " + str(df['tweetid'].duplicated().any()) + '\r\n') #to see if there is a tweetid duplicate in the data yet!
    f.write("Columns Labels : " + str(df.columns.values) + '\r\n') #to get the column labels
    Size_Frame = df.groupby('id').size()
    number_unique_users = len(Size_Frame.index)
    f.write("Number of Unique Users : " + str(number_unique_users) + '\r\n') #to get the number of unique users in the final dataframe    
    df.to_csv("/mnt/tb/amirdata/Merge_All_Waves.zip", index = False, quoting = csv.QUOTE_NONNUMERIC, compression = 'zip')
    t10 = time.perf_counter()
    f.write(str(t10-t9) + '\r\n')
    #----------------------------------------------------

    f.close()
#==========================
main()
input ("press enter to end")
