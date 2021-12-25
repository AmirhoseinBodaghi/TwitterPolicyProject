from pandas import read_csv
import time
#==========================
def main():
        f = open ('/mnt/tb/amirdata/Merged_File_Reading_Info.txt', 'w')
        f.write("reading the dataframe and getting its details \r\n")
        t0 = time.perf_counter() 
        df = read_csv("/mnt/tb/amirdata/Merge_All_Waves.zip", compression='zip', encoding="ISO-8859-1", engine='python')
        f.write("Duplicate Tweetid : " + str(df['tweetid'].duplicated().any()) + '\r\n')
        df = df.drop_duplicates('tweetid').reset_index(drop=True)
        f.write("The first timestamp after sorting : " + str(df['created_at'].iloc[[0]].values)+'\r\n') #getting the first timestamp
        f.write("The last timestamp after sorting: " + str(df['created_at'].iloc[[-1]].values)+'\r\n') #getting the last timestamp
        f.write("Data Shape : " + str(df.shape) + '\r\n') #getting the data shape
        f.write("Duplicate Tweetid : " + str(df['tweetid'].duplicated().any()) + '\r\n') #to see if there is a tweetid duplicate in the data yet!
        f.write("Columns Labels : " + str(df.columns.values) + '\r\n') #to get the column labels
        Size_Frame = df.groupby('id').size()
        number_unique_users = len(Size_Frame.index)
        f.write("Number of Unique Users : " + str(number_unique_users) + '\r\n') #to get the number of unique users in the final dataframe    
        t1 = time.perf_counter()
        f.write(str(t1-t0) + '\r\n')
        
        f.close()
#==========================
main()
input ("press enter to end")
