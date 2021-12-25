import matplotlib
matplotlib.use('Agg') #this is because generating matplot figures on the server requires X running, however when we bring up idle we automatically do it by calling xmanager, but when we run the code in the backend (by nohup) then we need to bring this command
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot
import matplotlib.ticker as mticker
from pandas.plotting import lag_plot
from pandas import read_csv
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from multiprocessing import Process, current_process 
#==========================
def Read_Dataframe (f,address_input):
        f.write("reading the dataframe and getting its details \r\n")
        t0 = time.perf_counter()
##        df = read_csv(address_input, compression='zip', encoding="ISO-8859-1", engine='python', nrows=10000000)
        df = read_csv(address_input, compression='zip', encoding="ISO-8859-1", engine='python')
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
        f.write("------------------------------" + '\r\n')
        f.write("Time Elapsed in Read_Dataframe : " + str(t1-t0) + '\r\n')
        f.write("------------------------------" + '\r\n')
        return df , f
#==========================
def Turn_to_Favorite_Dataframe(df , f):
        # turning the dataframe into this:
        # timestamp id tweetid original retweet quote reply
        # ...        .   ...     1\0      1\0    1\0   1\0
        # 1 == True , 0 == False

        t0 = time.perf_counter()
        # first turning the date into standard pandas timestamp - and then setting it as index
        df['created_at'] = pd.to_datetime(df['created_at'])
        df.set_index('created_at', inplace=True)
        
        # turning retweet status into digital (1\0)
        df['retweeted_userid'] = df['retweeted_userid'].fillna(0)
        df.loc[(df.retweeted_userid != 0),'retweeted_userid']= 1        
        
        # turning quote status into digital (1\0)
        df['is_quote_status'] = df['is_quote_status'].replace([True],1)
        df['is_quote_status'] = df['is_quote_status'].replace([False],0)

        # turning reply status into digital (1\0)
        df['replyto_userid'] = df['replyto_userid'].fillna(0) #replace NaN by 0
        df.loc[(df.replyto_userid != 0),'replyto_userid']= 1

        
        # making sure the format of the values are int
        cols = ['id', 'tweetid', 'retweeted_userid', 'is_quote_status', 'replyto_userid']
        df[cols] = df[cols].applymap(np.int64)

        # creating a new column for the original status
        df['original'] = (df['retweeted_userid'] + df['is_quote_status'] + df['replyto_userid'] == 0).astype(int) 

        # keeping the favorite columns and deleting the rest of the columns
        col_list = ['id', 'tweetid', 'original', 'retweeted_userid', 'is_quote_status', 'replyto_userid']
        df = df[col_list]

        # changing the name of the favorite columns
        df.columns = ['id', 'tweetid', 'original', 'retweet', 'quote', 'reply']
        
        t1 = time.perf_counter()
        f.write("------------------------------" + '\r\n')
        f.write("Time Elapsed in Turn_to_Favorite_Dataframe : " + str(t1-t0) + '\r\n')
        f.write("------------------------------" + '\r\n')
        
        return df , f
#==========================
def Basic_Statistics (df, address_output , f):
        t0 = time.perf_counter()
        # first we add a new column named 'mode' to the dataframe. This column is a tuple of 4 elements (original, retweet, quote, reply) that shows the status of the tweet. for ex. (0,1,0,1) means the tweet is a retweet in reply 
        df['mode'] = df.apply(lambda row: (row.original, row.retweet, row.quote, row.reply), axis=1)
        # then based on this new column ('mode') we group the dataframe to see size of each group and plot it in a pie chart
        Size_Mode = df.groupby('mode').size()
        labels = []
        sizes = []
        for i in range (len(Size_Mode.index)):
                labels.append (Size_Mode.index[i])
                sizes.append (Size_Mode[i])

        x_pos = np.arange(len(labels))
        fig1, ax1 = pyplot.subplots()
        bars = pyplot.bar(x_pos, height = sizes)
        pyplot.xticks(x_pos, labels, fontsize = 10, rotation = 40)
        for bar in bars:
            yval = bar.get_height()
            pyplot.text(bar.get_x() + 0.15, yval + 1, yval, fontsize = 8)
        
        pyplot.tight_layout()
        pyplot.savefig(address_output + 'bar_plot.tif', dpi=500)

        t1 = time.perf_counter()
        f.write("------------------------------" + '\r\n')
        f.write("Time Elapsed in Basic_Statistics : " + str(t1-t0) + '\r\n')
        f.write("------------------------------" + '\r\n')

        return f
##        pyplot.show()
#==========================
def General_Visualization (df, address_output, f):
        t0 = time.perf_counter()
        
        df_general = df.groupby([df.index.date]).sum()
        df_general = df_general.drop(['id','tweetid'], axis=1) # ----> turning the dataframe into :  #day  [#original, #retweet, #quote, #reply]
        
        #------------------------------------------------------
        # line plot 
        fig, ax = pyplot.subplots(2, 2)
##        myLocator = mticker.MultipleLocator(250)

        ax[0,0].plot(df_general['original'])
        ax[0,0].title.set_text('a) Original')
        ax[0,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[0,0].tick_params(axis='y', labelsize=6)
##        ax[0,0].tick_params(axis='y', labelsize=6)


        ax[0,1].plot(df_general['retweet'])
        ax[0,1].title.set_text('b) Retweet')
        ax[0,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[0,1].tick_params(axis='y', labelsize=6)
##        ax[0,1].tick_params(axis='y', labelsize=6)

        ax[1,0].plot(df_general['quote'])
        ax[1,0].title.set_text('c) Quote')
        ax[1,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[1,0].tick_params(axis='y', labelsize=6)
##        ax[1,0].tick_params(axis='y', labelsize=6)

        ax[1,1].plot(df_general['reply'])
        ax[1,1].title.set_text('d) Reply')
        ax[1,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[1,1].tick_params(axis='y', labelsize=6)
##        ax[1,1].tick_params(axis='y', labelsize=6)


##        fig.autofmt_xdate() #rotates the xtick labels
        pyplot.tight_layout()
        
##        pyplot.setp(ax.get_xticklabels(), visible=True)
        pyplot.savefig(address_output + 'general_line_plot.tif', dpi=500)
##        pyplot.show()
        #------------------------------------------------------

        #------------------------------------------------------
        # hist plot (the frequency of number of tweets in different format per day) 
        fig, ax = pyplot.subplots(2, 2)
        ax[0,0].hist(df_general['original'], bins = 8)
        ax[0,0].title.set_text('a) Original')
        ax[0,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[0,0].tick_params(axis='y', labelsize=6)


        ax[0,1].hist(df_general['retweet'], bins = 8)
        ax[0,1].title.set_text('b) Retweet')
        ax[0,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[0,1].tick_params(axis='y', labelsize=6)


        ax[1,0].hist(df_general['quote'], bins = 8)
        ax[1,0].title.set_text('c) Quote')
        ax[1,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[1,0].tick_params(axis='y', labelsize=6)


        ax[1,1].hist(df_general['reply'], bins = 8)
        ax[1,1].title.set_text('d) Reply')
        ax[1,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[1,1].tick_params(axis='y', labelsize=6)


##        fig.autofmt_xdate() #rotates the xtick labels
        pyplot.tight_layout()
##        pyplot.setp(ax.get_xticklabels(), visible=True)
        pyplot.savefig(address_output + 'general_hist_plot.tif', dpi=500)
##        pyplot.show()
        #------------------------------------------------------

        #------------------------------------------------------
        # boxplot plot 
        fig, ax = pyplot.subplots(2, 2)
        ax[0,0].boxplot(df_general['original'])
        ax[0,0].title.set_text('a) Original')

        ax[0,1].boxplot(df_general['retweet'])
        ax[0,1].title.set_text('b) Retweet')

        ax[1,0].boxplot(df_general['quote'])
        ax[1,0].title.set_text('c) Quote')

        ax[1,1].boxplot(df_general['reply'])
        ax[1,1].title.set_text('d) Reply')

        pyplot.tight_layout()
        pyplot.savefig(address_output + 'general_box_plot.tif', dpi=500)
##        pyplot.show()
        #------------------------------------------------------

        #------------------------------------------------------
        # lag_plot (shows the relationship between an observation and the previous observation)
        fig, ax = pyplot.subplots(2, 2)

        lag_plot(df_general['original'], lag=1, ax =ax[0,0])
        ax[0,0].title.set_text('a) Original')
        ax[0,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[0,0].tick_params(axis='y', labelsize=6)
        
        lag_plot(df_general['retweet'], lag=1, ax =ax[0,1])
        ax[0,1].title.set_text('b) Retweet')
        ax[0,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[0,1].tick_params(axis='y', labelsize=6)
        
        lag_plot(df_general['quote'], lag=1, ax =ax[1,0])
        ax[1,0].title.set_text('c) Quote')
        ax[1,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[1,0].tick_params(axis='y', labelsize=6)
        
        lag_plot(df_general['reply'], lag=1, ax =ax[1,1])
        ax[1,1].title.set_text('d) Reply')
        ax[1,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
        ax[1,1].tick_params(axis='y', labelsize=6)
        
        pyplot.tight_layout()        
        pyplot.savefig(address_output + 'general_lag_plot.tif', dpi=500)
##        pyplot.show()
        #------------------------------------------------------

        #------------------------------------------------------
        # autocorrelation_plot (shows the relationship between an observation and the previous observation)
##        fig, ax = pyplot.subplots(4, 2, constrained_layout=True)
        fig, ax = pyplot.subplots(4, 2, constrained_layout=False)
##        pyplot.rc('font', size=6)
        modes = ['original', 'retweet', 'quote', 'reply']
        titles = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]
        counter_row = 0
        counter_column = 0
        for i in modes:
                autocorrelation_plot(df_general[i], ax =ax[counter_row,counter_column])
                result = adfuller(df_general[i])  #Dickey_Fuller Test
                labels = []
                labels.append('ADF Statistic: %.2f' % result[0])
                labels.append('p-value: %.2f' % result[1])
                ax[counter_row,counter_column].legend(labels, loc='upper right', fontsize=6, fancybox=True, handlelength=0, handletextpad=0)
                ax[counter_row,counter_column].set_title(titles[counter_row][counter_column] + ') ' + i + ' before differencing', fontsize=8)
                ax[counter_row,counter_column].set_ylabel('Autocorrelation', fontsize=6)
                ax[counter_row,counter_column].set_xlabel('Lag', fontsize=6)
                ax[counter_row,counter_column].tick_params(axis='x', labelsize=6)
                ax[counter_row,counter_column].tick_params(axis='y', labelsize=6)
                ax[counter_row,counter_column].grid(False)
                
                # if result[1] > 0.05 means our time series is not stationary and we need to do somthing (like differencing) to make it stationary
                counter_column += 1
                if result[1] > 0.05:                
                        # differencing to turn to stationary 
                        df_general[i] = df_general[i].diff()
                        # the above command leaves the dataframe with the first row (because diff comes with 1 differencing as default) as NaN, so by the following command we replace the NaN with 0
                        df_general[i] = df_general[i].fillna(0)               
                        # Again Checking the stationarity
                        result = adfuller(df_general[i]) #Dickey_Fuller Test
                        autocorrelation_plot(df_general[i], ax =ax[counter_row,counter_column])
                        labels = []
                        labels.append('ADF Statistic: %.2f' % result[0])
                        labels.append('p-value: %.2f' % result[1])
                        ax[counter_row,counter_column].legend(labels, loc='upper right', fontsize=6, fancybox=True, handlelength=0, handletextpad=0)
                        ax[counter_row,counter_column].set_title(titles[counter_row][counter_column] + ') ' + i + ' after 1-step differencing', fontsize=8)
                        ax[counter_row,counter_column].set_ylabel('Autocorrelation', fontsize=6)
                        ax[counter_row,counter_column].set_xlabel('Lag', fontsize=6)
                        ax[counter_row,counter_column].tick_params(axis='x', labelsize=6)
                        ax[counter_row,counter_column].tick_params(axis='y', labelsize=6)
                        ax[counter_row,counter_column].grid(False)
                        # if still result[1] > 0.05 then you need to increase the differencing from one time step to more than one, or use another method for making the time series stationary
                else:
                        result = adfuller(df_general[i]) #Dickey_Fuller Test
                        autocorrelation_plot(df_general[i], ax =ax[counter_row,counter_column])
                        labels = []
                        labels.append('ADF Statistic: %.2f' % result[0])
                        labels.append('p-value: %.2f' % result[1])
                        ax[counter_row,counter_column].legend(labels, loc='upper right', fontsize=6, fancybox=True, handlelength=0, handletextpad=0)
                        ax[counter_row,counter_column].set_title(titles[counter_row][counter_column] + ') ' + i + ' without differencing', fontsize=8)
                        ax[counter_row,counter_column].set_ylabel('Autocorrelation', fontsize=6)
                        ax[counter_row,counter_column].tick_params(axis='x', labelsize=6)
                        ax[counter_row,counter_column].tick_params(axis='y', labelsize=6)
                        ax[counter_row,counter_column].set_xlabel('Lag', fontsize=6)
                        ax[counter_row,counter_column].grid(False)
                        

                counter_column = 0
                counter_row += 1

        pyplot.tight_layout()
        pyplot.savefig(address_output + 'autocorrelation_plot.tif', dpi=500)

        t1 = time.perf_counter()
        f.write("------------------------------" + '\r\n')
        f.write("Time Elapsed in General_Visualization : " + str(t1-t0) + '\r\n')
        f.write("------------------------------" + '\r\n')

        return f
##        pyplot.show()
        #------------------------------------------------------        
#==========================
def main():

        t0 = time.perf_counter()
        # Setting the addresses
        address_output = '/home/abodaghi/Twitter_Project/Data_Processing/Results/Data_Exploration_MP/'
        address_input = '/mnt/tb/amirdata/Merge_All_Waves.zip'

        # This text file is going to contain the details of our data along with the time spent to process each part of the program by the server
        f = open (address_output + 'Data_Exploration_Info.txt', 'w')
        
        
        # Reading the dataframe
        df , f = Read_Dataframe (f,address_input)

##        # showing the data types
##        pd.set_option('display.max_rows', df.shape[0]+1) #to show all the rows and not just show three dots for them (...)
        
        # Changing the dataframe into the favorite dataframe ----> ['id', 'tweetid', 'original', 'retweet', 'quote', 'reply'] with digital values
        df, f = Turn_to_Favorite_Dataframe(df, f)


##        # Multiprocessing
        tp1s = time.perf_counter()
        process_1 = Process(target = General_Visualization   , args = (df, address_output , f))
        
        tp2s = time.perf_counter()
        process_2 = Process(target = Basic_Statistics  , args = (df, address_output , f))
        
        
        process_1.start()
        process_2.start()

        process_1.join()
        tp1e = time.perf_counter()
        f.write("------------------------------" + '\r\n')
        f.write('Total Time of the process 1 : ' + str(tp1e-tp1s) + '\r\n')
        f.write("------------------------------" + '\r\n')
        
        process_2.join()        
        tp2e = time.perf_counter()
        f.write("------------------------------" + '\r\n')
        f.write('Total Time of the process 2 : ' + str(tp2e-tp2s) + '\r\n')
        f.write("------------------------------" + '\r\n')
        
##        # Basic Statistics of the Data
##        Basic_Statistics (df, address_output)
##        t3 = time.perf_counter()
##        f.write('Time for Basic Statistics : ' + str(t3-t2) + '\r\n')
##
##        
##        # Visualization
##        General_Visualization (df, address_output)
##        t4 = time.perf_counter()
##        f.write('Time for General_Visualization : ' + str(t4-t3) + '\r\n')

        
        
        t1 = time.perf_counter()
        f.write("------------------------------" + '\r\n')
        f.write('Total Time of this program with Multiprocessing : ' + str(t1-t0) + '\r\n')
        f.write("------------------------------" + '\r\n')
        f.close()
#==========================
##main()
##input ("press enter to end")
if __name__ == '__main__':
        main ()
