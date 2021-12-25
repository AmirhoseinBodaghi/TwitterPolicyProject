import vaex as vx
import matplotlib
matplotlib.use('Agg') #this is because generating matplot figures on the server requires X running, however when we bring up idle we automatically do it by calling xmanager, but when we run the code in the backend (by nohup) then we need to bring this command
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot
import matplotlib.ticker as mticker
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
#------------
# Just to make sure there is no difference between retweeted_userid, retweeted_tweetid and retweeted_created as a representative for retweet being
def Just_to_Make_Sure_We_Choose_Correct_Column_as_Representative_of_Being_Retweet(address_input):
    df = vx.open (address_input)
    ## knowing if there is any difference between these features 'retweeted_userid' , 'retweeted_tweetid' , 'retweeted_created'
    ## because we want to choose one of them as the feature that shows the retweet being of the tweet
    ## we see all three of the features could be used equally as the above desired feature (when a tweet is a retweet, the value of those features would be 1 by the following commands)
    df['retweeted_userid'] = df['retweeted_userid'].fillna(0)
    df2 = df [df['retweeted_userid'] != 0]
    print (df2.shape)

    df['retweeted_tweetid'] = df['retweeted_tweetid'].fillna(0)
    df3 = df [df['retweeted_tweetid'] != 0]
    print (df3.shape)


    df['retweeted_created']=df['retweeted_created'].where(df['retweeted_created'].values[i].as_py() == None, 0)  #this simple command took a couple hours to be devised! hold to it when you need to handle a large_string dtype object
    df4 = df [df['retweeted_created'] != 0]
    print (df4.shape)
    #Since the all three (df2, df3, df4) dataframes were in same shape, we noticed it doesn't matter to choose which one (otherwise the bigger one would be in priority, or we would think about it) so we chose retweeted_userid.
    #there is no need to run this function, just one time we did it to make sure there is no difference between the size of those three dataframes, thats it. :)
#------------
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

##    df['mode'] = array(list(zip(df.retweeted_userid.values,df.is_quote_status.values,df.replyto_userid.values,df.original.values)))
##    df['modes'] = array(str(tuple(zip(df.retweeted_userid.values,df.is_quote_status.values,df.replyto_userid.values,df.original.values))))
##    df['mode'] = df.retweeted_userid.astype(str) + df.is_quote_status.astype(str) + df.replyto_userid.astype(str) + df.original.astype(str)
    df['modes'] = 1000*df.retweeted_userid + 100*df.is_quote_status + 10*df.replyto_userid + df.original 


    df['created_at'] = df.created_at.astype('datetime64') #turn the created_at column to timestamp



    return df
#------------
# Gives the information about different types and their frequencies in the dataset including hybrid modes
def Types_Info (df, address_output, with_hybrid):    
    # Bar Plot for Modes in Dataset
    df_modes = df.groupby(df.modes, agg='count')
    labels = []
    sizes = []
    for i in range (df_modes['modes'].shape[0]):
        labels.append (df_modes['modes'].values[i])
        sizes.append (df_modes['count'].values[i])
    x_pos = np.arange(df_modes['modes'].shape[0])
    bars = pyplot.bar(x_pos, height = sizes)
    pyplot.xticks(x_pos, labels, fontsize = 10, rotation = 40)
    for bar in bars:
        yval = bar.get_height()
        pyplot.text(bar.get_x() + 0.15, yval + 1, yval, fontsize = 8)

    pyplot.tight_layout()
    if with_hybrid == 1:
        pyplot.savefig(address_output + 'bar_plot_with_hybrid_modes.tif', dpi=300)
        pyplot.close('all') #if we do not put this line here, the next bar plot we draw in this program will have the numbers and texts of this fig messed with its own!
    else :
        pyplot.savefig(address_output + 'bar_plot_without_hybrid_modes.tif', dpi=300)
        pyplot.close('all')
#-------------
# Merging Hybrid Modes into Pure Modes
# Since we had seen the results of the Types_Info_with_Hybrid_Modes function we knew what the hybrids mode are,
# so here we simply merged (1100 and 110) them into pure modes (1000, 100)
# Indeed:
    # retweet of a quote (1100) is a retweet (1000)
    # a reply in the form of quote (110) is a quote (100)
    
def Hybrid_to_Pure_Conversion (df):
    df ['is_quote_status'] = df.func.where(df.retweeted_userid == 1, 0, df.is_quote_status) # turning 1100 mode ---> 1000 mode 
    df ['replyto_userid'] = df.func.where(df.is_quote_status == 1, 0, df.replyto_userid) # turning 110 mode ---> 100 mode
    df ['modes'] = 1000*df.retweeted_userid + 100*df.is_quote_status + 10*df.replyto_userid + df.original # now we don't have hybrid modes in the dataset anymore
    return df 
#-------------
# Provides different general plots for different types 
def General_Visualization (df, address_output):
    # normal Plot

    # we can go on by vaex groupby for dates but the problem would be the x-axis ticks labels which would be the number of days instead of the dates
    # unless we change the number od days into the dates and set the tick lables manually
    # However, the other solution would be to turn the dataframe into pandas dataframe (don't worry this time the runtime of the code doesn't increase that huge but a few seconds)
    # After, turning the dataframe into pandas dataframe we can use the timeseries index (ie putting created_at column as index) and then plots would automatically have dates as the tick labels

    # According to the above explanations we did 'coment_out' the following lines and instead used pandas dataframe
##    df_date = df.groupby(by=[vx.BinnerTime.per_day(df.created_at)]).agg({"retweeted_userid": "sum","is_quote_status": "sum","replyto_userid": "sum","original": "sum"})

##    fig, ax = pyplot.subplots(2, 2)
##
##    ax[0,0].plot(df_date['original'].values)
##    ax[0,0].title.set_text('a) Original')
##    ax[0,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
##    ax[0,0].tick_params(axis='y', labelsize=6)
##
##    ax[0,1].plot(df_date['retweeted_userid'].values)
##    ax[0,1].title.set_text('b) Retweet')
##    ax[0,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
##    ax[0,1].tick_params(axis='y', labelsize=6)
##
##    ax[1,0].plot(df_date['is_quote_status'].values)
##    ax[1,0].title.set_text('c) Quote')
##    ax[1,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
##    ax[1,0].tick_params(axis='y', labelsize=6)
##    
##    ax[1,1].plot(df_date['replyto_userid'].values)
##    ax[1,1].title.set_text('d) Reply')
##    ax[1,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
##    ax[1,1].tick_params(axis='y', labelsize=6)
##    
##    pyplot.tight_layout()
##    pyplot.savefig(address_output + 'general_plot.tif', dpi=300)



    # Turning into pandas dataframe to enjoy timeseries index
    df_pandas = df.to_pandas_df()
    df_pandas['created_at'] = pd.to_datetime(df_pandas['created_at'])
    df_pandas.set_index('created_at', inplace=True)
    df_pandas_date = df_pandas.groupby([df_pandas.index.date]).sum()

    # Dropping the unnessasry columns
    df_pandas_date.drop('modes', axis=1, inplace=True)
    df_pandas_date.drop('id', axis=1, inplace=True)
    df_pandas_date.drop('tweetid', axis=1, inplace=True)
    
    # Setting the start date from 2019-10-09, since we only consider 1 year before the issuing of the policy (Dr Jonathan Suggestion)
    startdate = pd.to_datetime("2019-10-09").date()
    df_pandas_date = df_pandas_date.loc[startdate:]
    
    # Setting the Daily Frequency for the Dataframe and Filling the gaps (days with empty values) with zero values (however since 2020-10-09 to 2021-02-02 there was no empty day (a day with no published tweet from the users of the dataset))
    index = pd.DatetimeIndex(df_pandas_date.index)
    df_pandas_date.set_index(index, inplace=True) # turning the index to timeseries index
    df_pandas_date = df_pandas_date.asfreq(freq = 'D', fill_value=0) # setting the frequency as daily and filling the gaps with days with 0 value for all the columns
    
    fig, ax = pyplot.subplots(2, 2)

    ax[0,0].plot(df_pandas_date['original'])
    ax[0,0].title.set_text('a) Original')
    ax[0,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[0,0].tick_params(axis='y', labelsize=6)

    ax[0,1].plot(df_pandas_date['retweeted_userid'])
    ax[0,1].title.set_text('b) Retweet')
    ax[0,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[0,1].tick_params(axis='y', labelsize=6)

    ax[1,0].plot(df_pandas_date['is_quote_status'])
    ax[1,0].title.set_text('c) Quote')
    ax[1,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[1,0].tick_params(axis='y', labelsize=6)
    
    ax[1,1].plot(df_pandas_date['replyto_userid'])
    ax[1,1].title.set_text('d) Reply')
    ax[1,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[1,1].tick_params(axis='y', labelsize=6)
    
    pyplot.tight_layout()
    pyplot.savefig(address_output + 'general_plot.tif', dpi=300)
    #------------------------------

    #------------------------------
    # Max values (The max values in each type along with the information of the day in which that happened)
    f = open (address_output + 'Max_Values_Info.txt', 'w')
    f.write("The Dataframe of all rows with max value in the column of 'retweeted_userid' : " + '\r\n')
    df_max_retweet = df_pandas_date[df_pandas_date['retweeted_userid'] == df_pandas_date['retweeted_userid'].max()]
    f.write(str(df_max_retweet)+'\r\n')

    f.write("The Dataframe of all rows with max value in the column of 'is_quote_status' : " + '\r\n')
    df_max_quote = df_pandas_date[df_pandas_date['is_quote_status'] == df_pandas_date['is_quote_status'].max()]
    f.write(str(df_max_quote)+'\r\n')

    f.write("The Dataframe of all rows with max value in the column of 'replyto_userid' : " + '\r\n')
    df_max_reply = df_pandas_date[df_pandas_date['replyto_userid'] == df_pandas_date['replyto_userid'].max()]
    f.write(str(df_max_reply)+'\r\n')

    f.write("The Dataframe of all rows with max value in the column of 'original' : " + '\r\n')
    df_max_original = df_pandas_date[df_pandas_date['original'] == df_pandas_date['original'].max()]
    f.write(str(df_max_original)+'\r\n')

    df_pandas_date['sum_all'] = df_pandas_date['retweeted_userid'] + df_pandas_date['is_quote_status'] + df_pandas_date['replyto_userid'] + df_pandas_date['original']

    f.write("The Dataframe of all rows with max value in the total number of tweets : " + '\r\n')
    df_max_total = df_pandas_date[df_pandas_date['sum_all'] == df_pandas_date['sum_all'].max()]
    f.write(str(df_max_total)+'\r\n')

    print (df_pandas_date)
    #------------------------------
    
    # Hist Plot (the frequency of number of tweets in different format per day)
    fig, ax = pyplot.subplots(2, 2)

    ax[0,0].hist(df_pandas_date['original'].values, bins = 20)
    ax[0,0].title.set_text('a) Original')
    ax[0,0].locator_params(axis='x', nbins=13)
    ax[0,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[0,0].tick_params(axis='y', labelsize=6)
    
    ax[0,1].hist(df_pandas_date['retweeted_userid'].values, bins = 20)
    ax[0,1].title.set_text('b) Retweet')
    ax[0,1].locator_params(axis='x', nbins=13)
    ax[0,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[0,1].tick_params(axis='y', labelsize=6)

    ax[1,0].hist(df_pandas_date['is_quote_status'].values, bins = 20)
    ax[1,0].title.set_text('c) Quote')
    ax[1,0].locator_params(axis='x', nbins=13)
    ax[1,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[1,0].tick_params(axis='y', labelsize=6)

    ax[1,1].hist(df_pandas_date['replyto_userid'].values, bins = 20)
    ax[1,1].title.set_text('d) Reply')
    ax[1,1].locator_params(axis='x', nbins=13)
    ax[1,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[1,1].tick_params(axis='y', labelsize=6)

    pyplot.tight_layout()
    pyplot.savefig(address_output + 'general_hist_plot.tif', dpi=300)     
    #------------------------------
    
    # boxplot plot 
    fig, ax = pyplot.subplots(2, 2)
    ax[0,0].boxplot(df_pandas_date['original'].values)
    ax[0,0].title.set_text('a) Original')

    ax[0,1].boxplot(df_pandas_date['retweeted_userid'].values)
    ax[0,1].title.set_text('b) Retweet')

    ax[1,0].boxplot(df_pandas_date['is_quote_status'].values)
    ax[1,0].title.set_text('c) Quote')

    ax[1,1].boxplot(df_pandas_date['replyto_userid'].values)
    ax[1,1].title.set_text('d) Reply')

    pyplot.tight_layout()
    pyplot.savefig(address_output + 'general_box_plot.tif', dpi=300)
    #--------------------------------

    # lag plot (shows the relationship between an observation and the previous observation)
    fig, ax = pyplot.subplots(2, 2)
    lag_plot(df_pandas_date['original'] , lag=1, ax = ax[0,0])
    ax[0,0].title.set_text('a) Original')
    ax[0,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[0,0].tick_params(axis='y', labelsize=6)
        
    lag_plot(df_pandas_date['retweeted_userid'], lag=1, ax = ax[0,1])
    ax[0,1].title.set_text('b) Retweet')
    ax[0,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[0,1].tick_params(axis='y', labelsize=6)
        
    lag_plot(df_pandas_date['is_quote_status'], lag=1, ax = ax[1,0])
    ax[1,0].title.set_text('c) Quote')
    ax[1,0].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[1,0].tick_params(axis='y', labelsize=6)
        
    lag_plot(df_pandas_date['replyto_userid'], lag=1, ax = ax[1,1])
    ax[1,1].title.set_text('d) Reply')
    ax[1,1].tick_params(axis='x', labelsize=6 , labelrotation = 35)
    ax[1,1].tick_params(axis='y', labelsize=6)
        
    pyplot.tight_layout()        
    pyplot.savefig(address_output + 'general_lag_plot.tif', dpi=300)    
    #--------------------------------

    # autocorrelation_plot (shows the relationship between an observation and the previous observation)
    fig, ax = pyplot.subplots(4, 2, constrained_layout=False)
    modes = ['original', 'retweeted_userid', 'is_quote_status', 'replyto_userid']
    titles = [['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h']]
    counter_row = 0
    counter_column = 0
    for i in modes:
        autocorrelation_plot(df_pandas_date[i], ax = ax[counter_row,counter_column])
        result = adfuller(df_pandas_date[i])  #Dickey_Fuller Test
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
            df_pandas_date[i] = df_pandas_date[i].diff()
            # the above command leaves the dataframe with the first row (because diff comes with 1 differencing as default) as NaN, so by the following command we replace the NaN with 0
            df_pandas_date[i] = df_pandas_date[i].fillna(0)               
            # Again Checking the stationarity
            result = adfuller(df_pandas_date[i]) #Dickey_Fuller Test
            autocorrelation_plot(df_pandas_date[i], ax = ax[counter_row,counter_column])
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
            result = adfuller(df_pandas_date[i]) #Dickey_Fuller Test
            autocorrelation_plot(df_pandas_date[i], ax = ax[counter_row,counter_column])
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
    pyplot.savefig(address_output + 'autocorrelation_plot.tif', dpi=300)

##    t1 = time.perf_counter()
##    f.write("------------------------------" + '\r\n')
##    f.write("Time Elapsed in General_Visualization : " + str(t1-t0) + '\r\n')
##    f.write("------------------------------" + '\r\n')
#------------
def main():
    address_input = '/mnt/tb/amirdata/Merge_All_Waves.zip.hdf5'
    address_output = '/home/abodaghi/Twitter_Project/Data_Processing/Results/Data_Exploration_Vaex/'

##    Just_to_Make_Sure_We_Choose_Correct_Column_as_Representative_of_Being_Retweet(address_input)
    df = Desired_Data_Frame(address_input)
    with_hybrid = 1
    Types_Info (df, address_output, with_hybrid)
    df = Hybrid_to_Pure_Conversion (df)
    with_hybrid = 0
    Types_Info (df, address_output, with_hybrid)
    General_Visualization(df, address_output)    
#------------------------
##main()
##input ("press enter to end")
if __name__ == '__main__':
        main ()
