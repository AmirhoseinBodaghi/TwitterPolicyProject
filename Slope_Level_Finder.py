import vaex as vx
import matplotlib
##matplotlib.use('Agg') #this is because generating matplot figures on the server requires X running, however when we bring up idle we automatically do it by calling xmanager, but when we run the code in the backend (by nohup) then we need to bring this command
import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import matplotlib.ticker as mticke
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.formula.api as sm
from pandas.plotting import autocorrelation_plot
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression import linear_model
from patsy import dmatrices
import statsmodels.graphics.tsaplots as tsa
from statsmodels.tsa.arima.model import ARIMA as ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from pmdarima.arima import ADFTest
import warnings
from multiprocessing import Process, current_process
from datetime import datetime
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
def best_SARIMAX_model_estimation_finder (y_train, X_train_minus_intercept):
    list_orders = []
    list_results = []

    # Getting the results for each set of order
    i = 0
    for p in range (4):
        for d in range (3):
            for q in range (4):
                # Building and Fitting Model
                order = (p,d,q)
##                print ('order : ', order)
##                sarimax_model = ARIMA(endog=y_train, exog=X_train_minus_intercept, order=order, initialization='approximate_diffuse')
##                sarimax_model = ARIMA(endog=y_train, exog=X_train_minus_intercept, order=order, enforce_stationarity=False)

##                sarimax_model = ARIMA(endog=y_train, exog=X_train_minus_intercept, order=order)

##                if order != (0,0,0): #this order is not a SARIMAX or ARIMAX or AR or anything :)
                sarimax_model = SARIMAX(endog=y_train, exog=X_train_minus_intercept, order=order, enforce_stationarity=False) 
                sarimax_results = sarimax_model.fit()

                # Reading the results for each set of order (p, d, q)
                result_tab1_html = sarimax_results.summary().tables[1].as_html()
                result_tab1_pandas = pd.read_html(result_tab1_html, header=0, index_col=0)[0]  #module lxml have to be installed for html, pip install lxml

                fs = sarimax_results.summary()

                number_non_significant_param = 0
                for p_value in result_tab1_pandas["P>|z|"]:
                    if p_value != 0:
                        number_non_significant_param += 1

                result = [] # [number_non_significant_paramteres, Ljung-Box P_Value, Ljung-Box Coeficient, AIC, Sigma2, Jarque-Bera P-Value, Jarque-Bera Coeficient]
                result.append (number_non_significant_param) # Must be 0 --- Number of coefiecents of the parameters in the model which are not significant (p-value > 0.05)               
                result.append (float (fs.tables[2].data[1][1])) # Ljung-Box P_Value  (higher is better and at least Must be > 0.05) 
                result.append (float (fs.tables[2].data[0][1])) # Ljung-Box Coeficient  (higher is better)
                result.append (sarimax_results.aic) # AIC (Lower is better)
                result.append (result_tab1_pandas["coef"].iloc[-1]) # Sigma2 --- volatility of the model (lower is better)
                result.append (float (fs.tables[2].data[1][3])) # Jarque-Bera P_Value (should be < 0.05)
                result.append (float (fs.tables[2].data[0][3])) # Jarque-Bera Coeficient (higher is better)
                result.append (float (result_tab1_pandas["coef"].iloc[0])) # The slope of linear regression --- Coeficient of the exogenous variable 
                result.append (i)

                # Storing the results
                list_orders.append (order)
                list_results.append (result)
                
                i += 1


    # Choosing The Best Model
##    results_which_satisfy_the_least_conditions = [result for result in list_results if result[0] == 0 and result[1] > 0.05]
    results_which_satisfy_the_least_conditions = [result for result in list_results if result[0] == 0 and result[1] > 0.05 and result[3] < 5000]
    if len (results_which_satisfy_the_least_conditions) > 0:
        results_which_satisfy_the_least_conditions_sorted_based_on_SIGMA = sorted(results_which_satisfy_the_least_conditions, key = lambda f: f[4])
##        print ("results_which_satisfy_the_least_conditions_sorted_based_on_SIGMA : ",results_which_satisfy_the_least_conditions_sorted_based_on_SIGMA)
        best_order = list_orders[results_which_satisfy_the_least_conditions_sorted_based_on_SIGMA[0][8]]
##        print ("best_order : ", best_order)
        sarimax_model = SARIMAX(endog=y_train, exog=X_train_minus_intercept, order=best_order, enforce_stationarity=False)
        sarimax_results = sarimax_model.fit()
        preds = sarimax_results.predict(start=min(y_train.index), end=max(y_train.index))
##        print (preds)
        AutocorrelationValue_For_ARIMAX_Errors = durbin_watson(sarimax_results.resid)
##        print ("AutocorrelationValue_For_ARIMAX_Errors : ", AutocorrelationValue_For_ARIMAX_Errors)
##
##        print ("level at the begining : ", preds[0])
##        print ("level at the end : ", preds[-1])
##        print ("Slope of the linear regression : ", results_which_satisfy_the_least_conditions_sorted_based_on_SIGMA[0][7])

        Level_Start = preds[0]
        Level_End = preds[-1]
        Slope = results_which_satisfy_the_least_conditions_sorted_based_on_SIGMA[0][7]
        Significance_Status = 4
        

    else:
        Level_Start = 0
        Level_End = 0
        Slope = 0
        Significance_Status = 5

##    print ("===================")
##    print ("===================")
##    print ("===================")


    return Level_Start, Level_End, Slope, Significance_Status
#----------------------    
def slope_level_finder (df_pre, df_within, df_post, tweet_type):
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    
##    print ('df_pre.shape : ', df_pre.shape)
##    print ('df_within.shape : ', df_within.shape)
##    print ('df_post.shape : ', df_post.shape)

    Dataframe_Slices = [df_pre, df_within, df_post]
    Dataframe_Slices_Names = ["Pre_Interval", "Within_Interval", "Post_Interval"]

##    print ('df_pre.shape : ', df_pre.shape)
##    print ('non-zero values in quote : ', (df_pre['is_quote_status'] != 0).sum())
##    print ('df_pre : ', df_pre)

##    print ('df_within.shape : ', df_within.shape)
##    print ('non-zero values in quote : ', (df_within['is_quote_status'] != 0).sum())
##    print ('df_within : ', df_within)
##    
##    print ('df_post.shape : ', df_post.shape)
##    print ('non-zero values in quote : ', (df_post['is_quote_status'] != 0).sum())
##    print ('df_post : ', df_post)
    
    j = 0
    Levels_Slopes = [] # ['A1','A2','A3','A4','B1','B2','B3','B4','C1','C2','C3','C4'] ==> A1 = Level at start of pre interval, A2 = Leve at end of pre interval  A3 = Slope of pre interval, A4 = Significance Status of pre interval, B and C also have the same structure like A but for within interval and post interval respectively
    # A4 or B4 or C4 are coded like this:
    #    0 : Both coeficients in linear regression are not significant (however Durbin Watsion test value has been in the range [1.5, 2.5], that means there had been no Auto corelation in the results, but linear model was not a good fit and probably a regression with higher degrees should have been done
    #    1 : Only the coeficients of intercept (ie. level of start) in linear regression is significant (the same as above explanation about Durbin Watsone)
    #    2 : Only the coeficients of row_number (ie, Slope) in linear regression is significant (the same as above explanation about Durbin Watsone)
    #    3 : Both coeficients (row_number and intercept) in linear regression are significant (the same as above explanation about Durbin Watsone)
    #    4 : All Coeficients in ARIMAX are significant (Durbin Watson had been out of range [1.5, 2.5] so we had to go for ARIMAX instead of linear regression)
    #    5 : At least one of the coeficients in ARIMAX has been non-significant (which is most probably the coeficient of row-number) so we considered the whole model as non-significant (that's why if for example a A4 be 4 then A1,A2 and A3 also would be zero)
    for dfs in Dataframe_Slices:
##        print ("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
##        print ("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
##        print ("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
##        print (Dataframe_Slices_Names[j])
        if dfs.shape[0] > 0 and ((dfs[tweet_type] != 0).sum() > 7): # At least 7 non zero values must be exist (ie at least 3 days with tweet published in the corrosponding type
            expr = tweet_type + ' ~ ' + 'row_number' 
            y_train, X_train = dmatrices(expr, dfs, return_type='dataframe') # X_train would have an intercept column inside by this way
            olsr_results = linear_model.OLS(y_train, X_train).fit()
##            print ("olsr_results : ", olsr_results.summary())
##            print ("Intercept : ", olsr_results.params.Intercept)
##            print ("Slope : ", olsr_results.params.row_number)
            fs = olsr_results.summary()
            AutocorrelationValue_For_OLS_Errors = float (fs.tables[2].data[0][3]) # A Durbin_Watson Test
##            print ("AutocorrelationValue_For_OLS_Errors : ", AutocorrelationValue_For_OLS_Errors)
##            print ("sexxxx")
            preds = olsr_results.predict(X_train)
##            for i in preds:
##                print (i)
##            print (preds)

            if 1.5 < AutocorrelationValue_For_OLS_Errors < 2.5 :  # This is the acceptable range for not having autocorrelation
                Level_Start = preds[0]
                Level_End = preds[-1]
                Slope = olsr_results.params.row_number

                result_tab1_html = olsr_results.summary().tables[1].as_html()
                result_tab1_pandas = pd.read_html(result_tab1_html, header=0, index_col=0)[0]
##                print (result_tab1_pandas)
##                print (' result_tab1_pandas["P>|t|"].iloc[1] : ', result_tab1_pandas["P>|t|"].iloc[1])
##                print (' result_tab1_pandas["P>|t|"].iloc[0] : ', result_tab1_pandas["P>|t|"].iloc[0])
                if (result_tab1_pandas["P>|t|"].iloc[0] < 0.05) and (result_tab1_pandas["P>|t|"].iloc[1] < 0.05):
                    Significance_Status = 3
                elif result_tab1_pandas["P>|t|"].iloc[1] < 0.05 :
                    Significance_Status = 2
                elif result_tab1_pandas["P>|t|"].iloc[0] < 0.05 :
                    Significance_Status = 1
                else :
                    Significance_Status = 0
                
            else :
                X_train_minus_intercept = X_train.drop('Intercept', axis=1)
                X_train_minus_intercept = X_train_minus_intercept.asfreq('D')
                y_train = y_train.asfreq('D') 
                Level_Start, Level_End, Slope, Significance_Status = best_SARIMAX_model_estimation_finder (y_train, X_train_minus_intercept)
        else:
            Level_Start = 0 
            Level_End = 0
            Slope = 0
            Significance_Status = 0
            
        Levels_Slopes.extend ((Level_Start, Level_End, Slope, Significance_Status))
        j += 1


    return  Levels_Slopes
#----------------------
def Chunk_Chunk_Analysis (DF_Level_Slope_For_All_Users, grouper, Columns_List, i, tweet_type, address_output):
    j = 5000*(i)
    name_output_file = 'DF_Level_Slope_For_All_Users' + str(i) + '.csv'
    if j != 80000: # Early chunks contains 5k users each
        while j < 5000*(i+1):
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[j])
            Levels_Slopes = slope_level_finder (df_pre, df_within, df_post, tweet_type)
            for data in Levels_Slopes:
                DF_Level_Slope_For_All_Users.loc[j, Columns_List[Levels_Slopes.index(data)]] = data
            j += 1        
        DF_Level_Slope_For_All_Users.to_csv(address_output + name_output_file, index=False)
    else :   # the last chunk goes from user number 80k to the last user in the dataset (+86k)   
        while j < len(grouper):
            df_pre, df_within, df_post = Slice_Each_User_DataFrame_Into_Pre_Within_Post_Intervals (grouper[j])
            Levels_Slopes = slope_level_finder (df_pre, df_within, df_post, tweet_type)
            for data in Levels_Slopes:
                DF_Level_Slope_For_All_Users.loc[j, Columns_List[Levels_Slopes.index(data)]] = data
            j += 1            
        DF_Level_Slope_For_All_Users.to_csv(address_output + name_output_file, index=False)
##    return DF_Level_Slope_For_All_Users 
#----------------------    
def MultiProcessing (DF_Level_Slope_For_All_Users, grouper, Columns_List, tweet_type, address_output): #see if we should save each sub dataframe seperately
    i = 0
    Process_List = []
    while i < 17:
        Process_List.append (Process(target = Chunk_Chunk_Analysis   , args = (DF_Level_Slope_For_All_Users, grouper, Columns_List, i, tweet_type, address_output)))
        i += 1

    print (len (Process_List))
    i = 0
    while i < 17:
        Process_List[i].start()
        i += 1

    i = 0
    while i < 17:
        Process_List[i].join()
        i += 1    
#----------------------
def main():
    
##    f = open ('Micro.txt', 'w')

##    t0 = time.perf_counter()
##    f.write("------------------------------" + '\r\n')
##    f.write('Time 0 : ' + str(t0) + '\r\n')
    
    warnings.filterwarnings("ignore") # to supress warnings
    address_input = '/mnt/tb/amirdata/Merge_All_Waves.zip.hdf5'
    df = Desired_Data_Frame(address_input)
    df = Hybrid_to_Pure_Conversion (df)
    grouper = Make_Panda_DataFrame_For_Each_User (df)
    i = 0
    Columns_List = ['A1','A2','A3','A4','B1','B2','B3','B4','C1','C2','C3','C4']
##    print ("len(grouper) : ", len(grouper))
    DF_Level_Slope_For_All_Users = pd.DataFrame(0, index=np.arange(len(grouper)), columns=Columns_List) #making the zero dataframe so to be filled with the slope, level, .. of all users in the future

##    t1 = time.perf_counter()
##    f.write("------------------------------" + '\r\n')
##    f.write('Time 1 : ' + str(t1) + '\r\n')

    address = '/home/abodaghi/Twitter_Project/Data_Processing/Results/Slope_Level_Finder/'
    address_output_list = [address + 'Original/', address + 'Quote/', address + 'Reply/', address + 'Retweet/' ]
    tweet_type_list = ['original', 'is_quote_status', 'replyto_userid' ,'retweeted_userid']
##    address_output_list = [address + 'Original/', address + 'Quote/']
##    tweet_type_list = ['original', 'is_quote_status']
    k = 0
    while k < 4 :
        print ("========================")
        print ("Begining of the Stage : ", tweet_type_list[k])
        MultiProcessing (DF_Level_Slope_For_All_Users, grouper, Columns_List, tweet_type_list[k], address_output_list[k])
        print ("Ending of the Stage : ", tweet_type_list[k])
        print ("========================")
        k += 1
        
#----------------------
if __name__ == '__main__':
        main ()
