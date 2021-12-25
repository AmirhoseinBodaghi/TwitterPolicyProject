# READ AND PLOT Results
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import os
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime, timedelta
#--------------------------------------------------------------------------
def finder_max_fall_rise_in_an_interval (loaded_list,file_name_input,OutputText):
    paired_difference_all = []
    for tweettype in loaded_list:
        i = 0
        paired_difference = []
        while i < (len(tweettype)-1):
            paired_difference.append (tweettype[i+1] - tweettype [i])
            i += 1
        paired_difference_all.append (paired_difference)
        
    OutputText.write ("+++++++++++++++++++++++++++" + "\n")
    OutputText.write ("+++++++++++++++++++++++++++" + "\n")
##    print ("+++++++++++++++++++++++++++")
##    print ("+++++++++++++++++++++++++++")
##    print (file_name_input[:-8])
    kk = 0
    for tweettype_difference in paired_difference_all:
        OutputText.write ("============" + "\n")
##        print ("============")
        OutputText.write ("Number of days in interval : " + str(len(loaded_list[0])) + "\n")
        OutputText.write ("Number of paired difference : " + str(len(tweettype_difference)) + "\n")  #which is one less than the number of days 
##        print ("Number of days in interval : ", len(loaded_list[0]))
##        print ("Number of paired difference : ", len(tweettype_difference))  #which is one less than the number of days      
        if kk == 0:
            OutputText.write ("Original" + "\n")
##            print ("Original")
        elif kk == 1:
            OutputText.write ("Quote" + "\n")
##            print ("Quote")
        elif kk == 2:
            OutputText.write ("Reply" + "\n")
##            print ("Reply")
        else:
            OutputText.write ("Retweet" + "\n")
##            print ("Retweet")
        kk +=1
            
        OutputText.write ("Max fall value: " + str(min(tweettype_difference)) + "\n")
        OutputText.write ("Max fall index: " + str(tweettype_difference.index (min(tweettype_difference))) + "\n")
##        print ("Max fall value: ", min(tweettype_difference))
##        print ("Max fall index: ", tweettype_difference.index (min(tweettype_difference)))
        adding_day_to_the_origin = tweettype_difference.index (min(tweettype_difference)) + 1 
        if file_name_input[:-8] == "Pre":
            origin = datetime(2019,10,9)
        elif file_name_input[:-8] == "Within":
            origin = datetime(2020,10,9)
        else:
            origin = datetime(2020,12,16)
        Max_Fall_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Fall_Date would be the date in which maximum amount of fall reletive to the previous day in whole interval has happened
        OutputText.write ("Max fall date: " + str(Max_Fall_Date.year) + '-' + str(Max_Fall_Date.month) + '-' + str(Max_Fall_Date.day) + "\n")
##        print ("Max fall date: ", str(Max_Fall_Date.year) + '-' + str(Max_Fall_Date.month) + '-' + str(Max_Fall_Date.day))

        OutputText.write ("+=+=+=+=+=+=+=+=+=+=+=+=+=+=+" + "\n")
##        print ("+=+=+=+=+=+=+=+=+=+=+=+=+=+=+")

        OutputText.write ("Max rise value: " + str(max(tweettype_difference)) + "\n")
        OutputText.write ("Max rise index: " + str(tweettype_difference.index (max(tweettype_difference))) + "\n")
##        print ("Max rise value: ", max(tweettype_difference))
##        print ("Max rise index: ", tweettype_difference.index (max(tweettype_difference)))
        adding_day_to_the_origin = tweettype_difference.index (max(tweettype_difference)) + 1 
        if file_name_input[:-8] == "Pre":
            origin = datetime(2019,10,9)
        elif file_name_input[:-8] == "Within":
            origin = datetime(2020,10,9)
        else:
            origin = datetime(2020,12,16)
        Max_Rise_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Fall_Date would be the date in which maximum amount of fall reletive to the previous day in whole interval has happened
        OutputText.write ("Max rise date: " + str(Max_Rise_Date.year) + '-' + str(Max_Rise_Date.month) + '-' + str(Max_Rise_Date.day) + "\n")
##        print ("Max rise date: ", str(Max_Rise_Date.year) + '-' + str(Max_Rise_Date.month) + '-' + str(Max_Rise_Date.day))

##    kk = 0
##    for tweettype in loaded_list_number_nonzero:
##        if kk == 0:
##            print ("Original")
##        elif kk == 1:
##            print ("Quote")
##        elif kk == 2:
##            print ("Reply")
##        else:
##            print ("Retweet")
##        kk +=1
##
##        ii = 0
##        for day_mean in tweettype:
##            print ("=======")
##            print (ii)
##            print (day_mean)
##            ii += 1
    return OutputText 
#--------------------------------------------------------------------------
def visualization (file_name_input, input_address, output_address, OutputText):
    open_file = open(input_address + file_name_input, "rb")
    loaded_list = pickle.load(open_file)
    loaded_list_number_nonzero = []
    loaded_list_mean = []
    for tweettype in loaded_list:
        loaded_list_number_nonzero.append ([np.count_nonzero (day_set) for day_set in tweettype]) #Active Users (Having at least 1 tweet in each type means active in that type)
        loaded_list_mean.append ([np.mean(day_set) for day_set in tweettype]) #Average Number Tweets (Average number of daily tweets in each type in each day)

    
    #Max of Fall and Rise in the Interval
    OutputText.write ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" + "\n")
    OutputText.write (file_name_input[:-8] + "\n")
    OutputText.write ("================================" + "\n")
    OutputText.write ("Number Active Users" + "\n")
    OutputText.write ("================================" + "\n")
    OutputText = finder_max_fall_rise_in_an_interval (loaded_list_number_nonzero,file_name_input,OutputText)

    OutputText.write ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" + "\n")
    OutputText.write (file_name_input[:-8] + "\n")
    OutputText.write ("================================" + "\n")
    OutputText.write ("Number Tweets" + "\n")
    OutputText.write ("================================" + "\n")
    OutputText = finder_max_fall_rise_in_an_interval (loaded_list_mean,file_name_input,OutputText)

##    print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
##    print (file_name_input[:-8])
##    print ("================================")
##    print ("Number Active Users")
##    print ("================================")    
##    finder_max_fall_rise_in_an_interval (loaded_list_number_nonzero,file_name_input)
##    print ("================================")
##    print ("Number Tweets")
##    print ("================================")    
##    finder_max_fall_rise_in_an_interval (loaded_list_mean,file_name_input)
##    print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    #Visualization Active Users
    output_address_active_users = output_address + '/' + 'active_users' + '/'
    if not os.path.exists(output_address_active_users):
        os.makedirs(output_address_active_users)
        
    interval = file_name_input [:-8]
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_number_nonzero[0])
    yticks = np.arange(0, 50000, step=5000) #these numbers such as 45000 and step 5000 are found after trials and see the results on the figures
    xticks = np.arange(0, 370, step=30) #these numbers such as 370 and step 30 are found after trials and see the results on the figures
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address_active_users + "Original_Active_Users_" + interval + ".tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_number_nonzero[1])
    yticks = np.arange(0, 35000, step=5000)
    xticks = np.arange(0, 370, step=30)
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address_active_users + "Quote_Active_Users_" + interval + ".tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_number_nonzero[2])
    yticks = np.arange(0, 46000, step=5000)
    xticks = np.arange(0, 370, step=30)
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address_active_users + "Reply_Active_Users_" + interval + ".tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_number_nonzero[3])
    yticks = np.arange(0, 56000, step=5000)
    xticks = np.arange(0, 370, step=30)
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address_active_users + "Retweet_Active_Users_" + interval + ".tif", dpi=150)
    plt.close()

    #Visualization Average Number of Tweets
    output_address_average_number_tweets = output_address + '/' + 'average_number_tweets' + '/'
    if not os.path.exists(output_address_average_number_tweets):
        os.makedirs(output_address_average_number_tweets)
        
    interval = file_name_input [:-8]
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[0])
    yticks = np.arange(0, 5, step=0.5)
    xticks = np.arange(0, 370, step=30)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(output_address_average_number_tweets + "Original_Average_Number_Tweet_" + interval + ".tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[1])
    yticks = np.arange(0, 3, step=0.5)
    xticks = np.arange(0, 370, step=30)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(output_address_average_number_tweets + "Quote_Average_Number_Tweet_" + interval + ".tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[2])
    yticks = np.arange(0, 8, step=1)
    xticks = np.arange(0, 370, step=30)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(output_address_average_number_tweets + "Reply_Average_Number_Tweet_" + interval + ".tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[3])
    yticks = np.arange(0, 13, step=1)
    xticks = np.arange(0, 370, step=30)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(output_address_average_number_tweets + "Retweet_Average_Number_Tweet_" + interval + ".tif", dpi=150)
    plt.close()

    return OutputText
#--------------------------------------------------------------------------
def main():

    input_address = "/home/abodaghi/Twitter_Project/Data_Processing/Results/Visualization_New_all_1/"
    output_address = "/home/abodaghi/Twitter_Project/Data_Processing/Results/Visualization_NewNew_all_2/"

    file_name_input_list = ["Pre_All.pkl", "Within_All.pkl", "Post_All.pkl"]

    if not os.path.exists(output_address):
        os.makedirs(output_address)
    OutputText = open(output_address + '/' + 'OutputText.txt',"w")

    for file_name_input in file_name_input_list:
        OutputText = visualization (file_name_input, input_address, output_address, OutputText)
#----------------------
if __name__ == '__main__':
        main ()
