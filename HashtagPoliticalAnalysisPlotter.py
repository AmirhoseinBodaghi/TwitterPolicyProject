# READ AND PLOT Results
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import os
from matplotlib.ticker import FormatStrFormatter
from datetime import datetime, timedelta
#--------------------------------------------------------------------------
def Reading_Data(input_address):
    open_file = open(input_address + "0.pkl", "rb")
    loaded_list = pickle.load(open_file)
    lengh_pre_interval_days = len (loaded_list[0])
    lengh_within_interval_days = len (loaded_list[1])
    lengh_post_interval_days = len (loaded_list[2])
    Final_List = [[[] for i in range (0, lengh_pre_interval_days)], [[] for i in range (0, lengh_within_interval_days)], [[] for i in range (0, lengh_post_interval_days)]]
    open_file.close()
    for i in range(0,17):
        open_file = open(input_address + str(i) + ".pkl", "rb")
        loaded_list = pickle.load(open_file)

        j = 0
        while j < lengh_pre_interval_days:
            Final_List[0][j] += loaded_list[0][j]
            j+=1

        j = 0
        while j < lengh_within_interval_days:
            Final_List[1][j] += loaded_list[1][j]
            j+=1
        
        j = 0
        while j < lengh_post_interval_days:
            Final_List[2][j] += loaded_list[2][j]
            j+=1    

    return Final_List
#--------------------------------------------------------------------------
def Finder_Hashtag_Use_Statistics (Final_List):

    sum_political_hashtag_used_per_day_by_all_users = [[],[],[]]
    number_users_used_political_hashtag_per_day = [[],[],[]]
    number_users_did_not_use_any_political_hashtag_per_day = [[],[],[]]

    j = 0
    while j < len(Final_List[0]): #lengh_pre_interval_days
        sum_political_hashtag_used_per_day_by_all_users[0].append(sum(Final_List[0][j]))
        number_users_used_political_hashtag_per_day[0].append(np.count_nonzero(Final_List[0][j]))
        number_users_did_not_use_any_political_hashtag_per_day[0].append(Final_List[0][j].count(0))
        j+=1
    
    j = 0
    while j < len(Final_List[1]): #lengh_within_interval_days
        sum_political_hashtag_used_per_day_by_all_users[1].append(sum(Final_List[1][j]))
        number_users_used_political_hashtag_per_day[1].append(np.count_nonzero(Final_List[1][j]))
        number_users_did_not_use_any_political_hashtag_per_day[1].append(Final_List[1][j].count(0))
        j+=1

    j = 0
    while j < len(Final_List[2]): #lengh_post_interval_days
        sum_political_hashtag_used_per_day_by_all_users[2].append(sum(Final_List[2][j]))
        number_users_used_political_hashtag_per_day[2].append(np.count_nonzero(Final_List[2][j]))
        number_users_did_not_use_any_political_hashtag_per_day[2].append(Final_List[2][j].count(0))
        j+=1

    return sum_political_hashtag_used_per_day_by_all_users, number_users_used_political_hashtag_per_day, number_users_did_not_use_any_political_hashtag_per_day
#--------------------------------------------------------------------------
def Visualization (sum_political_hashtag_used_per_day_by_all_users, number_users_used_political_hashtag_per_day, number_users_did_not_use_any_political_hashtag_per_day, output_address):

    #--------------- Number of Times Political Hashtags Are Used Per Day --------------------------
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(sum_political_hashtag_used_per_day_by_all_users[0])
    yticks = np.arange(0, 50000, step=5000) #these numbers such as 45000 and step 5000 are found after trials and see the results on the figures
    xticks = np.arange(0, 77, step=7) #these numbers such as 370 and step 30 are found after trials and see the results on the figures
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address + "Sum_Political_Hashtag_Used_Per_Day_By_All_Users_Pre.tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(sum_political_hashtag_used_per_day_by_all_users[1])
    yticks = np.arange(0, 50000, step=5000) #these numbers such as 45000 and step 5000 are found after trials and see the results on the figures
    xticks = np.arange(0, 77, step=7) #these numbers such as 370 and step 30 are found after trials and see the results on the figures
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address + "Sum_Political_Hashtag_Used_Per_Day_By_All_Users_Within.tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(sum_political_hashtag_used_per_day_by_all_users[2])
    yticks = np.arange(0, 50000, step=5000) #these numbers such as 45000 and step 5000 are found after trials and see the results on the figures
    xticks = np.arange(0, 77, step=7) #these numbers such as 370 and step 30 are found after trials and see the results on the figures
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address + "Sum_Political_Hashtag_Used_Per_Day_By_All_Users_Post.tif", dpi=150)
##    plt.show()
    plt.close()
    #-----------------------------------------

    #---------------- Number Users Used Political Hashtags Per Day -------------------------
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(number_users_used_political_hashtag_per_day[0])
    yticks = np.arange(0, 15000, step=1500) #these numbers such as 45000 and step 5000 are found after trials and see the results on the figures
    xticks = np.arange(0, 77, step=7) #these numbers such as 370 and step 30 are found after trials and see the results on the figures
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address + "Number_Users_Used_Political_Hashtag_Per_Day_Pre.tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(number_users_used_political_hashtag_per_day[1])
    yticks = np.arange(0, 15000, step=1500) #these numbers such as 45000 and step 5000 are found after trials and see the results on the figures
    xticks = np.arange(0, 77, step=7) #these numbers such as 370 and step 30 are found after trials and see the results on the figures
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address + "Number_Users_Used_Political_Hashtag_Per_Day_Within.tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(number_users_used_political_hashtag_per_day[2])
    yticks = np.arange(0, 15000, step=1500) #these numbers such as 45000 and step 5000 are found after trials and see the results on the figures
    xticks = np.arange(0, 77, step=7) #these numbers such as 370 and step 30 are found after trials and see the results on the figures
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address + "Number_Users_Used_Political_Hashtag_Per_Day_Post.tif", dpi=150)
##    plt.show()
    plt.close()

    #---------------- Number Users Did not Use any Political Hashtags Per Day -------------------------
    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(number_users_did_not_use_any_political_hashtag_per_day[0])
    yticks = np.arange(0, 95000, step=5000) #these numbers such as 45000 and step 5000 are found after trials and see the results on the figures
    xticks = np.arange(0, 77, step=7) #these numbers such as 370 and step 30 are found after trials and see the results on the figures
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address + "Number_Users_Did_Not_Use_Any_Political_Hashtag_Per_Day_Pre.tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(number_users_did_not_use_any_political_hashtag_per_day[1])
    yticks = np.arange(0, 95000, step=5000) #these numbers such as 45000 and step 5000 are found after trials and see the results on the figures
    xticks = np.arange(0, 77, step=7) #these numbers such as 370 and step 30 are found after trials and see the results on the figures
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address + "Number_Users_Did_Not_Use_Any_Political_Hashtag_Per_Day_Within.tif", dpi=150)
##    plt.show()
    plt.close()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(number_users_did_not_use_any_political_hashtag_per_day[2])
    yticks = np.arange(0, 95000, step=5000) #these numbers such as 45000 and step 5000 are found after trials and see the results on the figures
    xticks = np.arange(0, 77, step=7) #these numbers such as 370 and step 30 are found after trials and see the results on the figures
    fmt = lambda x: "{:.2f}".format(x)
##    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.savefig(output_address + "Number_Users_Did_Not_Use_Any_Political_Hashtag_Per_Day_Post.tif", dpi=150)
##    plt.show()
    plt.close() 
#--------------------------------------------------------------------------
def finder_max_fall_rise_in_an_interval (sum_political_hashtag_used_per_day_by_all_users, number_users_used_political_hashtag_per_day, OutputText):

    paired_difference_all_sum = []
    for interval in sum_political_hashtag_used_per_day_by_all_users:
        i = 0
        paired_difference = []
        while i < (len(interval)-1):
            paired_difference.append (interval[i+1] - interval[i])
            i += 1
        paired_difference_all_sum.append (paired_difference)

    paired_difference_all_nu = []
    for interval in number_users_used_political_hashtag_per_day:
        i = 0
        paired_difference = []
        while i < (len(interval)-1):
            paired_difference.append (interval[i+1] - interval[i])
            i += 1
        paired_difference_all_nu.append (paired_difference)
        

    OutputText.write ("Number of days in Pre interval : " + str(len(sum_political_hashtag_used_per_day_by_all_users[0])) + "\n")
    OutputText.write ("Number of days in Within interval : " + str(len(sum_political_hashtag_used_per_day_by_all_users[1])) + "\n")
    OutputText.write ("Number of days in Post interval : " + str(len(sum_political_hashtag_used_per_day_by_all_users[2])) + "\n")
    
    OutputText.write ("Number of paired difference pre sm: " + str(len(paired_difference_all_sum[0])) + "\n")  #which is one less than the number of days
    OutputText.write ("Number of paired difference within sm: " + str(len(paired_difference_all_sum[1])) + "\n")  #which is one less than the number of days
    OutputText.write ("Number of paired difference post sm: " + str(len(paired_difference_all_sum[2])) + "\n")  #which is one less than the number of days

    OutputText.write ("Number of paired difference pre nu: " + str(len(paired_difference_all_nu[0])) + "\n")  #which is one less than the number of days
    OutputText.write ("Number of paired difference within nu: " + str(len(paired_difference_all_nu[1])) + "\n")  #which is one less than the number of days
    OutputText.write ("Number of paired difference post nu: " + str(len(paired_difference_all_nu[2])) + "\n")  #which is one less than the number of days
    
    kk = 0
    while kk < 3:
        if kk == 0:
            origin = datetime(2020,9,1)

            adding_day_to_the_origin = paired_difference_all_sum[0].index (min(paired_difference_all_sum[0])) + 1 
            Max_Fall_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Fall_Date would be the date in which maximum amount of fall reletive to the previous day in whole interval has happened
            OutputText.write ("Max fall date in Sum Pre: " + str(Max_Fall_Date.year) + '-' + str(Max_Fall_Date.month) + '-' + str(Max_Fall_Date.day) + "\n")
            OutputText.write ("Max fall value in Sum Pre: " + str(min(paired_difference_all_sum[0])) + "\n")
            OutputText.write ("Max fall index in Sum Pre: " + str(paired_difference_all_sum[0].index (min(paired_difference_all_sum[0]))) + "\n")

            adding_day_to_the_origin = paired_difference_all_sum[0].index (max(paired_difference_all_sum[0])) + 1 
            Max_Rise_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Rise_Date would be the date in which maximum amount of rise reletive to the previous day in whole interval has happened
            OutputText.write ("Max rise date in Sum Pre: " + str(Max_Rise_Date.year) + '-' + str(Max_Rise_Date.month) + '-' + str(Max_Rise_Date.day) + "\n")
            OutputText.write ("Max rise value in Sum Pre: " + str(max(paired_difference_all_sum[0])) + "\n")
            OutputText.write ("Max rise index in Sum Pre: " + str(paired_difference_all_sum[0].index (max(paired_difference_all_sum[0]))) + "\n")
        
            adding_day_to_the_origin = paired_difference_all_nu[0].index (min(paired_difference_all_nu[0])) + 1 
            Max_Fall_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Fall_Date would be the date in which maximum amount of fall reletive to the previous day in whole interval has happened
            OutputText.write ("Max fall date in Nu Pre: " + str(Max_Fall_Date.year) + '-' + str(Max_Fall_Date.month) + '-' + str(Max_Fall_Date.day) + "\n")
            OutputText.write ("Max fall value in Nu Pre: " + str(min(paired_difference_all_nu[0])) + "\n")
            OutputText.write ("Max fall index in Nu Pre: " + str(paired_difference_all_nu[0].index (min(paired_difference_all_nu[0]))) + "\n")

            adding_day_to_the_origin = paired_difference_all_nu[0].index (max(paired_difference_all_nu[0])) + 1 
            Max_Rise_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Rise_Date would be the date in which maximum amount of rise reletive to the previous day in whole interval has happened
            OutputText.write ("Max rise date in Nu Pre: " + str(Max_Rise_Date.year) + '-' + str(Max_Rise_Date.month) + '-' + str(Max_Rise_Date.day) + "\n")
            OutputText.write ("Max rise value in Nu Pre: " + str(max(paired_difference_all_nu[0])) + "\n")
            OutputText.write ("Max rise index in Nu Pre: " + str(paired_difference_all_nu[0].index (max(paired_difference_all_nu[0]))) + "\n")
            
        elif kk == 1:
            origin = datetime(2020,10,9)

            adding_day_to_the_origin = paired_difference_all_sum[1].index (min(paired_difference_all_sum[1])) + 1 
            Max_Fall_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Fall_Date would be the date in which maximum amount of fall reletive to the previous day in whole interval has happened
            OutputText.write ("Max fall date in Sum Within: " + str(Max_Fall_Date.year) + '-' + str(Max_Fall_Date.month) + '-' + str(Max_Fall_Date.day) + "\n")
            OutputText.write ("Max fall value in Sum Within: " + str(min(paired_difference_all_sum[1])) + "\n")
            OutputText.write ("Max fall index in Sum Within: " + str(paired_difference_all_sum[1].index (min(paired_difference_all_sum[1]))) + "\n")

            adding_day_to_the_origin = paired_difference_all_sum[1].index (max(paired_difference_all_sum[1])) + 1 
            Max_Rise_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Rise_Date would be the date in which maximum amount of rise reletive to the previous day in whole interval has happened
            OutputText.write ("Max rise date in Sum Within: " + str(Max_Rise_Date.year) + '-' + str(Max_Rise_Date.month) + '-' + str(Max_Rise_Date.day) + "\n")
            OutputText.write ("Max rise value in Sum Within: " + str(max(paired_difference_all_sum[1])) + "\n")
            OutputText.write ("Max rise index in Sum Within: " + str(paired_difference_all_sum[1].index (max(paired_difference_all_sum[1]))) + "\n")
        
            adding_day_to_the_origin = paired_difference_all_nu[1].index (min(paired_difference_all_nu[1])) + 1 
            Max_Fall_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Fall_Date would be the date in which maximum amount of fall reletive to the previous day in whole interval has happened
            OutputText.write ("Max fall date in Nu Within: " + str(Max_Fall_Date.year) + '-' + str(Max_Fall_Date.month) + '-' + str(Max_Fall_Date.day) + "\n")
            OutputText.write ("Max fall value in Nu Within: " + str(min(paired_difference_all_nu[1])) + "\n")
            OutputText.write ("Max fall index in Nu Within: " + str(paired_difference_all_nu[1].index (min(paired_difference_all_nu[1]))) + "\n")

            adding_day_to_the_origin = paired_difference_all_nu[1].index (max(paired_difference_all_nu[1])) + 1 
            Max_Rise_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Rise_Date would be the date in which maximum amount of rise reletive to the previous day in whole interval has happened
            OutputText.write ("Max rise date in Nu Within: " + str(Max_Rise_Date.year) + '-' + str(Max_Rise_Date.month) + '-' + str(Max_Rise_Date.day) + "\n")
            OutputText.write ("Max rise value in Nu Within: " + str(max(paired_difference_all_nu[1])) + "\n")
            OutputText.write ("Max rise index in Nu Within: " + str(paired_difference_all_nu[1].index (max(paired_difference_all_nu[1]))) + "\n")

        else :
            origin = datetime(2020,12,16)

            adding_day_to_the_origin = paired_difference_all_sum[2].index (min(paired_difference_all_sum[2])) + 1 
            Max_Fall_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Fall_Date would be the date in which maximum amount of fall reletive to the previous day in whole interval has happened
            OutputText.write ("Max fall date in Sum Post: " + str(Max_Fall_Date.year) + '-' + str(Max_Fall_Date.month) + '-' + str(Max_Fall_Date.day) + "\n")
            OutputText.write ("Max fall value in Sum Post: " + str(min(paired_difference_all_sum[2])) + "\n")
            OutputText.write ("Max fall index in Sum Post: " + str(paired_difference_all_sum[2].index (min(paired_difference_all_sum[2]))) + "\n")

            adding_day_to_the_origin = paired_difference_all_sum[2].index (max(paired_difference_all_sum[2])) + 1 
            Max_Rise_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Rise_Date would be the date in which maximum amount of rise reletive to the previous day in whole interval has happened
            OutputText.write ("Max rise date in Sum Post: " + str(Max_Rise_Date.year) + '-' + str(Max_Rise_Date.month) + '-' + str(Max_Rise_Date.day) + "\n")
            OutputText.write ("Max rise value in Sum Post: " + str(max(paired_difference_all_sum[2])) + "\n")
            OutputText.write ("Max rise index in Sum Post: " + str(paired_difference_all_sum[2].index (max(paired_difference_all_sum[2]))) + "\n")
        
            adding_day_to_the_origin = paired_difference_all_nu[2].index (min(paired_difference_all_nu[2])) + 1 
            Max_Fall_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Fall_Date would be the date in which maximum amount of fall reletive to the previous day in whole interval has happened
            OutputText.write ("Max fall date in Nu Post: " + str(Max_Fall_Date.year) + '-' + str(Max_Fall_Date.month) + '-' + str(Max_Fall_Date.day) + "\n")
            OutputText.write ("Max fall value in Nu Post: " + str(min(paired_difference_all_nu[2])) + "\n")
            OutputText.write ("Max fall index in Nu Post: " + str(paired_difference_all_nu[2].index (min(paired_difference_all_nu[2]))) + "\n")

            adding_day_to_the_origin = paired_difference_all_nu[2].index (max(paired_difference_all_nu[2])) + 1 
            Max_Rise_Date = origin + timedelta(days= adding_day_to_the_origin) #Max_Rise_Date would be the date in which maximum amount of rise reletive to the previous day in whole interval has happened
            OutputText.write ("Max rise date in Nu Post: " + str(Max_Rise_Date.year) + '-' + str(Max_Rise_Date.month) + '-' + str(Max_Rise_Date.day) + "\n")
            OutputText.write ("Max rise value in Nu Post: " + str(max(paired_difference_all_nu[2])) + "\n")
            OutputText.write ("Max rise index in Nu Post: " + str(paired_difference_all_nu[2].index (max(paired_difference_all_nu[2]))) + "\n")

        kk += 1
        
#--------------------------------------------------------------------------
def main():

    input_address = "/home/abodaghi/Twitter_Project/Data_Processing/Results/HashtagPoliticalAnalysis/"
    output_address = "/home/abodaghi/Twitter_Project/Data_Processing/Results/HashtagPoliticalAnalysisPlotter/"

    Final_List = Reading_Data(input_address)
    sum_political_hashtag_used_per_day_by_all_users, number_users_used_political_hashtag_per_day, number_users_did_not_use_any_political_hashtag_per_day = Finder_Hashtag_Use_Statistics (Final_List)
    Visualization (sum_political_hashtag_used_per_day_by_all_users, number_users_used_political_hashtag_per_day, number_users_did_not_use_any_political_hashtag_per_day, output_address)

    if not os.path.exists(output_address):
        os.makedirs(output_address)
    OutputText = open(output_address + 'OutputText.txt',"w")

    finder_max_fall_rise_in_an_interval (sum_political_hashtag_used_per_day_by_all_users, number_users_used_political_hashtag_per_day, OutputText)
#----------------------
if __name__ == '__main__':
        main ()
