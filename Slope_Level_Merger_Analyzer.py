import csv
from pandas import read_csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#-----------------
def Merger (address_input, address_output):
    fout_names = ['Retweet_All_Users.csv','Reply_All_Users.csv','Quote_All_Users.csv','Original_All_Users.csv']
    yy = 0
    while yy < 4:
        fout = open (address_output + fout_names[yy],"w")
        with open(address_input[yy] + 'DF_Level_Slope_For_All_Users0.csv', 'r') as infile:
            reader = csv.DictReader(infile)
##            header_list = reader.fieldnames
##            fout.write(",".join(header_list))
            fout.write('A1,A2,A3,A4,B1,B2,B3,B4,C1,C2,C3,C4')
            fout.write("\n")
            number_users = len(list(reader))  
##            print (number_users)
        for i in range (0, 17):
            with open(address_input[yy] + "DF_Level_Slope_For_All_Users" + str(i) + ".csv") as fd:
                reader=csv.reader(fd)
                next(reader)
                if i != 16:
##                    print (i)
##                    print (len(list(reader)))
                    interestingrows=[row for idx, row in enumerate(reader) if idx in range ((i)*5000,(i+1)*5000)]
##                    print (interestingrows)
##                    print (len (interestingrows))
##                    print ("=============")
                    for line in interestingrows:
                        fout.write(",".join(line))
                        fout.write("\n")
                else:
##                    print (i)
##                    print (len(list(reader)))
                    interestingrows=[row for idx, row in enumerate(reader) if idx in range ((i)*5000,number_users)]
##                    print (interestingrows)
##                    print (len (interestingrows))
##                    print ("=============")
                    j = 0
                    for line in interestingrows:
                        fout.write(",".join(line))
                        if j != len (interestingrows) - 1:
                            fout.write("\n")
                        j += 1
            

        fout.close()
        yy += 1

    return fout_names
  
#-----------------
def func1 (x,y):
	if x != 0:
		return (y-x)/x   
	else:
		return 0
#-----------------
def func2 (x,y):
    return (y-x)  
#-----------------
def main ():
    address_input_retweet = '/home/abodaghi/Twitter_Project/Data_Processing/Results/Slope_Level_Finder/Retweet/'
    address_input_reply = '/home/abodaghi/Twitter_Project/Data_Processing/Results/Slope_Level_Finder/Reply/'
    address_input_quote = '/home/abodaghi/Twitter_Project/Data_Processing/Results/Slope_Level_Finder/Quote/'
    address_input_original = '/home/abodaghi/Twitter_Project/Data_Processing/Results/Slope_Level_Finder/Original/'
    address_input = [address_input_retweet, address_input_reply, address_input_quote, address_input_original]

    address_output = '/home/abodaghi/Twitter_Project/Data_Processing/Results/Slope_Level_Merger_Analyzer/'
    pd.set_option("display.max_rows", None, "display.max_columns", None)


    fout_names = Merger (address_input, address_output)

    fout_2 = open (address_output + "Final_Results.txt","w")
    ss = 0
    while ss < 4:
        df = read_csv(address_output + fout_names[ss], encoding="ISO-8859-1", engine='python')
##        print ("{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}")
##        print (fout_names[ss])
        fout_2.write("{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}")
        fout_2.write("\n")
        fout_2.write(fout_names[ss])
        fout_2.write("\n")

        df_Pre_Within_Comparison = df[(df['A4'].isin([2,3,4])) & (df['B4'].isin([2,3,4]))]  #great great and forget about the level its not applicable when we have SARIMAX model
        df_Within_Post_Comparison = df[(df['B4'].isin([2,3,4])) & (df['C4'].isin([2,3,4]))]  #great great and forget about the level its not applicable when we have SARIMAX model
        df_Pre_Within_Post_Comparison = df[(df['A4'].isin([2,3,4])) & (df['B4'].isin([2,3,4])) & (df['C4'].isin([2,3,4]))]  #great great and forget about the level its not applicable when we have SARIMAX model


        df_Pre_Within_Comparison = df_Pre_Within_Comparison.copy()
        df_Within_Post_Comparison = df_Within_Post_Comparison.copy()
        df_Pre_Within_Post_Comparison = df_Pre_Within_Post_Comparison.copy()


##        print ("df_Pre_Within_Comparison : ", str(df_Pre_Within_Comparison.shape))
##        print ("df_Within_Post_Comparison : ", str(df_Within_Post_Comparison.shape))
##        print ("df_Pre_Within_Post_Comparison : ", str(df_Pre_Within_Post_Comparison.shape))
        
        fout_2.write("df_Pre_Within_Comparison : " + str(df_Pre_Within_Comparison.shape))
        fout_2.write("\n")
        fout_2.write("df_Within_Post_Comparison : " + str(df_Within_Post_Comparison.shape))
        fout_2.write("\n")
        fout_2.write("df_Pre_Within_Post_Comparison : " + str(df_Pre_Within_Post_Comparison.shape))
        fout_2.write("\n")
        fout_2.write("++++++++++++++++++++++++++++++++++++++++++++")
        fout_2.write("\n")
        
        

        #-------------------------------------------------------------
        # Pre_Interval vs. Within_Interval Comparison 
        if  df_Pre_Within_Comparison.shape[0] > 0 :
        
            df_Pre_Within_Comparison['level_change_21'] = df_Pre_Within_Comparison.apply(lambda x: func2(x.A2, x.B1), axis=1)
            df_Pre_Within_Comparison['slope_change_21'] = df_Pre_Within_Comparison.apply(lambda x: func2(x.A3, x.B3), axis=1)

            
##            print ('df_Pre_Within_Comparison_level21_change_mean : ', df_Pre_Within_Comparison['level_change_21'].mean())            
##            print ('df_Pre_Within_Comparison_slope21_change_mean : ', df_Pre_Within_Comparison['slope_change_21'].mean())

            fout_2.write('df_Pre_Within_Comparison_level21_change_mean : ' + str (df_Pre_Within_Comparison['level_change_21'].mean()))
            fout_2.write("\n")
            fout_2.write('df_Pre_Within_Comparison_slope21_change_mean : ' + str (df_Pre_Within_Comparison['slope_change_21'].mean()))
            fout_2.write("\n")
            fout_2.write("++++++++++++++++++++++++++++++++++++++++++++")
            fout_2.write("\n")
            
##            print ("++++++++++++++++++++++")
        #-------------------------------------------------------------
        # Within_Interval vs. Post_Interval Comparison
        if  df_Within_Post_Comparison.shape[0] > 0 :
        
            df_Within_Post_Comparison['level_change_32'] = df_Within_Post_Comparison.apply(lambda x: func2(x.B2, x.C1), axis=1)
            df_Within_Post_Comparison['slope_change_32'] = df_Within_Post_Comparison.apply(lambda x: func2(x.B3, x.C3), axis=1)
            
##            print ('df_Within_Post_Comparison_level32_change_mean : ', df_Within_Post_Comparison['level_change_32'].mean())
##            print ('df_Within_Post_Comparison_slope32_change_mean : ', df_Within_Post_Comparison['slope_change_32'].mean())

            fout_2.write('df_Within_Post_Comparison_level32_change_mean : ' + str (df_Within_Post_Comparison['level_change_32'].mean()))
            fout_2.write("\n")
            fout_2.write('df_Within_Post_Comparison_slope32_change_mean : ' + str (df_Within_Post_Comparison['slope_change_32'].mean()))
            fout_2.write("\n")
            fout_2.write("++++++++++++++++++++++++++++++++++++++++++++")
            fout_2.write("\n")
            
##            print ("++++++++++++++++++++++")
        #-------------------------------------------------------------
        # Pre_Interval vs. Within_Interval vs. Post_Interval Comparison
        if  df_Pre_Within_Post_Comparison.shape[0] > 0 :
            df_Pre_Within_Post_Comparison['level_change_21'] = df_Pre_Within_Post_Comparison.apply(lambda x: func2(x.A2, x.B1), axis=1)
            df_Pre_Within_Post_Comparison['level_change_32'] = df_Pre_Within_Post_Comparison.apply(lambda x: func2(x.B2, x.C1), axis=1)
            df_Pre_Within_Post_Comparison['slope_change_21'] = df_Pre_Within_Post_Comparison.apply(lambda x: func2(x.A3, x.B3), axis=1)
            df_Pre_Within_Post_Comparison['slope_change_32'] = df_Pre_Within_Post_Comparison.apply(lambda x: func2(x.B3, x.C3), axis=1)
            
##            fig, ax = plt.subplots()
##            ax.boxplot(df_Pre_Within_Post_Comparison['level_change_32'].values)
##            plt.show()

##            print ('df_Pre_Within_Post_Comparison_level21_change_mean : ', df_Pre_Within_Post_Comparison['level_change_21'].mean())
##            print ('df_Pre_Within_Post_Comparison_level32_change_mean : ', df_Pre_Within_Post_Comparison['level_change_32'].mean())
##
##            print ('df_Pre_Within_Post_Comparison_slope21_change_mean : ', df_Pre_Within_Post_Comparison['slope_change_21'].mean())
##            print ('df_Pre_Within_Post_Comparison_slope32_change_mean : ', df_Pre_Within_Post_Comparison['slope_change_32'].mean())

            fout_2.write('df_Pre_Within_Post_Comparison_level21_change_mean : ' + str (df_Pre_Within_Post_Comparison['level_change_21'].mean()))
            fout_2.write("\n")
            fout_2.write('df_Pre_Within_Post_Comparison_level32_change_mean : ' + str (df_Pre_Within_Post_Comparison['level_change_32'].mean()))
            fout_2.write("\n")
            fout_2.write('df_Pre_Within_Post_Comparison_slope21_change_mean : ' + str (df_Pre_Within_Post_Comparison['slope_change_21'].mean()))
            fout_2.write("\n")
            fout_2.write('df_Pre_Within_Post_Comparison_slope32_change_mean : ' + str (df_Pre_Within_Post_Comparison['slope_change_32'].mean()))
            fout_2.write("\n")
            fout_2.write("++++++++++++++++++++++++++++++++++++++++++++")
            fout_2.write("\n")

##            print ("++++++++++++++++++++++")
        #-------------------------------------------------------------
            
        ss += 1  
#-----------------            
if __name__ == '__main__':
        main ()           
