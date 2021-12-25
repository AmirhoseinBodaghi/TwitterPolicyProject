import csv
from pandas import read_csv
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#-----------------------
def Merger(address_input, address_output):
    fout = open (address_output,"w")
    with open(address_input + 'User_Data0.csv', 'r') as infile:
        reader = csv.DictReader(infile)
        header_list = reader.fieldnames
        fout.write(",".join(header_list))
##                fout.write('A1,A2,A3,A4,B1,B2,B3,B4,C1,C2,C3,C4')
        fout.write("\n")
        number_users = len(list(reader))  
        print (number_users)
    for i in range (0, 17): 
        with open(address_input + 'User_Data' + str(i) + ".csv") as fd:
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
#---------------------------
def main():
    address_input = "/home/abodaghi/Twitter_Project/Data_Processing/Results/User_Data_Analysis/UD/"
    address_output = '/home/abodaghi/Twitter_Project/Data_Processing/Results/User_Data_Analysis/User_Data_File_Integrated/UDFI.csv'
    Merger (address_input, address_output)
#-----------------            
if __name__ == '__main__':
        main () 
