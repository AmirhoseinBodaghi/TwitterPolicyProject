import rarfile
#==========================
def main():
    rar_file = rarfile.RarFile("//mnt//zhudata//CSV//users_csv//userInf_csv.rar")
    path_output = "//home//abodaghi//Twitter_Project//Data_Processing//Results//User_Data_Analysis//User_Data_File//"  #the address in which we want to save the extracted files from the rar_file
    rarfile.RarFile.extractall(rar_file , path = path_output)   
#==========================
main()
input ("Press Enter to end the program!")
