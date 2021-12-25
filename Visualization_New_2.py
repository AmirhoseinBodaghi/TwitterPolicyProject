# READ AND PLOT Results
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import os
from matplotlib.ticker import FormatStrFormatter
#--------------------------------------------------------------------------
def visualization_PW(file_name_input, input_address, output_address):
    open_file = open(input_address + file_name_input, "rb")
    loaded_list = pickle.load(open_file)
    loaded_list_mean = []
    for day_set in loaded_list:
        loaded_list_mean.append ([np.mean(y) for y in day_set])
    
    folder_name_output = file_name_input[:-4]
    my_path_output = output_address + 'PW' + '/' + folder_name_output
    if not os.path.exists(my_path_output):
        os.makedirs(my_path_output)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[0])
    yticks = np.arange(0, 5, step=0.5)
    xticks = np.arange(0, 370, step=30)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(my_path_output + "/" + "Original.tif", dpi=150)
##    plt.show()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[1])
    yticks = np.arange(0, 3, step=0.5)
    xticks = np.arange(0, 370, step=30)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(my_path_output + "/" + "Quote.tif", dpi=150)
##    plt.show()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[2])
    yticks = np.arange(0, 8, step=1)
    xticks = np.arange(0, 370, step=30)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(my_path_output + "/" + "Reply.tif", dpi=150)
##    plt.show()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[3])
    yticks = np.arange(0, 13, step=1)
    xticks = np.arange(0, 370, step=30)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(my_path_output + "/" + "Retweet.tif", dpi=150)
#--------------------------------------------------------------------------
def visualization_WP(file_name_input, input_address, output_address):
    open_file = open(input_address + file_name_input, "rb")
    loaded_list = pickle.load(open_file)
    loaded_list_mean = []
    for day_set in loaded_list:
        loaded_list_mean.append ([np.mean(y) for y in day_set])
    
    folder_name_output = file_name_input[:-4]
    my_path_output = output_address + 'WP' + '/' + folder_name_output
    if not os.path.exists(my_path_output):
        os.makedirs(my_path_output)

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[0])
    yticks = np.arange(0, 6, step=0.5)
    xticks = np.arange(0, 70, step=10)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(my_path_output + "/" + "Original.tif", dpi=150)
##    plt.show()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[1])
    yticks = np.arange(0, 5, step=0.5)
    xticks = np.arange(0, 70, step=10)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(my_path_output + "/" + "Quote.tif", dpi=150)
##    plt.show()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[2])
    yticks = np.arange(0, 10, step=1)
    xticks = np.arange(0, 70, step=10)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(my_path_output + "/" + "Reply.tif", dpi=150)
##    plt.show()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(8, 6)
    plt.plot(loaded_list_mean[3])
    yticks = np.arange(0, 17, step=2)
    xticks = np.arange(0, 70, step=10)
    fmt = lambda x: "{:.2f}".format(x)
    plt.yticks([float(fmt(i)) for i in yticks], [float(fmt(i)) for i in yticks])
    plt.xticks(xticks, xticks)
    plt.savefig(my_path_output + "/" + "Retweet.tif", dpi=150)
#--------------------------------------------------------------------------
def main():

    input_address = "/home/abodaghi/Twitter_Project/Data_Processing/Results/Visualization_New_1/"
    output_address = "/home/abodaghi/Twitter_Project/Data_Processing/Results/Visualization_New_2/"

    file_name_input_list_PW = ["PW_P_All.pkl", "PW_W_All.pkl"]
    file_name_input_list_WP = ["WP_W_All.pkl", "WP_P_All.pkl"]

    for file_name_input in file_name_input_list_PW:
        visualization_PW(file_name_input, input_address, output_address)

    for file_name_input in file_name_input_list_WP:
        visualization_WP(file_name_input, input_address, output_address)
        
#----------------------
if __name__ == '__main__':
        main ()
