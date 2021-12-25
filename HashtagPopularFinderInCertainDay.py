from collections import Counter
import xlsxwriter
import pickle
import os
import warnings
#----------------------
def most_popular_hashtags (address_input,address_output,number_most_popular_hashtags):
    HashtagDayAll = []

    input_list = ['0.pkl','1.pkl','2.pkl','3.pkl','4.pkl','5.pkl','6.pkl','7.pkl','8.pkl','9.pkl','10.pkl','11.pkl','12.pkl','13.pkl','14.pkl','15.pkl','16.pkl']
    for file in input_list:
        open_file = open(address_input + file, "rb")
        loaded_list = pickle.load(open_file)  # loaded_list = [[HashtagDayAll]] | for example :  loaded_list = [['EOTID', 'NoCutsTillVaccine', 'SanFrancisco']]
        HashtagDayAll += loaded_list[0]

        
    HashtagAll = HashtagDayAll 
    HashtagAllCount = Counter (HashtagAll)
    HashtagAllCountSorted = sorted(HashtagAllCount.items(), key=lambda c: c[1])
    print ("Number of All Hashtags : ", len(HashtagAllCountSorted))
    MostPopularHashtags = HashtagAllCountSorted[-number_most_popular_hashtags:]
    MostPopularHashtags.reverse()

    if not os.path.exists(address_output):
        os.makedirs(address_output)
    workbook = xlsxwriter.Workbook(address_output + 'MostPopularHashtags' + '.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.set_column(0, 0, 10)
    worksheet.set_column(1, 1, 20)
    worksheet.set_column(2, 2, 10)
    worksheet.set_column(3, 3, 20)
    worksheet.set_column(4, 4, 30)

    worksheet.write(0, 0, 'Rank') 
    worksheet.write(0, 1, 'Hashtag')
    worksheet.write(0, 2, 'Annotation')
    worksheet.write(0, 3, '#Frequency in Total') 
    worksheet.write(0, 4, '#Frequency in The Certain Day') 

    worksheet.write(0, 5, 'Description')

    row = 1
    for hashtag in MostPopularHashtags:
        worksheet.write(row, 0,  row)
        worksheet.write(row, 1,  MostPopularHashtags[row-1][0])
        worksheet.write(row, 3,  MostPopularHashtags[row-1][1])
        worksheet.write(row, 4,  HashtagDayAll.count (MostPopularHashtags[row-1][0]))

        row += 1

    workbook.close ()
#----------------------
def main():        
    warnings.filterwarnings("ignore") # to supress warnings
    address_input = '/home/abodaghi/Twitter_Project/Data_Processing/Results/HashtagExtractInCertainDay/'
    address_output = '/home/abodaghi/Twitter_Project/Data_Processing/Results/HashtagMostPopularInCertainDay/'
    number_most_popular_hashtags = 100
    most_popular_hashtags (address_input,address_output,number_most_popular_hashtags)
#----------------------
if __name__ == '__main__':
        main ()
