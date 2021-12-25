from pandas import set_option
from pandas import read_csv
#==========================
def main():
    df = read_csv("//mnt//zhudata//CSV//tweets_csv//tweets_wave1_csv.zip", encoding = "ISO-8859-1", compression='zip', low_memory=False) #Wave1 as earlier detected to have duplicatein tweet id
##    df = read_csv("//mnt//zhudata//CSV//tweets_csv//tweets_wave8_csv.gz", encoding = "ISO-8859-1", compression='gzip') #Wave8 as earlier detected to have duplicatein tweet id
    set_option('display.max_colwidth',None) #To stop Pandas from showing truncated text
    df['tweetid'] = df['tweetid'].apply(lambda x: '{:,.0f}'.format(x)) #To stop Pandas from turning large numbers into exponentials (just for show and printing use)


    dup = df['tweetid'].duplicated().any() #gives True or False output
    print ("Duplicate in rows: ", dup)

    if dup:
        h = df.groupby(['tweetid'])
        size = h.size()
        sz = size[size>1]
        print (sz)
        szd = sz.to_dict()
        for key in szd:
            print ("==========" + " Tweetid : " + str(key) + " ==========")
            print ("id of users with same tweet id")
            print (df.loc[df['tweetid']==key]['id'])
            print ('+++++++++++++++++++++++')
            print ("text of tweets with same tweet id")
            print (df.loc[df['tweetid']==key]['txt'])
            print ('+++++++++++++++++++++++')
            print ("timestamp of tweets with same tweet id")
            print (df.loc[df['tweetid']==key]['created_at'])
            print ('+++++++++++++++++++++++')
#==========================
main()
input ("Press Enter to end the program!")
