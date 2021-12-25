import vaex as vx
from nltk.sentiment import SentimentIntensityAnalyzer
from multiprocessing import Process, current_process, Queue
import nltk
import re
import pandas as pd 
#--------------------
def text_characteristics_show(text):

    #------
    text = str(text)
    if text:
        text_OK = ''.join([c for c in text if ord(c) < 128])
        text_OK_OK = re.sub(r"http\S+", "", text_OK)
        text_OK_OK = re.sub(r"<.*?>", "", text_OK_OK)
        text_OK_OK = re.sub(r"RT ", "", text_OK_OK)
        txt_letters_numbers = sum(c != ' ' for c in text_OK_OK)

        sa = SentimentIntensityAnalyzer()

        text_OK_OK_OK = re.sub(r"@\S+", "", text_OK_OK)
        text_OK_OK_OK_OK = re.sub(r"#\S+", "", text_OK_OK_OK)

        
        txt_letters_numbers_2 = sum(c != ' ' for c in text_OK_OK_OK_OK)

        if text_OK_OK:
                txt_sentiment_score = sa.polarity_scores(text_OK_OK)["compound"]
        else :
            txt_sentiment_score = 0

        if text_OK_OK_OK_OK:
            txt_sentiment_score_2 = sa.polarity_scores(text_OK_OK_OK_OK)["compound"]
        else :
            txt_sentiment_score_2 = 0
            
    else:
        txt_letters_numbers = 0
        txt_sentiment_score = 0
        txt_letters_numbers_2 = 0
        txt_sentiment_score_2 = 0
        
        text_OK_OK = "N/A"
        text_OK_OK_OK_OK = "N/A"

    print ("txt_letters_numbers : ", txt_letters_numbers)
    print ("txt_sentiment_score : ", txt_sentiment_score)
    print ("txt_letters_numbers_2 : ", txt_letters_numbers_2)
    print ("txt_sentiment_score_2 : ", txt_sentiment_score_2)
    print ("text_OK_OK : ", text_OK_OK)
    print ("text_OK_OK_OK_OK_OK : ", text_OK_OK_OK_OK)
###--------------------
##def number_letters_finders(text):
##    if text:
##        text_OK = ''.join([c for c in text if ord(c) < 128])
##        text_OK_OK = re.sub(r"http\S+", "", text_OK)
##        txt_letters_numbers = sum(c != ' ' for c in text_OK_OK)
##    else:
##        txt_letters_numbers = 0
##    return txt_letters_numbers
###--------------------
##def sentiment_score_finders(text):
##    if text:
##        text_OK = ''.join([c for c in text if ord(c) < 128])
##        text_OK_OK = re.sub(r"http\S+", "", text_OK)
##        sa = SentimentIntensityAnalyzer()
##        if text_OK_OK:
##            txt_sentiment_score = sa.polarity_scores(text_OK_OK)["compound"]
##        else:
##            txt_sentiment_score = 0
##    else:
##        txt_sentiment_score = 0
##    return txt_sentiment_score
###--------------------
##def number_letters_finders_without_mention_hashtag(text):
##    if text:
##        text_OK = ''.join([c for c in text if ord(c) < 128])
##        text_OK_OK = re.sub(r"http\S+", "", text_OK)
##        text_OK_OK_OK = re.sub(r"@\S+", "", text_OK_OK)
##        text_OK_OK_OK_OK = re.sub(r"#\S+", "", text_OK_OK_OK)
##        txt_letters_numbers = sum(c != ' ' for c in text_OK_OK_OK_OK)
##    else:
##        txt_letters_numbers = 0
##    return txt_letters_numbers
###--------------------
##def sentiment_score_finders_without_mention_hashtag(text):
##    if text:
##        text_OK = ''.join([c for c in text if ord(c) < 128])
##        text_OK_OK = re.sub(r"http\S+", "", text_OK)
##        text_OK_OK_OK = re.sub(r"@\S+", "", text_OK_OK)
##        text_OK_OK_OK_OK = re.sub(r"#\S+", "", text_OK_OK_OK)
##        sa = SentimentIntensityAnalyzer()
##        if text_OK_OK_OK_OK:
##            txt_sentiment_score = sa.polarity_scores(text_OK_OK_OK_OK)["compound"]
##        else:
##            txt_sentiment_score = 0
##    else:
##        txt_sentiment_score = 0
##    return txt_sentiment_score
#--------------------
def main():
    address_input_1 = '/mnt/tb/amirdata/Merge_All_Waves.zip.hdf5'
    df = vx.open (address_input_1)
##    df['txt']= df['txt'].astype('str')
##
##    # Adding text (without http) data
##    df['txt_letters_numbers']=df['txt'].apply(number_letters_finders)
##    df['txt_sentiment_score']=df['txt'].apply(sentiment_score_finders)
##
##    # Adding text (without http and mention and hashtag) data
##    df['txt_letters_numbers_without_HashtagMention']=df['txt'].apply(number_letters_finders_without_mention_hashtag)
##    df['txt_sentiment_score_without_HashtagMention']=df['txt'].apply(sentiment_score_finders_without_mention_hashtag)

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    i = 0
    while (i < df.shape[0]) :
##        if i > 300000000:
        print ("================================")
        print (i+300001000)
        text = df['txt'].values[i+300001000]
        retweet_count = df['retweet_count'].values[i+300001000]
        print (text)
        print (retweet_count)
        text_characteristics_show(text)
        i += 1
    

    
##    df.export_hdf5('/mnt/tb/amirdata/Merge_All_Waves_WithTextDataAnalysis.hdf5', progress=True)
#--------------------
if __name__ == '__main__':
        main ()
