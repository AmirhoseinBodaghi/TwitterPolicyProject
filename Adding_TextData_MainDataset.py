import vaex as vx
from nltk.sentiment import SentimentIntensityAnalyzer
from multiprocessing import Process, current_process, Queue
import nltk
import re
#--------------------
def number_letters_finders(text):
    if text:
        text = ''.join([c for c in text if ord(c) < 128]) #removing non ASCII letters
        text = re.sub(r"http\S+", "", text) #removing links 
        text = re.sub(r"<.*?>", "", text) #removing strings like <ab>..<f0> which are converted forms of emojies in the text
        text = re.sub(r"RT ", "", text) #removing RT which is the sign of retweet 
        txt_letters_numbers = sum(c != ' ' for c in text) 
    else:
        txt_letters_numbers = 0
    return txt_letters_numbers
#--------------------
def sentiment_score_finders(text):
    if text:
        text = ''.join([c for c in text if ord(c) < 128])
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"RT ", "", text)
        sa = SentimentIntensityAnalyzer()
        if text:
            txt_sentiment_score = sa.polarity_scores(text)["compound"]
        else:
            txt_sentiment_score = 0
    else:
        txt_sentiment_score = 0
    return txt_sentiment_score
#--------------------
def number_letters_finders_without_mention_hashtag(text):
    if text:
        text = ''.join([c for c in text if ord(c) < 128])
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"RT ", "", text)
        text = re.sub(r"@\S+", "", text) #removing mentions
        text = re.sub(r"#\S+", "", text) #removing hashtags
        txt_letters_numbers = sum(c != ' ' for c in text)
    else:
        txt_letters_numbers = 0
    return txt_letters_numbers
#--------------------
def sentiment_score_finders_without_mention_hashtag(text):
    if text:
        text = ''.join([c for c in text if ord(c) < 128])
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"RT ", "", text)
        text = re.sub(r"@\S+", "", text) 
        text = re.sub(r"#\S+", "", text) 
        sa = SentimentIntensityAnalyzer()
        if text:
            txt_sentiment_score = sa.polarity_scores(text)["compound"]
        else:
            txt_sentiment_score = 0
    else:
        txt_sentiment_score = 0
    return txt_sentiment_score
#--------------------
def main():
    address_input_1 = '/mnt/tb/amirdata/Merge_All_Waves.zip.hdf5'
    df = vx.open (address_input_1)
##    df['txt']= df['txt'].astype('str')

    # Adding text (without http) data
    df['txt_letters_numbers']=df['txt'].apply(number_letters_finders)
    df['txt_sentiment_score']=df['txt'].apply(sentiment_score_finders)

    # Adding text (without http and mention and hashtag) data
    df['txt_letters_numbers_without_HashtagMention']=df['txt'].apply(number_letters_finders_without_mention_hashtag)
    df['txt_sentiment_score_without_HashtagMention']=df['txt'].apply(sentiment_score_finders_without_mention_hashtag)
    
    df.export_hdf5('/mnt/tb/amirdata/Merge_All_Waves_WithTextDataAnalysis.hdf5', progress=True)
#--------------------
if __name__ == '__main__':
        main ()
