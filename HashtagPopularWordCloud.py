from wordcloud import WordCloud
import matplotlib.pyplot as plt
#------------------------------------------
def main():
    output_address = "/home/abodaghi/Twitter_Project/Data_Processing/Results/HashtagPopularWordCloud/"
    list_political_hashtags = ['MAGA','Debates2020','Trump','BidenHarris2020','Election2020','Trump2020','VOTE','VPDebate','Vote','Biden','vote','JoeBiden','BlackLivesMatter','SCOTUS','Obamagate','BLM','InaugurationDay','BreonnaTaylor','ElectionDay','VoteHimOut','TrumpKnew','FBI','MoscowMitch','HunterBiden','AmyConeyBarrett','BidenHarris','TrumpVirus','Trump2020Landslide','ProudBoys','China','VoteEarly','VoteBidenHarris2020']
    list_political_hashtags_frequency = ['64719','61733','46441','39230','36793','35814','35026','28608','27991','23247','21726','19845','16633','16055','15852','14961','14950','14906','14597','13563','13455','13123','12772','12673','12092','11417','10629','10500','10127','10122','9941','9711']

    # ------ Text Creation ----------------
    text = ''
    for political_hashtag in list_political_hashtags :
                i = 0
                political_hashtag_set_of_repeatition = ''
                while i < int(list_political_hashtags_frequency [list_political_hashtags.index(political_hashtag)]):
                            political_hashtag_with_space = political_hashtag + ' '
                            political_hashtag_set_of_repeatition += political_hashtag_with_space
                            i += 1
                text += political_hashtag_set_of_repeatition
                            
    # ------ Word Cloud Plot ----------------
    wordcloud = WordCloud(width = 400, height = 800, collocations=False, background_color ='white', min_font_size = 10).generate(text)
    plt.figure(figsize = (4, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    plt.savefig(output_address + "WordCloud.tif", dpi=500)
    plt.close()

#------------------------------------------
if __name__ == '__main__':
        main ()
