import os
import random as rnd

from utils.soccer_scraper import soccer_scraper
from utils.process_data import process_data
from utils.trans_scraper import trans_scraper
from utils.model_pipeline import model_pipeline

rnd.seed(33)

class SoccerNER:
    def __init__(self):
        self.args_step1 = (
            'https://www.youtube.com/results?search_query=', # youtube serach url
            'soccer+full',  # youtube search query
            20, # time to wait for element
            1000 # videos to scrape data from
        )

        self.args_step2 = (
            'https://sofifa.com/players?offset=', # players url
            'https://en.wikipedia.org/wiki/List_of_FIFA_international_referees', # referees url
            'https://sofifa.com/teams?offset=', # teams url
            'https://sofifa.com', # managers url
            'https://en.wikipedia.org/wiki/List_of_association_football_stadiums_by_capacity' # stadiums url
        )

        self.args_step3 = (
            'dataset', # dataset path
            'soccer_games.csv', # soccer games transcripts
            os.path.join('utils', 'en_words.txt'), # english vocabulary list
            os.path.join('utils', 'en_stopwords.txt'), # english stopwords list
            'words.txt', # current dataset vocabulary
            'tags.txt', # current dataset NER tags
            50 # embedding size (words per sentence)
        )

        self.args_step4 = (
            64, # data generator batch size
            100, # model train steps
        )

    def run_trans_scraper(self, *args):
        """ get soccer games transcripts """
        trans_scraper(*args)
    
    def run_soccer_scraper(self, *args):
        """ get data about soccer players, manager, teams, referees and stadiums """
        soccer_scraper(*args)
    
    def run_process_data(self, *args):
        """ process all the dataset to feed the nlp model """
        process_data(*args)
    
    def run_train_model(self, *args):
        """ create and train an NER LSTM model """
        model_pipeline(*args)


if __name__ == '__main__':
    pipeline = SoccerNER()
    #pipeline.run_trans_scraper(*pipeline.args_step1)
    #pipeline.run_soccer_scraper(*pipeline.args_step2)
    #pipeline.run_process_data(*pipeline.args_step3)
    pipeline.run_train_model(*pipeline.args_step4)