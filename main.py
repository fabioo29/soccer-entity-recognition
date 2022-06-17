import os
import trax 
import shutil
import numpy as np
import random as rnd

from trax import layers as tl
from trax.supervised import training
from utils.soccer_scraper import soccer_scraper
from utils.process_data import process_data
from utils.trans_scraper import trans_scraper
from utils.model_pipeline import model_pipeline

rnd.seed(33)

class SoccerNER:
    def __init__(self):
        self.args = []
        self.batch_size = 64
        self.train_steps = 100
        self.vocab_size = None
        self.d_model = None

    def run_soccer_scraper(self, *args):
        soccer_scraper(*args)
        self.args = []
    
    def run_process_data(self, *args):
        process_data(*args)
        self.args = []
    
    def run_trans_scraper(self, *args):
        trans_scraper(*args)
        self.args = []
    
    def run_train_model(self, *args):
        model_pipeline(*args)
        self.args = []


if __name__ == '__main__':
    pipeline = SoccerNER()

    """
    pipeline.args.append('https://www.youtube.com/results?search_query=')
    pipeline.args.append('soccer+full')
    pipeline.args.append(20)
    pipeline.args.append(1000)
    pipeline.run_trans_scraper(*pipeline.args)
    
    pipeline.args.append('https://sofifa.com/players?offset=') # players
    pipeline.args.append('https://en.wikipedia.org/wiki/List_of_FIFA_international_referees') # referees
    pipeline.args.append('https://sofifa.com/teams?offset=') # teams
    pipeline.args.append('https://sofifa.com') # managers
    pipeline.args.append('https://en.wikipedia.org/wiki/List_of_association_football_stadiums_by_capacity') # stadiums
    pipeline.run_soccer_scraper(*pipeline.args)

    pipeline.args.append('dataset')
    pipeline.args.append('soccer_games.csv')
    pipeline.args.append(os.path.join('utils', 'english_vocab.txt'))
    pipeline.args.append('words.txt')
    pipeline.args.append('tags.txt')
    pipeline.args.append(50)
    pipeline.run_process_data(*pipeline.args)
    """
    pipeline.args.append(pipeline.batch_size)
    pipeline.args.append(pipeline.train_steps)
    pipeline.args.append(pipeline.vocab_size)
    pipeline.args.append(pipeline.d_model)
    pipeline.run_train_model(*pipeline.args)