""" get players, referees, managers, teams, stadiums data """

import os
import re
import json
import requests
import pandas as pd
from tqdm import tqdm

def soccer_scraper(pla_url: str, ref_url: str, tea_url: str, man_url: str, sta_url: str) -> None:
    """ scrape soccer data from wikipedia and sofifa.com to help process the NER dataset """

    # initialize vars
    players, referees, managers, teams, stadiums = [], [], [], [], []

    # scrape football players
    for offset in tqdm(range(0, 20000, 60), desc='Players'):
        res = requests.get(f'{pla_url}{offset}')
        curr_players = re.findall(r'\/player\/\d*\/([\w-]*)\/', res.text)
        players += list(curr_players)
    players = [p.lower().replace('-', ' ') for p in players]
    players = set([f'{x.split()[0]} {x.split()[-1]}' if len(x) > 2 else x for x in players])

    # scrape football referees
    for offset in tqdm(range(1), desc='Referees'):
        res = requests.get(ref_url)
        referees = re.findall(r'(\w*\s\w*)\s\(', res.text)
        referees = list(referees)
        referees = [r.lower() for r in referees]
        referees = set([f'{x.split()[0]} {x.split()[-1]}' if len(x) > 2 else x for x in referees])

    # scrape football teams
    for offset in tqdm(range(0, 1000, 60), desc='Teams'):
        res = requests.get(f'{tea_url}{offset}')
        curr_teams = re.findall(r'"(\/team\/\d+\/.+\/)"', res.text)
        teams += list(curr_teams)
    teams = set(teams)
    
    # scrape football managers
    for team in tqdm(teams, desc='Managers'):
        res = requests.get(f'{man_url}{team}')
        curr_managers = re.findall(r'"\/coach\/\d+\/(.+)\/"', res.text)
        managers += list(curr_managers)
    managers = set(managers)

    # scrape football stadiums
    for offset in tqdm(range(1), desc='Stadiums'):
        stadiums = pd.read_html(sta_url)[2]['Stadium'].tolist()
        stadiums = set([str(s).lower().replace('-', ' ').replace(' ♦', '') for s in stadiums])

    # clear teams name
    teams = set([t.lower().split('/')[-2].replace('-', ' ') for t in teams])

    data = {
        'players': list(players),
        'referees': list(referees),
        'managers': list(managers),
        'teams': list(teams),
        'stadiums': list(stadiums)
    }

    # save data to json file
    with open(os.path.join('dataset','soccer_data.json'), 'w') as fp:
        json.dump(data, fp, ensure_ascii=False)

    # save NER labels to txt file (O: none, PLA: players, REF: referees, ...)
    with open(os.path.join('dataset', 'tags.txt'), 'w') as fp:
        fp.write('\n'.join(['O', 'PLA', 'REF', 'MAN', 'TEA', 'STA']))

    
if __name__ == "__main__":
    soccer_scraper(
        'https://sofifa.com/players?offset=', # players url
        'https://en.wikipedia.org/wiki/List_of_FIFA_international_referees', # referees url
        'https://sofifa.com/teams?offset=', # teams url
        'https://sofifa.com', # managers url
        'https://en.wikipedia.org/wiki/List_of_association_football_stadiums_by_capacity' # stadiums url
    )