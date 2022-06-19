import os
import re
import pandas as pd

from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def trans_scraper(base_url: str, query: str, wait: int, dataset_len: int) -> None:
    """ scrape youtube transcripts and save dataset into csv file """
    
    # initialize vars
    yt_vid_links = set()
    df = pd.DataFrame({'Title': [], 'Link': [], 'Text': []})

    # initialize chrome webdriver and search for videos
    options = webdriver.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument("window-size=1280,800")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36")

    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    driver.maximize_window()
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.get(f'{base_url}{query}')
    
    # get <dataset_len> number of video links from the search query
    while len(yt_vid_links) < dataset_len:
        html = driver.page_source
        page_yt_vid_links = set(re.findall(r'href=\"\/watch\?v=(.{11})', html))
        yt_vid_links = yt_vid_links.union(page_yt_vid_links)
        driver.execute_script("window.scrollBy(0,1000);")
        if 'no more results' in html.lower():
            break

    # scrape video title and transcripts for each video longer than 1 hour
    for link in tqdm(yt_vid_links):
        driver.get(f'https://www.youtube.com/watch?v={link}')
        html = driver.page_source

        try:
            WebDriverWait(driver, wait).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'h1>yt-formatted-string')))
        except:
            continue

        game = driver.find_element_by_css_selector('h1>yt-formatted-string').text
        duration = driver.find_element_by_css_selector('.ytp-time-duration').text
        if len(duration.split(':')) < 3:
            continue

        elem = driver.find_element_by_css_selector('#info>#menu-container>#menu>ytd-menu-renderer>yt-icon-button')
        driver.execute_script("arguments[0].click();", elem)
        
        try:
            WebDriverWait(driver, wait).until(EC.presence_of_element_located((By.CSS_SELECTOR, 'tp-yt-paper-listbox>ytd-menu-service-item-renderer')))
        except:
            continue
        
        elem = driver.find_element_by_css_selector('tp-yt-paper-listbox>ytd-menu-service-item-renderer')
        driver.execute_script("arguments[0].click();", elem)

        try:
            WebDriverWait(driver, wait).until(EC.presence_of_element_located((By.CLASS_NAME, 'ytd-transcript-segment-list-renderer')))
        except:
            continue

        transcript = driver.find_elements_by_css_selector('ytd-transcript-segment-renderer>div>yt-formatted-string')
        text = [elem.text for elem in transcript]

        df = df.append({'Title': game, 'Link': link, 'Text': ' '.join(text)}, ignore_index=True)

    # save dataset
    os.makedirs('dataset', exist_ok=True)
    df.to_csv(os.path.join('dataset','soccer_games.csv'), index=False)

    
if __name__ == "__main__":
    trans_scraper(
        'https://www.youtube.com/results?search_query=', # youtube serach url
        'soccer+full',  # youtube search query
        20, # time to wait for element
        1000 # videos to scrape data from
    )