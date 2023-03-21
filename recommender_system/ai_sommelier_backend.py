from selenium.webdriver.support import expected_conditions as ec
from sentence_transformers import SentenceTransformer, util
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup
import numpy as np
import requests
import torch
import re
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15'
PRICE_CLASS_NAME = 'purchaseAvailability__row--S-DoM purchaseAvailability__prices--1WNrU'
TIMEOUT = 1

options = Options()
options.add_argument('--headless')
options.add_argument(f'user-agent={USER_AGENT}')
DRIVER = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

embeddings_dict: dict = torch.load(
    '/Users/leonbecker/DataspellProjects/master_thesis_ai_sommelier/database/test_embeddings_likes.pt')
embeddings_list: torch.tensor = torch.stack([embeddings_dict[i][1] for i in range(len(embeddings_dict.values()))])
wine_id_list: list = [embeddings_dict[i][0] for i in range(len(embeddings_dict.values()))]
del embeddings_dict
embedder = SentenceTransformer.load(
    '/Users/leonbecker/DataspellProjects/master_thesis_ai_sommelier/models/simcse_en').to('mps')


def get_wine_types() -> dict:
    grapes_dict: dict = requests.get(
        url='https://www.vivino.com/api/grapes/',
        headers={
            'User-Agent': USER_AGENT
        }
    ).json()['grapes']
    return {item['id']: item for item in grapes_dict}


def get_wine_data(soup: BeautifulSoup, wine_id: int, probability: float) -> dict:
    wine_types = get_wine_types()
    try:
        a_list = soup.find_all('a', {'class': 'anchor_anchor__m8Qi-'})
        name = soup.find_all('span', {'class': 'vintage'})[0].text.replace('\n', '')
        wine_type = a_list[3].text
        country = a_list[0].text
        winery = a_list[2].text
        img_path = 'https:' + soup.find_all('img', {'class': 'image'})[0]['src']
        grapes_list = [str(s) for s in a_list if 'grape' in str(s)][0]
        matches = re.findall('grape_ids\[]=\d+', grapes_list)
        grapes = []
        for match in matches:
            grapes.append(wine_types[int(re.findall('\d+', match)[0])]['name'])

        if not soup.find_all('div', {'class': PRICE_CLASS_NAME}):
            price = 'N.A.'
        else:
            price = soup.find_all(
                'div', {'class': PRICE_CLASS_NAME})[0].text
        wine_data = {
            'Id': wine_id,
            'Probability': probability,
            'Name': name,
            'Wine Type': wine_type,
            'Country': country,
            'Winery': winery,
            'Main grapes': ', '.join(grapes),
            'Image path': img_path,
            'Price': price,
            'Match': 0
        }
    except Exception as e:
        raise e
    return wine_data


def request_vivino(wine_id: int):
    initial_response: requests = requests.get(
        url='https://www.vivino.com/w/{}'.format(wine_id),
        headers={
            'User-Agent': USER_AGENT,
            'Accept-Language': 'de-DE'
        }
    )
    initial_response.raise_for_status()
    main_url = initial_response.url

    DRIVER.get(main_url)
    try:
        element_present = ec.presence_of_element_located((By.CLASS_NAME, PRICE_CLASS_NAME))
        WebDriverWait(DRIVER, TIMEOUT).until(element_present)
    except TimeoutException:
        print('Timed out waiting for page to load (wine Id:' + str(wine_id) + ')')
    html_response = DRIVER.page_source
    return html_response


def get_recommendations(query: str, n: int) -> list:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarity_scores = util.cos_sim(query_embedding, embeddings_list)
    top_results = torch.topk(similarity_scores, k=300)
    top_similar_wines = [wine_id_list[i] for i in top_results[1][0]]
    recommendations_list: list = list()
    wine_data = dict()
    for i in range(n):
        id_var = max(set(top_similar_wines), key=top_similar_wines.count)
        # get probabilities values
        indices_max = [i for i, j in enumerate(top_similar_wines) if j == id_var]
        probabilities = [top_results[0][0][i] for i in indices_max]
        probability: float = np.round(np.mean(probabilities), decimals=2)

        top_similar_wines = [i for i in top_similar_wines if i != id_var]
        wine_data[id_var] = probability
    for wine_id in wine_data.keys():
        soup = BeautifulSoup(request_vivino(wine_id), 'html.parser')
        recommendations_list.append(
            get_wine_data(soup=soup, wine_id=wine_id, probability=wine_data[wine_id])
        )
    return recommendations_list
