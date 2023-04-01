from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
import numpy as np
import requests
import torch
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.3 Safari/605.1.15'

embeddings_dict: dict = torch.load(
    '/Users/leonbecker/DataspellProjects/master_thesis_ai_sommelier/database/test_embeddings_likes.pt')
embeddings_list: torch.tensor = torch.stack([embeddings_dict[i][1] for i in range(len(embeddings_dict.values()))])
wine_id_list: list = [embeddings_dict[i][0] for i in range(len(embeddings_dict.values()))]
del embeddings_dict
embedder = SentenceTransformer.load(
    '/Users/leonbecker/DataspellProjects/master_thesis_ai_sommelier/models/simcse_en')


def get_wine_data(highlights: json, reviews: json, prices: json, html: BeautifulSoup, wine_id: int,
                  probability: float) -> dict:
    no_value = '*k.A.*'

    if reviews['reviews'][0]['vintage']['wine']:
        try:
            name = reviews['reviews'][0]['vintage']['wine']['name']
        except Exception as e:
            print(e)
            name = no_value

        try:
            region = reviews['reviews'][0]['vintage']['wine']['region']['name']
        except Exception as e:
            print(e)
            region = no_value

        try:
            winery = reviews['reviews'][0]['vintage']['wine']['winery']['name']
        except Exception as e:
            print(e)
            winery = no_value
    else:
        name = no_value
        region = no_value
        winery = no_value

    if prices['checkout_prices'][0]['availability']['price']:
        try:
            price = str(np.round(prices['checkout_prices'][0]['availability']['price']['amount'],decimals=2)) + '€'
        except Exception as e:
            print(e)
            price = no_value
    elif prices['checkout_prices'][0]['availability']['median']:
        try:
            price = str(np.round(prices['checkout_prices'][0]['availability']['median']['amount'], decimals=2)) + '€'
        except Exception as e:
            print(e)
            price = no_value
    else:
        price = no_value

    if highlights['highlights'][0]['metadata']['style']:
        try:
            wine_type = highlights['highlights'][0]['metadata']['style']['name']
        except Exception as e:
            print(e)
            wine_type = no_value

        try:
            country = highlights['highlights'][0]['metadata']['style']['country']['name']
        except Exception as e:
            print(e)
            country = no_value
    else:
        wine_type = no_value
        country = no_value

    if highlights['highlights'][0]['metadata']['style']:
        grape_list = list()
        try:
            for i in range(len(highlights['highlights'][0]['metadata']['style']['grapes'])):
                grape_list.append(highlights['highlights'][0]['metadata']['style']['grapes'][i]['name'])
            grapes = ', '.join(grape_list)
        except Exception as e:
            print(e)
            grapes = no_value
    else:
        grapes = no_value

    wine_data = {
        'Id': str(wine_id),
        'Probability': "{0:.0%}".format(probability),
        'Name': name,
        'Wine Type': wine_type,
        'Country': country,
        'Region': region,
        'Winery': winery,
        'Main grapes': grapes,
        'Image path': 'https:' + html.find_all('img', {'class': 'image'})[0]['src'],
        'URL': 'https://vivino.com/w/{}'.format(wine_id),
        'Price': price
    }
    return wine_data


def request_vivino(wine_id: int):
    highlights = requests.get(
        url='https://www.vivino.com/api/wines/{}/highlights?per_page=1'.format(wine_id),
        headers={
            'User-Agent': USER_AGENT
        }
    )
    highlights.raise_for_status()
    reviews = requests.get(
        url='https://www.vivino.com/api/wines/{}/reviews?per_page=1'.format(wine_id),
        headers={
            'User-Agent': USER_AGENT
        }
    )
    prices = requests.get(
        url='https://www.vivino.com/api/wines/{}/checkout_prices?language=de'.format(wine_id),
        headers={
            'User-Agent': USER_AGENT
        }
    )
    prices.raise_for_status()
    reviews.raise_for_status()
    html = requests.get(
        url='https://www.vivino.com/w/{}/'.format(wine_id),
        headers={
            'User-Agent': USER_AGENT
        }
    )
    return [highlights, reviews, prices, html]


def get_recommendations(query: str, n: int) -> list:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarity_scores = util.cos_sim(query_embedding, embeddings_list)
    top_results = torch.topk(similarity_scores, k=1400)
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
        data_list: list = request_vivino(wine_id)
        highlights: json = data_list[0].json()
        reviews: json = data_list[1].json()
        prices: json = data_list[2].json()
        html = BeautifulSoup(data_list[3].text, 'html.parser')
        recommendations_list.append(
            get_wine_data(highlights=highlights, reviews=reviews, html=html, prices=prices,
                          wine_id=wine_id, probability=wine_data[wine_id])
        )
    return recommendations_list
