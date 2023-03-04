from sentence_transformers import SentenceTransformer, util
from torch.multiprocessing import set_start_method
import pandas as pd
import numpy as np
import torch

set_start_method("spawn", force=True)

EM = None


def get_embedder(path: str):
    global EM
    if EM is None:
        EM = SentenceTransformer.load(path)
    if torch.has_mps:
        EM.to('mps')
    return EM


def get_n_similar_user(
        input_user_id: int,
        review_pool: pd.DataFrame,
        embedder_path: str,
        n: int,
) -> pd.DataFrame:
    """
    :param input_user_id:
    :param review_pool: containing input_user's and other user's reviews
    :param embedder_path:
    :param n:
    :return: Pandas DataFrame; columns: [user_id, similarity]
    """
    user_list = []
    similarity_list = []
    embedder: SentenceTransformer = get_embedder(embedder_path)

    candidates = review_pool['user_id'].unique()

    if len(candidates) > n:
        candidates = np.random.RandomState(26).choice(candidates, size=n)

    input_user_notes = review_pool[review_pool['user_id'] == input_user_id]['note'].tolist()
    input_user_embedding = embedder.encode(input_user_notes, convert_to_tensor=True)

    for candidate in candidates:
        candidate_notes: list = review_pool[review_pool['user_id'] == candidate]['note'].tolist()
        candidate_embedding = embedder.encode(candidate_notes, convert_to_tensor=True)
        similarity = util.cos_sim(a=input_user_embedding, b=candidate_embedding).mean()
        user_list.append(candidate)
        similarity_list.append(np.round(float(similarity.mean()), decimals=6))
    similar_user = pd.DataFrame(
        {'user_id': user_list, 'similarity': similarity_list}
    )
    similar_user.sort_values(by='similarity', ascending=False, inplace=True)
    return similar_user
