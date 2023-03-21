from sentence_transformers import SentenceTransformer, util
from torch.multiprocessing import set_start_method
import pandas as pd
import numpy as np
from torch import has_mps

set_start_method("spawn", force=True)

EM = None


def get_embedder(path: str):
    global EM
    if EM is None:
        EM = SentenceTransformer.load(path)
    if has_mps:
        EM.to('mps')
    return EM


def get_n_similar_user(
        input_user_id: int,
        review_pool: pd.DataFrame,
        embedder_path: str,
        truncate: bool,
        n: int,
) -> pd.DataFrame:
    """
    :param truncate:
    :param input_user_id:
    :param review_pool: containing input_user's and other user's reviews
    :param embedder_path:
    :param n:
    :return: Pandas DataFrame; columns: [user_id, similarity]
    """
    user_list = []
    similarity_list = []
    embedder: SentenceTransformer = get_embedder(embedder_path)

    candidates = review_pool[review_pool['user_id'] != input_user_id]['user_id'].unique()

    if truncate and len(candidates) > n:
        candidates = np.random.RandomState(26).choice(candidates, size=n)

    input_user_notes = review_pool[review_pool['user_id'] == input_user_id].sort_values(by='wine_id')['note'].tolist()
    try:
        input_user_embedding = embedder.encode(input_user_notes, convert_to_tensor=True)
    except Exception as e:
        print(e)

    for candidate_id in candidates:
        candidate_notes: list = review_pool[review_pool['user_id'] == candidate_id
                                            ].sort_values(by='wine_id')['note'].tolist()
        try:
            candidate_embedding = embedder.encode(candidate_notes, convert_to_tensor=True)
        except Exception as e:
            print(e)
        similarity = util.cos_sim(a=input_user_embedding, b=candidate_embedding).mean()
        user_list.append(candidate_id)
        similarity_list.append(similarity.mean())
    similar_user = pd.DataFrame(
        {'user_id': user_list, 'similarity': similarity_list}
    )
    similar_user.sort_values(by='similarity', ascending=False, inplace=True)
    similar_user.drop(columns='similarity', inplace=True)
    return similar_user.head(n=n)
