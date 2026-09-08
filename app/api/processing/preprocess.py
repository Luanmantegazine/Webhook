import os
import string
from functools import lru_cache
from typing import List, Tuple, Any, Dict

import nltk
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from requests.exceptions import ReadTimeout
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.api.processing.client import movie_details

TMDB_KEY = os.getenv("TMDB_API_KEY") or "6177b4297dff132d300422e0343471fb"
if not TMDB_KEY:
    raise RuntimeError("Defina TMDB_API_KEY no ambiente")

HTTP_TIMEOUT = (15, 30)
POSTER_URL = "https://image.tmdb.org/t/p/w780/"

nltk.download("stopwords")
STEMMER = SnowballStemmer("portuguese")
STOP_WORDS = set(stopwords.words("portuguese"))


def _normalize_tokens(tokens: List[str]) -> List[str]:
    punc_table = str.maketrans("", "", string.punctuation)
    out = []
    for tok in tokens:
        cleaned = tok.lower().translate(punc_table)
        root = STEMMER.stem(cleaned)
        if len(root) > 2 and root not in STOP_WORDS:
            out.append(root)
    return out


def build_dataframe_from_tmdb(
        pages: int = 20,
        year_from: int | None = None,
        min_vote_count: int = 100,
        max_workers: int = 10
) -> pd.DataFrame:
    from app.api.processing.client import discover_movies

    stub: List[Dict[dict, Any]] = []
    for p in range(1, pages + 1):
        params = {
            "sort_by": "popularity.desc",
            "vote_count.gte": min_vote_count,
            "page": p,
        }
        if year_from:
            params["primary_release_date.gte"] = f"{year_from}-01-01"
        response_dict = discover_movies(**params)

        if response_dict and isinstance(response_dict, dict) and "results" in response_dict:
            movies_from_this_page = response_dict.get("results", [])
            if movies_from_this_page:
                stub.extend(movies_from_this_page)
    if not stub:
        return pd.DataFrame()

    ids = [m["id"] for m in stub if isinstance(m, dict) and "id" in m]
    if not ids:
        print("WARN: Lista de IDs vazia após processar o stub.")
        return pd.DataFrame()

    def _fetch_detail(mid: int) -> dict | None:
        try:
            det = movie_details(mid, lang="pt-BR")
            movie_keywords_list = []
            if det.get("keywords") and isinstance(det["keywords"], dict) and "keywords" in det["keywords"]:
                movie_keywords_list = [kw["name"] for kw in det["keywords"]["keywords"]]
            return {
                "movie_id": mid,
                "title": det.get("title", ""),
                "overview": det.get("overview") or "",
                "genres": [g["name"] for g in det.get("genres", [])],
                "keywords": movie_keywords_list,
                "director": next(
                    (c["name"] for c in det.get("credits", {}).get("crew", [])
                     if c.get("job") == "Director"),
                    ""
                ),
                "cast": [c["name"] for c in det.get("credits", {}).get("cast", [])[:10]],
                "poster": POSTER_URL + det["poster_path"] if det.get("poster_path") else None
            }
        except ReadTimeout:
            return None
        except Exception:
            return None

    records: List[dict] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_detail, mid): mid for mid in ids}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                records.append(res)

    df = pd.DataFrame.from_records(records)
    df.dropna(subset=["overview"], inplace=True)
    return df


class TMDBRecommender:
    def __init__(self, pages: int = 20, year_from: int | None = None):
        self.movies_raw = build_dataframe_from_tmdb(pages, year_from)
        self.movies_raw["tokens"] = (
                self.movies_raw["overview"].str.split()
                + self.movies_raw["genres"]
                + self.movies_raw["keywords"]
                + self.movies_raw["cast"]
        ).apply(_normalize_tokens)
        self._build_vectors()
        self.title_to_indices = self.movies_raw.reset_index().groupby('title')['index'].apply(list).to_dict()

    def _build_vectors(self) -> None:
        self.cv = CountVectorizer(
            max_features=5_000,
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            lowercase=False,
        )
        mat = self.cv.fit_transform(self.movies_raw["tokens"])
        self.similarity = cosine_similarity(mat)

        pass

    @lru_cache(maxsize=32)
    def recommend(self, title: str, top_k: int = 25) -> list[Any] | list[dict[str, int | None | Any]]:
        matching_indices = self.movies_raw.index[self.movies_raw["title"] == title]
        if matching_indices.empty:
            return []
        idx = matching_indices[0]

        sims = sorted(
            enumerate(self.similarity[idx]),
            key=lambda x: x[1],
            reverse=True
        )[1: top_k + 1]

        recommended_movies = []
        for i, _score in sims:
            movie_data = self.movies_raw.iloc[i]
            recommended_movies.append({
                "title": movie_data["title"],
                "movie_id": int(movie_data["movie_id"]) if pd.notna(movie_data["movie_id"]) else None,
                "poster": movie_data["poster"] or None
            })
        return recommended_movies


if __name__ == "__main__":
    rec = TMDBRecommender(pages=50, year_from=2020)
