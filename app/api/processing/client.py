import os
import requests
import functools
from typing import Any, Dict, Optional

TMDB_KEY = os.getenv("TMDB_API_KEY") or "6177b4297dff132d300422e0343471fb"
TMDB_V3_ROOT = "https://api.themoviedb.org/3"
IMG_W500 = "https://image.tmdb.org/t/p/w500"
IMG_W185 = "https://image.tmdb.org/t/p/w185"
IMG_W780 = "https://image.tmdb.org/t/p/w780"
IMG_W1280 = "https://image.tmdb.org/t/p/w1280"
TIMEOUT = (15, 30)


def _safe_request(url: str, params: dict[str, Any]) -> dict:
    resp = requests.get(url, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def gett(path: str, **params: Any) -> dict:
    url = f"{TMDB_V3_ROOT}/{path.lstrip('/')}"
    payload = {**params, "api_key": TMDB_KEY}
    resp = requests.get(url, params=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


@functools.lru_cache(maxsize=256)
def list_genres(lang: str = "pt-BR") -> dict[str, int]:
    data = gett("/genre/movie/list", language=lang)
    return {g["name"].lower(): g["id"] for g in data.get("genres", [])}


@functools.lru_cache(maxsize=1024)
def discover_movies(**filters: Any) -> dict:
    data = gett("/discover/movie", **filters)
    return data


@functools.lru_cache(maxsize=4096)
def movie_details(movie_id: int,
                  lang: str = "pt-BR",
                  append_to_response: Optional[str] = "credits") -> dict:
    return gett(f"/movie/{movie_id}", language=lang, append_to_response=append_to_response)


@functools.lru_cache(maxsize=4096)
def fetch_person_photo(person_id: int) -> str | None:
    data = gett(f"/person/{person_id}/images")
    profiles = data.get("profiles", [])
    if profiles:
        return IMG_W185 + profiles[0].get("file_path", "")
    return None


@functools.lru_cache(maxsize=4096)
def movie_recommendations(movie_id: int, lang: str = "pt-BR") -> list[dict]:
    data = gett(f"/movie/{movie_id}/recommendations", language=lang)
    return data.get("results", [])


def search_movie(query: str, lang: str = "pt-BR") -> list[dict]:
    data = gett("/search/movie", language=lang, query=query)
    return data.get("results", [])


@functools.lru_cache(maxsize=1024)
def discover_tv_shows(**filters: Any) -> dict:
    data = gett("/discover/tv", **filters)
    return data


@functools.lru_cache(maxsize=256)
def tv_series_details(series_id: int,
                      lang: str = "pt-BR",
                      append_to_response: Optional[str] = None
                      ) -> Dict[str, Any]:
    path = f"/tv/{series_id}"
    api_call_params = {"language": lang}
    if append_to_response:
        api_call_params["append_to_response"] = append_to_response

    data = gett(path, **api_call_params)
    return data
