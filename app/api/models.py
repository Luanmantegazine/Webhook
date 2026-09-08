from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class MovieBrief(BaseModel):
    id: int
    title: str
    poster: Optional[str] = None
    year: Optional[str] = None
    rating: Optional[float] = None


class RecResp(BaseModel):
    title: str
    movie_id: Optional[int] = None
    poster: Optional[str] = None


class Cast(BaseModel):
    id: int
    name: str
    photo: str | None = None


class MovieDetail(BaseModel):
    movie_id: int
    title: str
    overview: str
    genres: list[str]
    director: str
    cast: list[Cast] = []
    poster: str

    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    trailer_key: Optional[str] = None  #
    watch_providers: Optional[WatchProviderRegionDetails] = None


class RecommendRequest(BaseModel):
    liked_movies: List[str] = Field("likedMovies", alias="title")
    top_k: Optional[int] = 10

    class Config:
        allow_population_by_field_name = True
        allow_population_by_alias = True


class Provider(BaseModel):
    logo_path: Optional[str] = None
    provider_id: int
    provider_name: str
    display_priority: Optional[int] = None


class WatchProviderRegionDetails(BaseModel):
    link: Optional[str] = None
    flatrate: Optional[List[Provider]] = None
    rent: Optional[List[Provider]] = None
    buy: Optional[List[Provider]] = None


class PaginatedMovieResponse(BaseModel):
    page: int
    results: List[MovieBrief]
    total_pages: int
    total_results: int


class TVSeriesBrief(BaseModel):
    id: int
    name: str
    poster_url: Optional[str] = None
    first_air_year: Optional[str] = None
    vote_average: Optional[float] = None


class PaginatedTVSeriesResponse(BaseModel):
    page: int
    results: List[TVSeriesBrief]
    total_pages: int
    total_results: int


class TVSeriesDetail(BaseModel):
    id: int
    name: str
    overview: Optional[str] = None
    genres: List[str] = []
    created_by: List[Dict[str, Any]] = []
    episode_run_time: List[int] = []
    first_air_date: Optional[str] = None
    last_air_date: Optional[str] = None
    number_of_episodes: Optional[int] = None
    number_of_seasons: Optional[int] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    trailer_key: Optional[str] = None
    status: Optional[str] = None
    tagline: Optional[str] = None
    homepage: Optional[str] = None
    watch_providers: Optional[WatchProviderRegionDetails] = None
    aggregate_credits: Optional[Dict[str, List[Cast]]] = None
    keywords: List[str] = []
    cast: List[Cast] = []
    seasons: Optional[List[Dict[str, Any]]] = None


class Creator(BaseModel):
    id: int
    credit_id: str
    name: str
    gender: Optional[int] = None
    profile_path: Optional[str] = None


class UserBase(BaseModel):
    username: str


class User(UserBase):
    id: int

    class Config:
        orm_mode = True


class Token(BaseModel):
    access_token: str
    token_type: str


class GoogleToken(BaseModel):
    code: str
