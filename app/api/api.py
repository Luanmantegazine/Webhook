import uvicorn
import httpx
import os
from typing import List
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session
from app.api.database import SessionLocal, engine

from app.api import db_models, security
from app.api.models import (MovieBrief, MovieDetail, Cast, RecResp, Provider, WatchProviderRegionDetails,
                    PaginatedMovieResponse, TVSeriesDetail,
                    PaginatedTVSeriesResponse, TVSeriesBrief, GoogleToken)

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.security import OAuth2PasswordBearer
from app.api.processing.preprocess import TMDBRecommender

from fastapi.middleware.cors import CORSMiddleware
from app.api.processing.client import (discover_movies, movie_details, IMG_W185, IMG_W500, fetch_person_photo,
                               search_movie, discover_tv_shows, IMG_W1280, IMG_W780, tv_series_details)

rec_engine = TMDBRecommender(pages=20, year_from=2000)
db_models.Base.metadata.create_all(bind=engine)
app = FastAPI(
    title="Movie Recommender API",
    openapi_prefix="/api"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.get("/movies", response_model=PaginatedMovieResponse, tags=["Movies"])
def get_movies(
        page: int = Query(1, ge=1, description="Página de resultados"),
        sort_by: str = Query("popularity.desc", description="Ordenação TMDB"),
        vote_count_gte: int = Query(100, ge=0, description="Min. de votos"),
):
    params = {
        "sort_by": sort_by,
        "vote_count.gte": vote_count_gte,
        "page": page,
    }

    tmdb_data = discover_movies(**params)

    processed_results = [
        MovieBrief(
            id=m["id"],
            title=m["title"],
            poster=IMG_W500 + m["poster_path"] if m.get("poster_path") else None,
            year=(m.get("release_date") or "")[:4] or None,
            rating=m.get("vote_average")
        )
        for m in tmdb_data.get("results", [])
    ]

    return PaginatedMovieResponse(
        page=tmdb_data.get("page"),
        results=processed_results,
        total_pages=tmdb_data.get("total_pages"),
        total_results=tmdb_data.get("total_results")
    )


@app.post("/auth/google")
async def auth_google(token: GoogleToken, db: Session = Depends(get_db)):
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": token.code,
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
        "redirect_uri": "http://localhost:5173",
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient() as client:
        google_response = await client.post(token_url, data=data)
        if google_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Could not validate Google token")

        id_token = google_response.json().get("id_token")

    user_info_url = f"https://oauth2.googleapis.com/tokeninfo?id_token={id_token}"
    async with httpx.AsyncClient() as client:
        user_info_response = await client.get(user_info_url)
        user_info = user_info_response.json()

    google_id = user_info.get("sub")
    email = user_info.get("email")
    username = email.split('@')[0]

    user = db.query(db_models.User).filter(db_models.User.google_id == google_id).first()

    is_new_user = False
    if not user:
        is_new_user = True
        user = db_models.User(
            username=username,
            email=email,
            google_id=google_id,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    access_token = security.create_access_token(data={"sub": user.username})

    return {"access_token": access_token, "token_type": "bearer", "is_new_user": is_new_user}


@app.get("/details/{movie_id}", response_model=MovieDetail, tags=["Movies"])
def details(movie_id: int):
    try:
        data = movie_details(movie_id, lang="pt-BR", append_to_response="credits,keywords,videos,watch/providers")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Erro TMDB: {e}")

    cast_list: List[Cast] = []
    for member in data.get("credits", {}).get("cast", [])[:10]:
        pid = member.get("id")
        name = member.get("name", "")
        profile = member.get("profile_path")

        if profile:
            photo_url = IMG_W185 + profile
        else:
            photo_url = fetch_person_photo(pid) if pid else None

        cast_list.append(Cast(id=pid, name=name, photo=photo_url))

    crew = data.get("credits", {}).get("crew", [])
    director = next((c["name"] for c in crew if c.get("job") == 'Director'), None)

    vote_average = data.get("vote_average")
    vote_count = data.get("vote_count")

    trailer_key = None

    if data.get("videos") and data["videos"].get("results"):
        videos = data["videos"]["results"]
        official_trailers = [
            v for v in videos
            if v.get("site") == "YouTube" and v.get("type") == "Trailer" and v.get("official") is True
        ]
        if official_trailers:
            trailer_key = official_trailers[0].get("key")
        else:
            any_trailers = [
                v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Trailer"
            ]
            if any_trailers:
                trailer_key = any_trailers[0].get("key")

    watch_providers_data = None
    if data.get("watch/providers") and data["watch/providers"].get("results"):
        results_by_country = data["watch/providers"]["results"]
        if "BR" in results_by_country:
            br_providers = results_by_country["BR"]
            watch_providers_data = WatchProviderRegionDetails(
                link=br_providers.get("link"),
                flatrate=[Provider(**p) for p in br_providers.get("flatrate", [])],
                rent=[Provider(**p) for p in br_providers.get("rent", [])],
                buy=[Provider(**p) for p in br_providers.get("buy", [])]
            )

    return MovieDetail(
        movie_id=movie_id,
        title=data.get("title", ""),
        overview=data.get("overview"),
        genres=[g["name"] for g in data.get("genres", [])],
        director=director,
        cast=cast_list,
        poster=IMG_W500 + data["poster_path"] if data.get("poster_path") else None,
        vote_average=round(vote_average, 1) if vote_average is not None else None,
        vote_count=vote_count,
        trailer_key=trailer_key,
        watch_providers_data=watch_providers_data,

    )


@app.get("/search", response_model=List[MovieBrief], tags=["Movies"])
def search(q: str):
    results = search_movie(q)[:10]
    return [
        MovieBrief(
            id=m["id"],
            title=m["title"],
            poster=IMG_W500 + m["poster_path"] if m.get("poster_path") else None,
        )
        for m in results
    ]


@app.get("/recommend/{title}", response_model=List[RecResp], tags=["Recommender"])
async def get_content_recommendations(title: str, top_k: int = Query(45, ge=1, le=50)):
    recommendations = rec_engine.recommend(title=title, top_k=top_k)

    return recommendations


@app.get("/tv/discover", response_model=PaginatedTVSeriesResponse, tags=["TV Séries"])
async def get_discover_tv_shows(
        page: int = Query(1, ge=1, description="Página de resultados"),
        sort_by: str = Query("popularity.desc", description="Ordenação TMDB"),
        vote_count_gte: int = Query(50, ge=0, description="Minimo de votos")
):
    params = {
        "page": page,
        "sort_by": sort_by,
        "vote_count_gte": vote_count_gte,

    }

    processed_results = []
    current_page_from_response = page
    total_page_from_response = 0
    total_results_from_response = 0

    tmdb_data = await run_in_threadpool(discover_tv_shows, **params)

    if tmdb_data and isinstance(tmdb_data, dict) and "results" in tmdb_data:
        current_page_from_response = tmdb_data.get("page", page)
        total_page_from_response = tmdb_data.get("total_pages", 0)
        total_results_from_response = tmdb_data.get("total_results", 0)

        for s in tmdb_data.get("results", []):
            first_air_year = None
            first_air_date_str = s.get("first_air_date")
            if first_air_date_str:
                if isinstance(first_air_date_str, str) and len(first_air_date_str) >= 4:
                    first_air_year = first_air_date_str[:4]
            poster_full_url = None
            poster_path = s.get("poster_path")
            if poster_path:
                poster_full_url = IMG_W500 + poster_path

            processed_results.append(
                TVSeriesBrief(
                    id=s.get("id"),
                    name=s.get("name"),
                    poster_url=poster_full_url,
                    first_air_year=first_air_year,
                    vote_average=round(s.get("vote_average", 0.0), 1) if s.get("vote_average") is not None else None
                )
            )

    return PaginatedTVSeriesResponse(
        page=current_page_from_response,
        results=processed_results,
        total_pages=total_page_from_response,
        total_results=total_results_from_response
    )


@app.get("/tv/{series_id}/details", response_model=TVSeriesDetail, tags=["TV Séries"])
async def get_tv_series_details(series_id: int, lang: str = Query("pt-BR")):
    append_items = "aggregate_credits,keywords,videos,watch/providers"
    data = await run_in_threadpool(
        tv_series_details,
        series_id=series_id,
        lang=lang,
        append_to_response=append_items,
    )
    genres = [g["name"] for g in data.get("genres", []) if isinstance(g, dict) and "name" in g]

    created_by_dicts = []
    for creator_data in data.get("created_by", []):
        if isinstance(creator_data, dict):
            profile_full_url = None
            if creator_data.get("profile_path"):
                profile_full_url = IMG_W185 + creator_data["profile_path"]

            creator_id = creator_data.get("id")
            credit_id_val = creator_data.get("credit_id")
            name_val = creator_data.get("name")

            if creator_id is not None and credit_id_val is not None and name_val is not None:
                created_by_dicts.append({
                    "id": creator_id,
                    "credit_id": credit_id_val,
                    "name": name_val,
                    "gender": creator_data.get("gender"),
                    "profile_path": profile_full_url
                })

    main_cast_list_dicts = []
    if data.get("aggregate_credits") and isinstance(data["aggregate_credits"], dict):
        for member_data in data["aggregate_credits"].get("cast", [])[:15]:
            if isinstance(member_data, dict):
                photo_full_url = None
                if member_data.get("profile_path"):
                    photo_full_url = IMG_W185 + member_data["profile_path"]

                character_name = None
                roles = member_data.get("roles", [])
                if roles and isinstance(roles, list) and len(roles) > 0 and isinstance(roles[0], dict):
                    character_name = roles[0].get("character")

                main_cast_list_dicts.append({
                    "id": member_data.get("id"),
                    "name": member_data.get("name"),
                    "photo": photo_full_url,
                    "character": character_name
                })

    keywords_list = []
    if data.get("keywords") and isinstance(data["keywords"], dict) and "results" in data[
        "keywords"]:
        keywords_list = [kw["name"] for kw in data["keywords"]["results"] if isinstance(kw, dict) and "name" in kw]
    elif data.get("keywords") and isinstance(data["keywords"], dict) and "keywords" in data[
        "keywords"]:
        keywords_list = [kw["name"] for kw in data["keywords"]["keywords"] if isinstance(kw, dict) and "name" in kw]

    trailer_key = None
    if data.get("videos") and data["videos"].get("results"):
        videos = data["videos"]["results"]
        official_trailers = [v for v in videos if
                             v.get("site") == "YouTube" and v.get("type") == "Trailer" and v.get("official") is True]
        if official_trailers:
            trailer_key = official_trailers[0].get("key")
        else:
            any_trailers = [v for v in videos if v.get("site") == "YouTube" and v.get("type") == "Trailer"]
            if any_trailers:
                trailer_key = any_trailers[0].get("key")

    watch_providers_data = None
    watch_providers_response_key = "watch/providers"
    if data.get(watch_providers_response_key) and data[watch_providers_response_key].get("results"):
        results_by_country = data[watch_providers_response_key]["results"]
        if "BR" in results_by_country:
            br_providers = results_by_country["BR"]
            watch_providers_data = WatchProviderRegionDetails(
                link=br_providers.get("link"),
                flatrate=[Provider(**p) for p in br_providers.get("flatrate", []) if isinstance(p, dict)],
                rent=[Provider(**p) for p in br_providers.get("rent", []) if isinstance(p, dict)],
                buy=[Provider(**p) for p in br_providers.get("buy", []) if isinstance(p, dict)]
            )

    poster_full_url = None
    if data.get("poster_path"):
        poster_full_url = IMG_W500 + data["poster_path"]

    backdrop_full_url = None
    if data.get("backdrop_path"):
        backdrop_full_url = IMG_W1280 + data["backdrop_path"] if 'IMG_W1280' in globals() else IMG_W780 + data[
            "backdrop_path"]

    return TVSeriesDetail(
        id=data.get("id"),
        name=data.get("name"),
        overview=data.get("overview"),
        genres=genres,
        created_by=created_by_dicts,
        episode_run_time=data.get("episode_run_time", []),
        first_air_date=data.get("first_air_date"),
        last_air_date=data.get("last_air_date"),
        number_of_episodes=data.get("number_of_episodes"),
        number_of_seasons=data.get("number_of_seasons"),
        poster_url=poster_full_url,
        backdrop_url=backdrop_full_url,
        vote_average=round(data.get("vote_average", 0.0), 1) if data.get("vote_average") is not None else None,
        vote_count=data.get("vote_count"),
        status=data.get("status"),
        tagline=data.get("tagline"),
        homepage=data.get("homepage"),
        trailer_key=trailer_key,
        watch_providers=watch_providers_data,
        cast=main_cast_list_dicts,
        keywords=keywords_list
    )


if __name__ == '__main__':
    uvicorn.run(app=app, port=8000)
