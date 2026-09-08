import logging
from pathlib import Path
from typing import Tuple
from functools import lru_cache
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import preprocess

logger = logging.getLogger(__name__)


class Main:
    DATA_DIR = Path("Files")
    DATA_DIR.mkdir(exist_ok=True)

    def __init__(self) -> None:
        self.new_df: pd.Dataframe | None = None
        self.movies: pd.Dataframe | None = None
        self.movies2: pd.Dataframe | None = None

    def get_df(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.new_df is None:
            self.load_data()
        return self.new_df, self.movies, self.movies2

    def load_data(self) -> None:
        df_path = self.DATA_DIR / 'new_dif_dict.pkl'

        if df_path.exists():
            logger.info("Lendo Dataframes")
            self.new_df = pd.read_parquet(df_path)
            self.movies = pd.read_parquet(self.DATA_DIR / "movies.parquet")
            self.movies2 = pd.read_parquet(self.DATA_DIR / "movies2.parquet")
        else:
            logger.info("Cache nao encontrado: rodando preprocess...")
            self.movies, self.new_df, self.movies2 = preprocess.read_csv_to_df()
            self.movies.to_parquet(self.DATA_DIR / "movies.parquet", index=False)
            self.movies2.to_parquet(self.DATA_DIR / "movies2.parquet", index=False)
            self.new_df.to_parquet(df_path, index=False)

    @staticmethod
    @lru_cache(maxsize=8)
    def _vectorise(series: pd.Series) -> pd.DataFrame:
        cv = CountVectorizer(max_features=5_000, stop_words="english")
        matrix = cv.fit_transform(series).toarray()
        return cosine_similarity(matrix)

    def _ensure_similarity(self, column: str) -> None:
        sim_file = self.DATA_DIR / f"sim_{column}.joblib"
        if sim_file.exists():
            return
        logger.info("Calculando matriz de similaridade para %s ...", column)
        sim = self._vectorise(self.new_df[column])
        joblib.dump(sim, sim_file, compress=3)

    def prepare_all_similarities(self) -> None:
        self.load_data()
        for feature in ("tags", "genres", "keywords", "tcast", "tproduction_comp"):
            self._ensure_similarity(feature)




