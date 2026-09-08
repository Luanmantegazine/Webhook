// src/pages/RecommendationFinderPage.jsx
import React, { useState } from 'react';
import { useDebounce } from '@/hooks/useDebounc';
import { useSearch } from '@/hooks/useSearch';
import { useRecommend } from '@/hooks/useRecommend';
import PosterGrid from '@/components/PosterGrid/PosterGrid';
import './RecommendationFinderPage.css';

const getItemId = (item) => item.id || item.movie_id;

export default function RecommendationFinderPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedMovieDisplay, setSelectedMovieDisplay] = useState(null);
  const [movieTitleForApi, setMovieTitleForApi] = useState('');

  const debouncedSearchQuery = useDebounce(searchQuery, 400);

  const {
    data: searchResults,
    isLoading: isSearching,
    isError: isSearchError,
    error: searchError,
  } = useSearch(debouncedSearchQuery);

  const {
    data: recommendations,
    isLoading: isLoadingRecommendations,
    isError: isRecommendationsError,
    error: recommendationsError,
  } = useRecommend(movieTitleForApi, 10); // topK = 10

  const handleSearchChange = (event) => {
    setSearchQuery(event.target.value);
    setMovieTitleForApi('');
    setSelectedMovieDisplay(null);
  };

  const handleSelectMovie = (movie) => {
    setSearchQuery('');
    setSelectedMovieDisplay(movie);
    setMovieTitleForApi(movie.title);
  };

  return (
    <div className="page-container p-4 sm:p-6 w-full max-w-7xl mx-auto">
      <h1 className="page-title text-3xl sm:text-4xl font-bold mb-6 sm:mb-8 text-center text-white">
        Recomendações de Filmes
      </h1>

      <div className="search-section">
        <input
          type="text"
          placeholder="Busque por um filme que você gosta..."
          value={searchQuery}
          onChange={handleSearchChange}
          className="search-input"
        />
        {isSearching && <p className="search-status mt-2 text-sm">Buscando...</p>}

        {debouncedSearchQuery && searchResults && !selectedMovieDisplay && (
          <div className="search-results-container">
            {searchResults.length === 0 && !isSearching && (
              <p className="search-status p-3 text-sm">Nenhum filme encontrado para "{debouncedSearchQuery}".</p>
            )}
            {searchResults.length > 0 && (
                // <h3 className="p-2 text-sm font-semibold">Resultados:</h3> Se quiser título
                <ul className="search-results-list">
                {searchResults.slice(0, 10).map((movie) => (
                    <li
                    key={getItemId(movie)}
                    className="search-result-item"
                    onClick={() => handleSelectMovie(movie)}
                    >
                    {movie.poster && (
                        <img src={movie.poster} alt={`Poster de ${movie.title}`} />
                    )}
                    <span>{movie.title} ({movie.year || 'N/A'})</span>
                    </li>
                ))}
                </ul>
            )}
          </div>
        )}
        {isSearchError && <p className="search-status error mt-2 text-sm">Erro ao buscar: {searchError?.message || "Tente novamente."}</p>}
      </div>

      {movieTitleForApi && selectedMovieDisplay && (
        <div className="recommendations-section">
          <h2 className="text-xl sm:text-2xl font-semibold mb-4 sm:mb-6 text-center text-white">
            Filmes recomendados baseados em: <strong className="highlight-movie">{selectedMovieDisplay.title}</strong>
          </h2>
          {isLoadingRecommendations && <p className="status-message text-lg text-center">Carregando recomendações…</p>}
          {isRecommendationsError && (
            <p className="status-message error text-lg text-center">
              Erro ao carregar recomendações: {recommendationsError?.message || 'Tente outro filme.'}
            </p>
          )}

          {!isLoadingRecommendations && !isRecommendationsError && recommendations && recommendations.length > 0 && (
            <PosterGrid items={recommendations} />
          )}
          {!isLoadingRecommendations && !isRecommendationsError && (!recommendations || recommendations.length === 0) && (
             <p className="status-message mt-6 text-lg text-center">
              Nenhuma recomendação encontrada para este filme. Que tal tentar outro?
            </p>
          )}
        </div>
      )}

      {!movieTitleForApi && !searchQuery && (
         <p className="initial-prompt mt-10 text-lg text-center">
            Para começar, use a busca acima para encontrar um filme. Nós te daremos recomendações baseadas na sua escolha!
          </p>
      )}
    </div>
  );
}