import React, { useState } from 'react';
import PropTypes from 'prop-types';
import InfoModal from "@/components/InfoModal/InfoModal";
import MovieCard from '../MovieCard/MovieCard';
import './PosterGrid.css';
import { useDetails } from "@/hooks/useDetails";

const getItemId = (item) => {
  if (!item) {
    console.warn("getItemId foi chamado com um item inválido:", item);
    return null;
  }
  return item.id || item.movie_id;
};

export default function PosterGrid({ items, className = '' }) {
  const [selectedItemId, setSelectedItemId] = useState(null);
  const { data: movieDetails, isFetching: isLoadingMovieDetails } = useDetails(selectedItemId);


  const isGridStyling = Array.isArray(items) && items.length > 5;

  const handleMovieCardClick = (itemId) => {
    setSelectedItemId(itemId);
  };

  const closeModal = () => {
    setSelectedItemId(null);
  };

  const validItems = Array.isArray(items) ? items.filter(item => item != null) : [];

  return (
    <>
      <div
        className={[
          'poster-grid',
          isGridStyling ? 'columns' : '',
          className,
        ].filter(Boolean).join(' ')}
      >
        {validItems.map((item) => {
          const itemId = getItemId(item);

          if (!itemId) {
            console.warn("Item no PosterGrid sem ID válido após filtragem:", item);
            return null;
          }

          const displayTitle = item.title || item.name;

          return (
            <div key={itemId} className="poster-item">
              <MovieCard
                movieId={itemId}
                title={displayTitle}
                poster={item.poster || item.poster_url}
                onClick={() => handleMovieCardClick(itemId)}
              />
            </div>
          );
        })}
      </div>

      <InfoModal
        isOpen={!!selectedItemId}
        onClose={closeModal}
        movie={movieDetails}
        loading={isLoadingMovieDetails}
      />
    </>
  );
}

PosterGrid.propTypes = {
  items: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.any,
      movie_id: PropTypes.any,
      title: PropTypes.string,
      name: PropTypes.string,
      poster: PropTypes.string,
      poster_url: PropTypes.string,
    })
  ).isRequired,
  className: PropTypes.string,
};;