import React from 'react';
import PropTypes from 'prop-types';
import './MovieCard.css';

export default function MovieCard({ movieId, title, poster, onClick }) {
  const placeholder = '/img/placeholder.jpg';

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (onClick) {
        onClick(movieId);
      }
    }
  };

  const cardTitle = title || "Título Indisponível";

  return (
    <div
      className="movie-card"
      role="button"
      tabIndex={0}
      onClick={() => {
        if (onClick) {
          onClick(movieId);
        }
      }}
      onKeyDown={handleKeyDown}
      aria-label={`Ver detalhes de ${cardTitle}`}
      title={cardTitle}
    >
      <img
        src={poster || placeholder}
        alt={`Poster de ${cardTitle}`}
        loading="lazy"
        draggable="false"
      />
    </div>
  );
}

MovieCard.propTypes = {
  movieId: PropTypes.oneOfType([PropTypes.string, PropTypes.number]),
  title: PropTypes.string.isRequired,
  poster: PropTypes.string,
  onClick: PropTypes.func,
};