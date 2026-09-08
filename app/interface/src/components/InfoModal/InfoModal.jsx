import React from 'react';
import YouTube from 'react-youtube';
import './InfoModal.css';


const TMDB_PROVIDER_LOGO_BASE_URL = "https://image.tmdb.org/t/p/w92";

export default function InfoModal({ isOpen, onClose, movie, loading }) {
  if (!isOpen) return null;

  const {
    title    = 'Carregando…',
    overview = '',
    genres   = [],
    director = '',
    cast     = [],
    poster,
    vote_average,
    vote_count,
    trailer_key,
    watch_providers
  } = movie || {};

  const renderStars = (rating) => {
    const starsTotal = 5;
    const starPercentage = (rating / 10) * 100;
    const starPercentageRounded = `${Math.round(starPercentage / 10) * 10}%`;
    return (
      <div className="stars-outer">
        <div className="stars-inner" style={{ width: starPercentageRounded }}></div>
      </div>
    );
  };
  
  const optsYouTube = {
    height: '390',
    width: '100%',
    playerVars: {
      autoplay: 0,
      modestbranding: 1,
      rel: 0,
    },
  };

  const renderProviders = (providersList, sectionTitle) => {
    if (!providersList || providersList.length === 0) return null;
    return (
      <div className="providers-section">
        <h4 className="providers-title">{sectionTitle}</h4>
        <ul className="providers-list">
          {providersList.map(provider => (
            <li key={provider.provider_id} className="provider-item" title={provider.provider_name}>
              <img 
                src={`${TMDB_PROVIDER_LOGO_BASE_URL}${provider.logo_path}`} 
                alt={provider.provider_name} 
                className="provider-logo"
              />
            </li>
          ))}
        </ul>
      </div>
    );
  };


  return (
    <div
      className="info-modal-backdrop"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="info-modal-title"
    >
      <div className="info-modal" onClick={e => e.stopPropagation()}>
        <button className="info-modal__close" onClick={onClose} aria-label="Fechar">✕</button>

        <div className="modal-layout-primary">
          <img
            src={poster || '/img/placeholder.jpg'}
            alt={title === 'Carregando…' ? 'Carregando poster' : `Poster de ${title}`}
            className="info-modal__poster"
          />

          <div className="info-modal__content">
            <h2 className="info-modal__title" id="info-modal-title">{title}</h2>
            
            {/* AVALIAÇÕES */}
            {loading ? <p>Carregando avaliações...</p> : (vote_average !== null && vote_average > 0) && (
              <div className="ratings-section">
                <span className="rating-value">{vote_average.toFixed(1)}</span>
                {renderStars(vote_average)}
                <span className="vote-count">({vote_count?.toLocaleString('pt-BR')} votos)</span>
              </div>
            )}

            <p className="info-modal__overview">{loading ? "Carregando..." : overview}</p>

            <div className="info-modal__meta">
              <p><strong>Gênero:</strong> {loading ? "..." : (Array.isArray(genres) ? genres.join(', ') : '')}</p>
              <p><strong>Diretor:</strong> {loading ? "..." : director}</p>
            </div>

            {!loading && watch_providers && (watch_providers.flatrate || watch_providers.buy || watch_providers.rent) && (
              <div className="watch-on-section">
                <h3 className="info-modal__subtitle">Onde Assistir (BR)</h3>
                {renderProviders(watch_providers.flatrate, "Streaming")}
                {renderProviders(watch_providers.buy, "Comprar")}
                {renderProviders(watch_providers.rent, "Alugar")}
                {watch_providers.link && (
                   <a href={watch_providers.link} target="_blank" rel="noopener noreferrer" className="tmdb-providers-link">
                     Ver todas as opções no TMDB
                   </a>
                )}
              </div>
            )}

            {!loading && Array.isArray(cast) && cast.length > 0 && (
              <>
                <h3 className="info-modal__subtitle">Elenco Principal</h3>
                <ul className="cast-list">
                  {cast.slice(0, 8).map((castMember) => (
                    <li key={castMember.id || castMember.name} className="cast-item">
                      <img
                        src={castMember.photo || '/img/avatar-placeholder.png'} // Fallback para foto do ator
                        alt={castMember.name}
                        className="cast-photo"
                        loading="lazy"
                      />
                      <span className="cast-name">{castMember.name}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </div>

        {!loading && trailer_key && (
          <div className="trailer-section">
            <h3 className="info-modal__subtitle trailer-title">Trailer Oficial</h3>
            <div className="video-responsive-container">
              <YouTube videoId={trailer_key} opts={optsYouTube} />
            </div>
          </div>
        )}

      </div>
    </div>
  );
}