import api from '../api.js';

export async function getMovieDetails(movieId) {
  const { data } = await api.get(`/details/${movieId}`);

  return {
    id:       data.movie_id,
    title:    data.title,
    overview: data.overview,
    genres:   Array.isArray(data.genres) ? data.genres : [],
    director: data.director,
    cast:     data.cast.map(c => ({
      id: c.id,
      name:  c.name,
      photo: c.photo || '/img/avatar-placeholder.png'
    })),
    poster:   data.poster
    // Você pode querer adicionar outros campos que o backend envia, se necessário
    // ex: year, rating, etc., se data tiver esses campos.
  };
}