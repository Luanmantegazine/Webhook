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
      name:  c.name,
      photo: c.photo || '/img/avatar-placeholder.png'
    })),
    poster:   data.poster
  };
}