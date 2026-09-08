import { useInfiniteQuery } from '@tanstack/react-query';
import api from '../api';

export function useInfiniteMovies(sortBy = 'popularity.desc', voteCountGte = 100) {
  return useInfiniteQuery({
    queryKey: ['infiniteMovies', sortBy, voteCountGte],

    queryFn: async ({ pageParam = 1 }) => {
      const { data } = await api.get('/movies', {
        params: {
          page: pageParam,
          sort_by: sortBy,
          vote_count_gte: voteCountGte,
        },
      });

      return data;
    },

    getNextPageParam: (lastPage) => {
      if (lastPage.page < lastPage.total_pages) {
        return lastPage.page + 1;
      }
      return undefined;
    },

    initialPageParam: 1,

    staleTime: 1000 * 60 * 5,
  });
}