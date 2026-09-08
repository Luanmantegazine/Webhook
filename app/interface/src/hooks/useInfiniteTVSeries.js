import { useInfiniteQuery } from '@tanstack/react-query';
import api from '../api';

export function useInfiniteTVSeries(sortBy = 'popularity.desc', voteCountGte = 50) {
  return useInfiniteQuery({
    queryKey: ['infiniteTVSeries', sortBy, voteCountGte],
    queryFn: async ({ pageParam = 1 }) => {
      const { data } = await api.get('/tv/discover', {
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