import { useQuery } from '@tanstack/react-query'
import api           from '../api.js'

export function useDetails(movieId) {
  return useQuery({
    queryKey: ['details', movieId],
    enabled:  Boolean(movieId),
    queryFn:  () => api
      .get(`/details/${movieId}`)
      .then(res => res.data),
    staleTime: 60 * 60 * 1000, // 1h
  })
}
