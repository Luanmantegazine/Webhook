import { useQuery } from '@tanstack/react-query';
import api from '../api';

export function useSearch(query, options = {}) {
  return useQuery({
    queryKey: ['search', query],
    queryFn: () => api.get('/search', { params: { q: query } }).then(res => res.data),
    enabled: !!query && query.length > 2 && (options.enabled !== undefined ? options.enabled : true),
    staleTime: 1000 * 60 * 5, // 5 minutos de stale time
    ...options,
  });
}