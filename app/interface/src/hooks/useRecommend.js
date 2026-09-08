import { useQuery } from '@tanstack/react-query'
import { fetchRecommend} from "@/hooks/fetchRecommend.js";

export function useRecommend(title, topK = 30) {
  return useQuery({
    queryKey: ['recommend', title, topK],
    queryFn: () => fetchRecommend({ title, topK }),
    enabled: Boolean(title),
    staleTime: 1000 * 60 * 10,
  });
}