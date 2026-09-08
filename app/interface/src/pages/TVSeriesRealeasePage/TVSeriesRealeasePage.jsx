// src/pages/TVSeriesReleasePage.jsx (NOVO)
import React, { useEffect } from 'react';
import PosterGrid from '@/components/PosterGrid/PosterGrid';
import { useInfiniteTVSeries } from '@/hooks/useInfiniteTVSeries'; // Novo hook
import { useInView } from 'react-intersection-observer';
import './TVSeriesReleasePage.css'

export default function TVSeriesReleasePage() {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
    error,
  } = useInfiniteTVSeries(
    'first_air_date.desc', // Ordenar por data de lançamento mais recente para séries
    50                     // Mínimo de 50 votos (ajuste conforme necessário)
  );

  const { ref, inView } = useInView({
    threshold: 0.5,
  });

  useEffect(() => {
    if (inView && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [inView, hasNextPage, isFetchingNextPage, fetchNextPage]);

  const series = data?.pages.flatMap(page => page.results) || [];

  return (
    <div className="page-container p-4 sm:p-6 w-full max-w-7xl mx-auto">
      <h1 className="page-title text-3xl sm:text-4xl font-bold mb-6 sm:mb-8 text-center text-white">
        Séries Lançamento
      </h1>

      {isLoading && (
        <p className="status-message mt-10 text-lg text-center">Carregando séries…</p>
      )}

      {isError && (
        <p className="status-message error mt-10 text-lg text-center">
          Ocorreu um erro ao buscar as séries: {error?.message || 'Erro desconhecido'}
        </p>
      )}

      {!isLoading && !isError && series.length > 0 && (
        <PosterGrid items={series} /> // PosterGrid deve funcionar para séries se os dados (id, name/title, poster_url) forem compatíveis
      )}

      <div className="load-more-trigger" ref={ref}>
        {isFetchingNextPage && (
          <p className="status-message mt-8 text-center">Carregando mais séries…</p>
        )}
        {!isFetchingNextPage && !hasNextPage && series.length > 0 && (
          <p className="status-message mt-8 text-center">Você chegou ao fim da lista!</p>
        )}
      </div>

      {!isLoading && !isError && series.length === 0 && (
        <p className="status-message mt-10 text-lg text-center">
          Nenhuma série lançamento encontrada no momento.
        </p>
      )}
    </div>
  );
}