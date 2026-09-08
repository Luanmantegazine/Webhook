// src/pages/ReleaseMoviesPage.jsx
import React, { useEffect } from 'react';
import PosterGrid from '@/components/PosterGrid/PosterGrid';
import { useInfiniteMovies } from '@/hooks/useMovies';
import { useInView } from 'react-intersection-observer';
import './ReleaseMoviesPage.css';

export default function ReleaseMoviesPage() {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    isError,
    error,
  } = useInfiniteMovies('primary_release_date.desc', 50);

  // DEBUG: Inspecionar os dados das páginas
  useEffect(() => {
    if (data?.pages) {
      console.log("Dados brutos de data.pages:", data.pages);
      data.pages.forEach((page, index) => {
        if (!page || !page.results) {
          console.warn(`Página ${index + 1} está malformada ou sem results:`, page);
        } else if (page.results.some(movie => movie === undefined || movie === null)) {
          console.warn(`Página ${index + 1} contém filmes undefined/null em results:`, page.results);
        }
      });
    }
  }, [data]);

  // AJUSTE: Tornar a construção do array 'movies' mais segura
  const movies = data?.pages.reduce((acc, page) => {
    if (page && Array.isArray(page.results)) {
      // Filtra quaisquer itens nulos ou indefinidos dentro de page.results
      const validResults = page.results.filter(movie => movie != null);
      acc.push(...validResults);
    }
    return acc;
  }, []) || [];

  // DEBUG: Ver o array 'movies' processado
  useEffect(() => {
    if (movies.length > 0 && movies.some(movie => movie === undefined || movie === null)) {
        console.error("Array 'movies' final contém undefined/null:", movies);
    }
  }, [movies]);


  const { ref, inView } = useInView({
    threshold: 0.5,
  });

  useEffect(() => {
    if (inView && hasNextPage && !isFetchingNextPage) {
      fetchNextPage();
    }
  }, [inView, hasNextPage, isFetchingNextPage, fetchNextPage]);

  return (
    <div className="page-container p-4 sm:p-6 w-full max-w-7xl mx-auto">
      {/* ... (H1, isLoading, isError, etc. como antes) ... */}
      <h1 className="page-title text-3xl sm:text-4xl font-bold mb-6 sm:mb-8 text-center text-white">
        Filmes Lançamento
      </h1>

      {isLoading && (
        <p className="status-message mt-10 text-lg text-center">Carregando lançamentos…</p>
      )}

      {isError && (
        <p className="status-message error mt-10 text-lg text-center">
          Ocorreu um erro ao buscar os filmes: {error?.message || 'Erro desconhecido'}
        </p>
      )}

      {!isLoading && !isError && movies.length > 0 && (
        <PosterGrid items={movies} />
      )}

      <div className="load-more-trigger" ref={ref}>
        {isFetchingNextPage && (
          <p className="status-message mt-8 text-center">Carregando mais filmes…</p>
        )}
        {!isFetchingNextPage && !hasNextPage && movies.length > 0 && (
          <p className="status-message mt-8 text-center">Você chegou ao fim da lista!</p>
        )}
      </div>

      {!isLoading && !isError && movies.length === 0 && (
        <p className="status-message mt-10 text-lg text-center">
          Nenhum lançamento encontrado no momento.
        </p>
      )}
    </div>
  );
}