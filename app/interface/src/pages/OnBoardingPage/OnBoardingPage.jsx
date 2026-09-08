import React, { useState } from 'react';
import { useInfiniteMovies } from '@/hooks/useMovies';
import { useNavigate } from 'react-router-dom';
import MovieCard from '@/components/MovieCard/MovieCard';
import api from '@/api';
import './OnboardingPage.css';

export default function OnboardingPage() {
    const [selectedIds, setSelectedIds] = useState([]);
    const navigate = useNavigate();

    // Busca filmes populares para o usuário escolher
    const { data: moviesData, isLoading } = useInfiniteMovies('popularity.desc', 500);
    const movies = moviesData?.pages.flatMap(page => page.results).slice(0, 20) || [];

    const toggleSelection = (movieId) => {
        setSelectedIds(prev =>
            prev.includes(movieId)
            ? prev.filter(id => id !== movieId)
            : [...prev, movieId]
        );
    };

    const handleSubmit = async () => {
        if (selectedIds.length < 5) {
            alert('Por favor, selecione pelo menos 5 filmes.');
            return;
        }
        try {
            await api.post('/users/me/favorites', { movie_ids: selectedIds });
            navigate('/para-voce');
        } catch (error) {
            console.error("Erro ao salvar favoritos:", error);
            alert("Não foi possível salvar seus favoritos. Tente novamente.");
        }
    };

    return (
        <div className="page-container onboarding-page">
            <h1 className="page-title">Escolha seus filmes favoritos</h1>
            <p className="onboarding-subtitle">Selecione pelo menos 5 filmes para personalizarmos suas recomendações.</p>

            {isLoading && <p>Carregando filmes...</p>}

            <div className="selection-grid">
                {movies.map(movie => (
                    <div
                        key={movie.id}
                        className={`selection-item ${selectedIds.includes(movie.id) ? 'selected' : ''}`}
                        onClick={() => toggleSelection(movie.id)}
                    >
                        <MovieCard movieId={movie.id} title={movie.title} poster={movie.poster} />
                        <div className="selection-overlay">✓</div>
                    </div>
                ))}
            </div>

            <button onClick={handleSubmit} className="cta-button" disabled={selectedIds.length < 5}>
                Concluir ({selectedIds.length}/5)
            </button>
        </div>
    );
}