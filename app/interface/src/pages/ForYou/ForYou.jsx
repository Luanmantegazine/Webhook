import React from 'react';
import { useQuery } from '@tanstack/react-query';
import api from '@/api';
import PosterGrid from '@/components/PosterGrid/PosterGrid';
import './ForYou.css';

// Hook customizado para buscar as recomendações
const usePersonalRecommendations = () => {
    return useQuery({
        queryKey: ['personalRecommendations'],
        queryFn: async () => {
            const { data } = await api.get('/users/me/recommendations');
            return data;
        },
        staleTime: 1000 * 60 * 15, // 15 minutos
    });
};

export default function ForYouPage() {
    const { data: recommendations, isLoading, isError } = usePersonalRecommendations();

    return (
        <div className="page-container for-you-page">
            <h1 className="page-title">Especialmente Para Você</h1>
            <p className="page-subtitle">Com base nos seus filmes favoritos, achamos que você vai gostar destes:</p>

            {isLoading && <p className="status-message">Buscando recomendações...</p>}
            {isError && <p className="status-message error">Não foi possível carregar suas recomendações.</p>}

            {recommendations && recommendations.length > 0 && (
                <PosterGrid items={recommendations} />
            )}

            {recommendations && recommendations.length === 0 && (
                 <p className="status-message">Ainda não temos recomendações. Favorite mais filmes para nos ajudar!</p>
            )}
        </div>
    );
}