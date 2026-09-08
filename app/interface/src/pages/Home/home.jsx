import React from 'react';
import { Link } from 'react-router-dom';
import './home.css';

export default function HomePage() {
  return (
    <div className="page-container home-page-content">
      <header className="home-hero">
        <h1 className="home-main-title">
          Bem-vindo ao <span className="highlight-brand">Movie Recommender</span>
        </h1>
        <p className="home-subtitle">
          Seu portal definitivo para explorar o universo cinematográfico e descobrir filmes que você vai amar.
        </p>
      </header>

      <section className="home-section about-app">
        <h2 className="section-title">Sobre o Nosso App</h2>
        <p className="section-text">
          O Movie Recommender foi desenvolvido para cinéfilos e entusiastas que buscam uma maneira inteligente
          e intuitiva de encontrar novos filmes. Seja explorando os últimos lançamentos que estão agitando
          as bilheterias ou buscando recomendações personalizadas baseadas naquele filme que marcou sua vida,
          nossa plataforma está aqui para enriquecer sua jornada cinematográfica.
        </p>
        <p className="section-text">
          Utilizamos informações atualizadas e um sistema de recomendação pensado para conectar você
          com histórias que realmente ressoam com seus gostos.
        </p>
      </section>

      <section className="home-section features">
        <h2 className="section-title">O Que Você Pode Fazer?</h2>
        <div className="features-grid">
          <div className="feature-card">
            <h3 className="feature-title">Explorar Lançamentos</h3>
            <p className="feature-text">
              Navegue pelos filmes mais recentes, veja o que está em alta e adicione
              novos títulos à sua lista para assistir.
            </p>
            <Link to="/lancamentos" className="cta-button">
              Ver Lançamentos
            </Link>
          </div>
          <div className="feature-card">
            <h3 className="feature-title">Obter Recomendações</h3>
            <p className="feature-text">
              Diga-nos um filme que você adora, e nosso sistema inteligente encontrará
              outros filmes com temas, estilos ou atmosferas semelhantes.
            </p>
            <Link to="/me-recomende" className="cta-button">
              Encontrar Recomendações
            </Link>
          </div>
        </div>
      </section>

      <footer className="home-footer">
        <p>&copy; {new Date().getFullYear()} Movie Recommender. Todos os direitos reservados por Luan Mantegazine.</p>
        <p>Sua aventura pelo cinema começa aqui!</p>
      </footer>
    </div>
  );
}