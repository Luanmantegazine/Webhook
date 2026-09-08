import React from 'react';
import { BrowserRouter, Routes, Route, NavLink, useNavigate, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';

import HomePage from './pages/Home/home';
import LoginPage from './pages/LoginPage/LoginPage';
import OnboardingPage from './pages/OnboardingPage/OnboardingPage';
import ForYouPage from './pages/ForYou/ForYou';
import ReleaseMoviesPage from './pages/ReleaseMoviesPage/ReleaseMoviePage';
import TVSeriesReleasePage from './pages/TVSeriesRealeasePage/TVSeriesRealeasePage';
import RecommendationFinderPage from './pages/RecommendationFinderPage/RecommendationFinderPage';

import './App.css';

const AppTitle = () => {
  const navigate = useNavigate();
  return <h1 className="app-title" onClick={() => navigate('/')} style={{ cursor: 'pointer' }}>Movie Recommender</h1>;
};

// Componente para proteger rotas
const ProtectedRoute = ({ children }) => {
    const { user } = useAuth();
    return user ? children : <Navigate to="/login" />;
};

export default function App() {
  const { user, logout } = useAuth();

  return (
    <BrowserRouter>
      <div className="app-container">
        <header className="app-header">
          <AppTitle />
          <nav className="app-nav">
            <NavLink to="/" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'} end>Home</NavLink>
            <NavLink to="/lancamentos/filmes" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>Filmes</NavLink>
            <NavLink to="/lancamentos/series" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>Séries</NavLink>
            <NavLink to="/me-recomende" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>Me Recomende</NavLink>

            {user ? (
              <>
                <NavLink to="/para-voce" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>Para Você</NavLink>
                <button onClick={logout} className="nav-link">Sair</button>
              </>
            ) : (
              <NavLink to="/login" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>Entrar</NavLink>
            )}
          </nav>
        </header>

        <main className="app-main-content">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/lancamentos/filmes" element={<ReleaseMoviesPage />} />
            <Route path="/lancamentos/series" element={<TVSeriesReleasePage />} />
            <Route path="/me-recomende" element={<RecommendationFinderPage />} />
            <Route path="/login" element={<LoginPage />} />
            <Route path="/onboarding" element={<ProtectedRoute><OnboardingPage /></ProtectedRoute>} />
            <Route path="/para-voce" element={<ProtectedRoute><ForYouPage /></ProtectedRoute>} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}