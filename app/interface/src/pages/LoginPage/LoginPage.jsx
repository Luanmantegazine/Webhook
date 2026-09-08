import React from 'react';
import { useGoogleLogin } from '@react-oauth/google';
import { useAuth } from '@/context/AuthContext';
import { useNavigate } from 'react-router-dom';
import api from '../../api';
import './LoginPage.css';

export default function LoginPage() {
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleGoogleLogin = useGoogleLogin({
        onSuccess: async (codeResponse) => {
            try {
                const { data } = await api.post('/auth/google', { code: codeResponse.code });
                login(data.access_token);
                if (data.is_new_user) {
                    navigate('/onboarding');
                } else {
                    navigate('/para-voce');
                }
            } catch (error) {
                console.error("Erro no login com Google:", error);
                alert("Falha no login com Google.");
            }
        },
        flow: 'auth-code',
    });

    return (
        <div className="page-container login-page">
            <div className="login-box">
                <h1 className="login-title">Acesse sua conta</h1>
                <p className="login-subtitle">Entre com sua conta Google para receber recomendações personalizadas.</p>
                <button className="google-btn" onClick={() => handleGoogleLogin()}>
                    <img src="https://developers.google.com/identity/images/g-logo.png" alt="Google logo" />
                    Entrar com Google
                </button>
            </div>
        </div>
    );
}