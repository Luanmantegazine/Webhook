import api from '../api.js'

export async function fetchRecommend({ title, topK = 30}) {
    if (!title) throw new Error('title é obrigatório');

    const apiUrl = `/recommend/${encodeURIComponent(title)}`;

    const { data } = await api.get(apiUrl, {
        params: { top_k: topK }
    });
    return data;
}