import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Carrega o CSV
df = pd.read_csv("tags_cto.csv")

# Modelo pré-treinado multilingual (PT/EN)
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')


def validar_tags(df, colunas_tags, limiar=0.65):
    resultados = []

    for idx, row in df.iterrows():
        descricao = f"{row['product_name']} {row['product_description']}"
        embedding_descricao = model.encode(descricao, convert_to_tensor=True)

        # Valida cada coluna de tags
        for coluna in colunas_tags:
            if coluna not in df.columns:
                continue  # Pula caso a coluna não exista

            # Separa as tags (supondo que estejam separadas por '|')
            tags = str(row[coluna]).split('|')
            embeddings_tags = model.encode(tags, convert_to_tensor=True)
            scores = cosine_similarity(
                embedding_descricao.cpu().numpy().reshape(1, -1),
                embeddings_tags.cpu().numpy()
            )[0]

            for tag, score in zip(tags, scores):
                resultados.append({
                    'product_name': row['product_name'],
                    'coluna': coluna,
                    'tag': tag,
                    'similaridade': float(score),
                    'valido': score >= limiar
                })

    return pd.DataFrame(resultados)


# Lista de colunas a serem validadas
colunas_para_validar = ['positive_tags', 'negative_tags', 'uso_tags']  # Adicione as colunas que desejar

# Valida as tags e obtém os resultados
resultado_validacao = validar_tags(df, colunas_para_validar)

# Salva os resultados em um novo CSV
resultado_validacao.to_csv("resultado_validacao_tags.csv", index=False)

print(resultado_validacao.head())
