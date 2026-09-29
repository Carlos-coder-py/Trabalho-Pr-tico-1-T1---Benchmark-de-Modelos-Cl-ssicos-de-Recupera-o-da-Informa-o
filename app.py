"""
TRABALHO PRÁTICO 1 — Benchmark de modelos clássicos de Recuperação da Informação
Domínio: E-commerce (dataset de produtos do Kaggle)

Como rodar a demo:   streamlit run app.py

Este arquivo tem TUDO em ordem:
  1. Carregar os dados
  2. Pré-processamento (limpar o texto)
  3. Os 5 modelos de busca
  4. A interface do Streamlit (demo)
Os notebooks (exploracao.ipynb e resultados.ipynb) reutilizam este arquivo.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

ARQUIVO = Path(__file__).parent / "products.csv"


# ============================================================
# 1. DADOS
# ============================================================
def carregar_corpus(caminho=ARQUIVO) -> pd.DataFrame:
    """Lê o CSV e cria uma coluna 'texto' = marca + categoria + descrição do produto."""
    df = pd.read_csv(caminho)
    # "AURELIA7 - Indianwear-Women" -> "Indianwear-Women" (tira o código do produto)
    df["Product Name"] = df["Product Name"].str.replace(r"\b[A-Za-z]+\d+\s+-\s+", "", regex=True)
    df["texto"] = (df["BrandName"].astype(str) + " . " + df["Product Name"].astype(str)
                   + " . " + df["Brand Desc"].astype(str))
    df["texto"] = df["texto"].str.replace(r"\s+", " ", regex=True).str.strip()

    df = df[df["texto"].str.len() >= 20]            # tira textos muito curtos
    df = df.drop_duplicates("texto")                # tira produtos repetidos
    df = df.sample(frac=1, random_state=42)         # embaralha (sempre do mesmo jeito)
    df = df.reset_index(drop=True)
    return df.rename(columns={"Brand Desc": "descricao", "BrandName": "marca", "SellPrice": "preco"})


# ============================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================
def tokenizar(texto: str, tirar_plural: bool = True) -> list:
    """'Solid Cotton Dresses' -> ['solid', 'cotton', 'dress']
    minúsculas, só letras/números, sem stopwords ('the', 'and'...) e sem plural."""
    palavras = re.findall(r"[a-z0-9]+", str(texto).lower())
    palavras = [p for p in palavras if p not in ENGLISH_STOP_WORDS and (len(p) > 1 or p.isdigit())]
    if tirar_plural:
        palavras = [_sem_plural(p) for p in palavras]
    return palavras


def _sem_plural(p: str) -> str:
    if len(p) > 4 and p.endswith("ies"):
        return p[:-3] + "y"
    if len(p) > 3 and p.endswith("s") and not p.endswith("ss"):
        return p[:-1]
    return p


def top_k(scores: np.ndarray, k: int, so_positivos: bool = True) -> list:
    """Pega os k maiores scores. Retorna [(posição_do_documento, score), ...]."""
    ordem = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in ordem if not so_positivos or scores[i] > 0]


# ============================================================
# 3. OS 5 MODELOS  (todos têm: indexar(textos) e buscar(consulta, k))
# ============================================================
class BuscaLinear:
    """MODELO 1 — Busca linear ingênua (baseline).
    Sem índice: a cada busca, percorre TODOS os textos contando a palavra. Custo O(N)."""
    nome = "Linear"

    def indexar(self, textos):
        self.textos = [t.lower() for t in textos]
        return self

    def buscar(self, consulta, k=10):
        palavras = tokenizar(consulta, tirar_plural=False)
        scores = np.zeros(len(self.textos))
        for i, texto in enumerate(self.textos):           # <- varredura sequencial
            scores[i] = sum(texto.count(p) for p in palavras)
        return top_k(scores, k)


class Booleano:
    """MODELO 2 — Booleano.
    Matriz de incidência: para cada palavra, um vetor de True/False (o documento tem a palavra?).
    A busca combina os vetores com operações bitwise:  &  (AND)   |  (OR)   ~  (NOT).
    Exemplo de consulta:  kurta AND cotton AND NOT printed   |   dress OR skirt
    Não existe ranking: o documento ou atende ou não atende."""
    nome = "Booleano"

    def indexar(self, textos):
        self.n = len(textos)
        vetorizador = CountVectorizer(tokenizer=tokenizar, lowercase=False, token_pattern=None, binary=True)
        matriz = vetorizador.fit_transform(textos)                 # documentos x palavras
        self.matriz_incidencia = matriz.T.toarray().astype(bool)   # palavras x documentos
        self.vocabulario = vetorizador.vocabulary_
        return self

    def _vetor_da_palavra(self, palavra):
        linha = self.vocabulario.get(palavra)
        if linha is None:                                 # palavra que não existe no corpus
            return np.zeros(self.n, dtype=bool)
        return self.matriz_incidencia[linha]

    def documentos_que_atendem(self, consulta):
        """Devolve as posições de TODOS os documentos que atendem a consulta."""
        resultado = np.zeros(self.n, dtype=bool)
        for grupo in re.split(r"\s+OR\s+", consulta):             # OR separa os grupos
            grupo_ok = np.ones(self.n, dtype=bool)
            for parte in re.split(r"\s+AND\s+", grupo):           # AND separa as partes
                parte = parte.strip()
                negar = parte.startswith("NOT ")
                if negar:
                    parte = parte[4:]
                vetor = np.ones(self.n, dtype=bool)
                for palavra in tokenizar(parte):                  # várias palavras juntas = AND
                    vetor = vetor & self._vetor_da_palavra(palavra)
                grupo_ok = grupo_ok & (~vetor if negar else vetor)
            resultado = resultado | grupo_ok
        return np.flatnonzero(resultado)

    def buscar(self, consulta, k=10):
        return [(int(i), 1.0) for i in self.documentos_que_atendem(consulta)[:k]]


class TfIdf:
    """MODELO 3 — Espaço vetorial clássico: TF-IDF + cosseno.
    TF  = quantas vezes a palavra aparece no documento
    IDF = quão rara a palavra é no corpus (palavra rara pesa mais)
    Cada documento vira um vetor; o score é o COSSENO entre a consulta e o documento."""
    nome = "TF-IDF"

    def indexar(self, textos):
        self.vetorizador = TfidfVectorizer(tokenizer=tokenizar, lowercase=False, token_pattern=None)
        self.matriz = self.vetorizador.fit_transform(textos)     # já vem normalizada (tamanho 1)
        return self

    def buscar(self, consulta, k=10):
        q = self.vetorizador.transform([consulta])
        scores = (self.matriz @ q.T).toarray().ravel()           # produto escalar = cosseno
        return top_k(scores, k)


class BM25:
    """MODELO 4 — Probabilístico Okapi BM25.
    score = soma, para cada palavra da consulta, de:
        IDF * tf*(k1+1) / ( tf + k1*(1 - b + b*tamanho_doc/tamanho_médio) )
    k1 -> saturação: repetir a palavra 10x não vale 10x mais que 1x.
    b  -> normalização: documentos longos são penalizados (b=0 ignora tamanho, b=1 penaliza total)."""
    nome = "BM25"

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def indexar(self, textos):
        self.vetorizador = CountVectorizer(tokenizer=tokenizar, lowercase=False, token_pattern=None)
        self.tf = self.vetorizador.fit_transform(textos).tocsc()      # documentos x palavras (contagens)
        self.n = self.tf.shape[0]
        df = np.diff(self.tf.indptr)                                  # em quantos docs cada palavra aparece
        self.idf = np.log((self.n - df + 0.5) / (df + 0.5) + 1)
        self.tamanho = np.asarray(self.tf.sum(axis=1)).ravel()        # nº de palavras de cada doc
        self.tamanho_medio = self.tamanho.mean()
        return self

    def buscar(self, consulta, k=10):
        scores = np.zeros(self.n)
        for palavra in tokenizar(consulta):
            j = self.vetorizador.vocabulary_.get(palavra)
            if j is None:
                continue
            tf = self.tf[:, j].toarray().ravel()                      # contagem da palavra em cada doc
            norm = self.k1 * (1 - self.b + self.b * self.tamanho / self.tamanho_medio)
            scores += self.idf[j] * tf * (self.k1 + 1) / (tf + norm)
        return top_k(scores, k)


class LSA:
    """MODELO 5 — LSA (Análise Semântica Latente).
    Pega a matriz TF-IDF e comprime com SVD para ~100 "conceitos".
    Palavras que aparecem nos mesmos contextos ficam próximas -> acha sinônimos."""
    nome = "LSA"

    def __init__(self, componentes=100):
        self.componentes = componentes

    def indexar(self, textos):
        self.vetorizador = TfidfVectorizer(tokenizer=tokenizar, lowercase=False,
                                           token_pattern=None, sublinear_tf=True)
        X = self.vetorizador.fit_transform(textos)
        n = max(2, min(self.componentes, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=n, random_state=42)
        self.docs = normalize(self.svd.fit_transform(X))              # documentos no espaço de conceitos
        return self

    def buscar(self, consulta, k=10):
        q = self.svd.transform(self.vetorizador.transform([consulta]))
        if not np.linalg.norm(q):                                     # nenhuma palavra conhecida
            return []
        scores = self.docs @ normalize(q).ravel()                     # cosseno no espaço de conceitos
        return top_k(scores, k, so_positivos=False)


MODELOS = {"Linear": BuscaLinear, "Booleano": Booleano, "TF-IDF": TfIdf, "BM25": BM25, "LSA": LSA}


# ============================================================
# 4. INTERFACE STREAMLIT (demo ao vivo)
# ============================================================
def main():
    import time
    import streamlit as st

    st.set_page_config(page_title="Busca E-commerce", page_icon="🔎", layout="wide")
    st.title("🔎 Modelos clássicos de Recuperação da Informação")
    st.caption("Domínio: E-commerce (moda, joias, calçados e perfumes) — dataset do Kaggle")

    @st.cache_resource(show_spinner="Carregando dados e indexando os 5 modelos...")
    def preparar():
        corpus = carregar_corpus()
        modelos = {nome: cls().indexar(corpus["texto"].tolist()) for nome, cls in MODELOS.items()}
        return corpus, modelos

    corpus, modelos = preparar()

    # ---- barra lateral ----
    st.sidebar.header("Configurações")
    k = st.sidebar.slider("Quantos resultados (Top-K)", 3, 20, 5)
    st.sidebar.subheader("Hiperparâmetros do BM25")
    modelos["BM25"].k1 = st.sidebar.slider("k1 (saturação)", 0.0, 3.0, 1.5, 0.1)
    modelos["BM25"].b = st.sidebar.slider("b (tamanho do documento)", 0.0, 1.0, 0.75, 0.05)
    st.sidebar.info(f"{len(corpus)} produtos indexados")

    # ---- busca ----
    consulta = st.text_input("Digite a busca", "cotton kurta")
    escolhidos = st.multiselect("Modelos para comparar", list(modelos), default=["TF-IDF", "BM25", "LSA"])
    st.caption("Booleano: use AND, OR, NOT em maiúsculas. Ex.: `kurta AND cotton AND NOT printed`")

    if consulta and escolhidos:
        colunas = st.columns(len(escolhidos))
        for coluna, nome in zip(colunas, escolhidos):
            inicio = time.perf_counter()
            resultados = modelos[nome].buscar(consulta, k)
            ms = (time.perf_counter() - inicio) * 1000
            with coluna:
                st.subheader(nome)
                st.caption(f"⏱️ {ms:.2f} ms")
                for i, score in resultados:
                    st.markdown(f"**{corpus.loc[i, 'descricao']}**  \n"
                                f"{corpus.loc[i, 'marca']} · Rs. {corpus.loc[i, 'preco']} · `score {score:.3f}`")
                if not resultados:
                    st.info("Nenhum resultado.")

    with st.expander("Como cada modelo funciona (resumo)"):
        st.markdown("""
- **Linear:** percorre todos os textos a cada busca (lento quando o corpus cresce).
- **Booleano:** vetores de True/False por palavra + AND/OR/NOT. Sem ranking.
- **TF-IDF:** palavras raras pesam mais; score = cosseno entre consulta e documento.
- **BM25:** TF-IDF melhorado, com saturação (`k1`) e penalização de documentos longos (`b`).
- **LSA:** comprime o TF-IDF em "conceitos"; encontra sinônimos (ex.: *perfume* ↔ *EDP*).
""")


if __name__ == "__main__":
    main()
