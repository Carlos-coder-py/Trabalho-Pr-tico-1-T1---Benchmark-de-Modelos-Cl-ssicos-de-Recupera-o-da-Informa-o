# Trabalho Prático 1 T1 Benchmark de Modelos Clássicos de Recuperação da Informação


# Busca em E-commerce: benchmark de modelos clássicos de Recuperação da Informação

Trabalho Prático 1 (T1) — Recuperação da Informação / PLN.

Comparamos **5 algoritmos clássicos de busca** sobre um catálogo real de produtos de e-commerce (moda, joias, calçados e perfumes), medindo **tempo de resposta** e **qualidade dos resultados**. O projeto inclui uma demo interativa em Streamlit.

## Sumário

- [Modelos implementados](#modelos-implementados)
- [Dataset](#dataset)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Como executar](#como-executar)
- [Como usar a demo](#como-usar-a-demo)
- [Resultados](#resultados)
- [Decisões e limitações](#decisões-e-limitações)
- [Integrantes](#integrantes)

## Modelos implementados

Todos ficam em `app.py` e seguem a mesma interface: `indexar(textos)` e `buscar(consulta, k)`.

| # | Modelo | Como funciona |
|---|--------|---------------|
| 1 | **Busca Linear** (baseline) | Percorre todos os textos a cada busca, sem índice. Custo O(N). |
| 2 | **Booleano** | Matriz de incidência binária (palavra × documento) com operações bitwise: `&` (AND), `\|` (OR), `~` (NOT). Não ordena os resultados. |
| 3 | **TF-IDF + cosseno** | Palavras raras pesam mais; o score é o cosseno entre os vetores da consulta e do documento. |
| 4 | **BM25 (Okapi)** | Implementação própria, com `k1` (saturação da frequência, padrão 1,5) e `b` (normalização pelo tamanho do documento, padrão 0,75). |
| 5 | **LSA** | SVD (`TruncatedSVD`, 100 componentes) sobre a matriz TF-IDF; a busca acontece no espaço de "conceitos". |

**Pré-processamento** (igual para todos): minúsculas, apenas letras e números, remoção de *stopwords* em inglês e remoção simples de plural (`dresses` → `dress`).

## Dataset

- **Fonte:** [Detailed Products Datasets (Kaggle)](https://www.kaggle.com/datasets/sujaykapadnis/products-datasets/data)
- **Arquivo usado:** `products.csv` (colunas selecionadas do dataset), com **4.566 produtos**.
- **Colunas usadas na busca:** `BrandName` (marca), `Product Name` (categoria) e `Brand Desc` (descrição). Cada produto vira um documento de texto.
- **Limpeza:** remoção do código do produto (ex.: `AURELIA7 - `), de textos muito curtos e de **410 produtos repetidos**.
- **Corpus final:** **4.156 documentos**, com cerca de 13 palavras cada.

## Estrutura do projeto

```
trabalho-ri/
├── app.py              # Código principal: dados, pré-processamento, 5 modelos e demo Streamlit
├── exploracao.ipynb    # Exploração do dataset (tabelas, gráficos e conclusões)
├── resultados.ipynb    # Benchmark de tempo e comparação de relevância
├── products.csv        # Dados
├── requirements.txt    # Bibliotecas
└── README.md
```

Os dois notebooks importam os modelos de `app.py`, então o código não é duplicado.

## Instalação

Requer Python 3.9 ou superior.

```bash
# 1. Entre na pasta do projeto
cd trabalho-ri

# 2. Crie e ative um ambiente virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Instale as dependências
python3 -m pip install -r requirements.txt
```

## Como executar

**Demo interativa (Streamlit):**

```bash
python3 -m streamlit run app.py
```

O navegador abre em `http://localhost:8501`. Na primeira carga, o app leva alguns segundos indexando os 5 modelos.

**Notebooks:** abra `exploracao.ipynb` e `resultados.ipynb` no VS Code ou rode `jupyter notebook`. Ambos já vêm executados; para reproduzir, use *Run All*.

> Execute os comandos dentro da pasta do projeto: `products.csv` precisa estar ao lado de `app.py`.

## Como usar a demo

- **Campo de busca:** digite a consulta e aperte Enter.
- **Modelos para comparar:** escolha quais modelos aparecem lado a lado (padrão: TF-IDF, BM25 e LSA). Cada coluna mostra os produtos, o score e o tempo em ms.
- **Barra lateral:** ajusta o Top-K e os hiperparâmetros `k1` e `b` do BM25.
- **Booleano:** use `AND`, `OR` e `NOT` em maiúsculas. Exemplo: `kurta AND cotton AND NOT printed`.

Buscas sugeridas: `cotton kurta`, `perfume`, `set`, `shrug`, `dress`.

## Resultados

Detalhes e gráficos em `resultados.ipynb`.

### Escalabilidade

Tempo médio por consulta (ms), com 20 consultas iguais para todos os modelos:

| N | Linear | Booleano | TF-IDF | BM25 | LSA |
|---|-------:|---------:|-------:|-----:|----:|
| 100 | 0,047 | 0,010 | 0,210 | 0,109 | 0,285 |
| 500 | 0,224 | 0,011 | 0,234 | 0,116 | 0,391 |
| 1.000 | 0,440 | 0,013 | 0,259 | 0,122 | 0,474 |
| 4.156 | 1,835 | 0,020 | 0,408 | 0,159 | 0,783 |

- Com o corpus 41 vezes maior, o **Linear ficou cerca de 39 vezes mais lento** (comportamento O(N)).
- Os demais modelos ficaram, no máximo, cerca de 3 vezes mais lentos.
- Na indexação, o **LSA é o mais caro** (cerca de 100 ms, contra 30 a 40 ms dos outros), por causa do SVD.

Os tempos variam de computador para computador; o formato das curvas é o que importa.

### Relevância (Top-5)

| Consulta | Dificuldade | TF-IDF e BM25 | LSA |
|----------|-------------|---------------|-----|
| `perfume` | sinônimo | Só produtos que têm a palavra | Também perfumes "EDP" que não têm a palavra |
| `set` | palavra ambígua | Joias ("bracelet set") | Mistura sentidos (prendedores de cabelo, conjuntos de roupa) |
| `shrug` | termo raro | Exatamente os 2 produtos existentes | Os 2 primeiro, depois itens só parecidos |
| `dress` | termo frequente | Vestidos, com ordem arbitrária entre os modelos | Vestidos, com ordem diferente |
| `cotton kurta` | duas palavras | Kurtas de algodão | Kurtas de algodão |

### Conclusão

| Modelo | Veredito |
|--------|----------|
| Linear | Só como referência: o tempo cresce com o corpus |
| Booleano | Filtros exatos; não ordena os resultados |
| TF-IDF | Ranking simples e rápido |
| BM25 | Melhor equilíbrio entre velocidade e precisão |
| LSA | Bom para sinônimos; mais lento e menos preciso |

## Decisões e limitações

- **Tamanho do corpus:** o enunciado sugere até 10.000 documentos, mas o dataset tem menos que isso. Por isso o benchmark usa N = 100, 500, 1.000 e 4.156 (o total).
- **Consultas do benchmark:** geradas automaticamente a partir das palavras do menor subconjunto, para que existam em todos os tamanhos. O Booleano recebe as palavras unidas por `AND`.
- **Relevância:** avaliada qualitativamente, olhando o Top-5 de 5 consultas escolhidas. Não há avaliação formal com pessoas (precision@k, nDCG), o que seria o próximo passo.
- **Idioma:** o catálogo está em inglês; o pré-processamento usa *stopwords* e regras de plural em inglês.

## Integrantes

- Carlos, Melissa e Arthur
