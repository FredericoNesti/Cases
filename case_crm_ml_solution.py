"""
Case CRM - Solucao de IA / Machine Learning
Autor: Antonio Frederico / ChatGPT

Objetivo:
- Criar uma base ficticia inspirada no enunciado do Case CRM.
- Construir features de cliente, segmento e historico.
- Treinar um modelo de propensao de conversao.
- Recomendar a melhor campanha/canal/categoria por cliente.

Como rodar:
    python case_crm_ml_solution.py

Dependencias minimas:
    pip install pandas numpy scikit-learn joblib

Dependencias opcionais:
    pip install fastapi uvicorn
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
HAS_LIGHTGBM = False



@dataclass
class Config:
    n_clientes: int = 2500
    n_interacoes_campanha: int = 8000
    n_clusters: int = 5
    random_state: int = 42
    output_dir: str = "outputs_case_crm"


CATEGORIAS = ["AC", "RDI", "AP"]
CAMPANHAS = ["Outubro Rosa", "Check-up", "Marco Lilas", "Vacina", "Prevencao Cardiometabolica"]
CANAIS = ["WhatsApp", "Email", "SMS"]
MARCAS = ["Delboni", "Lavoisier", "Salomao Zoppi"]


def make_one_hot_encoder() -> OneHotEncoder:
    """Cria OneHotEncoder compativel com versoes antigas e novas do sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def gerar_clientes(cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)
    n = cfg.n_clientes

    clientes = pd.DataFrame({
        "cliente_id": np.arange(1, n + 1),
        "idade": rng.integers(18, 80, n),
        "sexo": rng.choice(["Feminino", "Masculino"], n, p=[0.56, 0.44]),
        "cidade": rng.choice(["Sao Paulo", "Campinas", "Santos", "Osasco"], n, p=[0.65, 0.15, 0.10, 0.10]),
        "forma_pagamento_padrao": rng.choice(["Convenio", "Particular"], n, p=[0.72, 0.28]),
        "marca_preferida": rng.choice(MARCAS, n, p=[0.50, 0.30, 0.20]),
        "qtd_exames_12m": rng.poisson(2.2, n),
        "ticket_medio": np.clip(rng.normal(260, 90, n), 40, 900),
        "dias_desde_ultimo_exame": rng.integers(1, 730, n),
        "qtd_campanhas_12m": rng.poisson(4, n),
        "taxa_conversao_whatsapp": rng.beta(2.2, 5.0, n),
        "taxa_conversao_email": rng.beta(1.5, 6.0, n),
        "taxa_conversao_sms": rng.beta(1.2, 7.0, n),
    })

    # Afinidades por categoria. A afinidade RDI e levemente maior em mulheres 40+ para simular Outubro Rosa.
    clientes["afinidade_AC"] = np.clip(rng.beta(2.0, 3.5, n) + 0.05 * (clientes["idade"] > 35), 0, 1)
    clientes["afinidade_RDI"] = np.clip(
        rng.beta(1.8, 3.5, n)
        + 0.22 * ((clientes["sexo"] == "Feminino") & (clientes["idade"] >= 40)),
        0,
        1,
    )
    clientes["afinidade_AP"] = np.clip(rng.beta(1.5, 4.0, n) + 0.04 * (clientes["idade"] < 35), 0, 1)

    clientes["taxa_conversao_geral"] = (
        0.45 * clientes["taxa_conversao_whatsapp"]
        + 0.35 * clientes["taxa_conversao_email"]
        + 0.20 * clientes["taxa_conversao_sms"]
    )

    return clientes


def segmentar_clientes(clientes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    features_cluster = [
        "idade", "qtd_exames_12m", "ticket_medio", "dias_desde_ultimo_exame",
        "qtd_campanhas_12m", "taxa_conversao_whatsapp", "taxa_conversao_email",
        "taxa_conversao_sms", "afinidade_AC", "afinidade_RDI", "afinidade_AP",
    ]
    X = StandardScaler().fit_transform(clientes[features_cluster])
    kmeans = KMeans(n_clusters=cfg.n_clusters, random_state=cfg.random_state, n_init=10)
    clientes = clientes.copy()
    clientes["cluster_cliente"] = kmeans.fit_predict(X).astype(str)
    return clientes


def gerar_interacoes_campanha(clientes: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state + 1)
    interacoes = pd.DataFrame({
        "cliente_id": rng.choice(clientes["cliente_id"], cfg.n_interacoes_campanha),
        "campanha": rng.choice(CAMPANHAS, cfg.n_interacoes_campanha),
        "canal": rng.choice(CANAIS, cfg.n_interacoes_campanha, p=[0.50, 0.35, 0.15]),
        "categoria": rng.choice(CATEGORIAS, cfg.n_interacoes_campanha, p=[0.40, 0.35, 0.25]),
        "mes": rng.integers(1, 13, cfg.n_interacoes_campanha),
    })

    base = interacoes.merge(clientes, on="cliente_id", how="left")

    afinidade_categoria = np.select(
        [base["categoria"].eq("AC"), base["categoria"].eq("RDI"), base["categoria"].eq("AP")],
        [base["afinidade_AC"], base["afinidade_RDI"], base["afinidade_AP"]],
        default=0.0,
    )

    score_canal = np.select(
        [base["canal"].eq("WhatsApp"), base["canal"].eq("Email"), base["canal"].eq("SMS")],
        [base["taxa_conversao_whatsapp"], base["taxa_conversao_email"], base["taxa_conversao_sms"]],
        default=0.0,
    )

    efeito_outubro_rosa = (
        base["campanha"].eq("Outubro Rosa")
        & base["sexo"].eq("Feminino")
        & base["categoria"].eq("RDI")
        & base["mes"].isin([9, 10, 11])
        & (base["idade"] >= 40)
    ).astype(float)

    efeito_checkup = (
        base["campanha"].eq("Check-up")
        & base["categoria"].eq("AC")
        & (base["idade"] >= 30)
    ).astype(float)

    # Formula latente criada apenas para mocar a realidade.
    linear = (
        -1.70
        + 1.80 * afinidade_categoria
        + 1.20 * score_canal
        + 0.10 * base["qtd_exames_12m"]
        - 0.0012 * base["dias_desde_ultimo_exame"]
        + 0.45 * efeito_outubro_rosa
        + 0.25 * efeito_checkup
        + 0.15 * (base["forma_pagamento_padrao"].eq("Convenio")).astype(float)
    )

    prob = 1 / (1 + np.exp(-linear))
    base["prob_real_simulada"] = prob
    base["converteu"] = rng.binomial(1, np.clip(prob, 0.02, 0.95))
    base["dias_ate_conversao"] = np.where(base["converteu"].eq(1), rng.integers(1, 31, len(base)), np.nan)

    return base


def treinar_modelo_propensao(base: pd.DataFrame, cfg: Config) -> Pipeline:
    cat_cols = [
        "sexo", "cidade", "forma_pagamento_padrao", "marca_preferida",
        "campanha", "canal", "categoria", "cluster_cliente",
    ]
    num_cols = [
        "idade", "qtd_exames_12m", "ticket_medio", "dias_desde_ultimo_exame",
        "qtd_campanhas_12m", "taxa_conversao_whatsapp", "taxa_conversao_email",
        "taxa_conversao_sms", "taxa_conversao_geral", "afinidade_AC", "afinidade_RDI",
        "afinidade_AP", "mes",
    ]

    X = base[cat_cols + num_cols]
    y = base["converteu"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=cfg.random_state, stratify=y
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", make_one_hot_encoder(), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    clf = LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs")

    model = Pipeline([("preprocess", preprocess), ("clf", clf)])
    model.fit(X_train, y_train)

    pred_proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    print("Modelo usado: LogisticRegression")
    print("AUC ROC:", round(roc_auc_score(y_test, pred_proba), 4))
    print(classification_report(y_test, pred))

    return model


def montar_opcoes_campanha() -> pd.DataFrame:
    return pd.DataFrame([
        {"campanha": "Outubro Rosa", "canal": "WhatsApp", "categoria": "RDI", "mes": 10},
        {"campanha": "Outubro Rosa", "canal": "Email", "categoria": "RDI", "mes": 10},
        {"campanha": "Check-up", "canal": "WhatsApp", "categoria": "AC", "mes": 1},
        {"campanha": "Check-up", "canal": "Email", "categoria": "AC", "mes": 1},
        {"campanha": "Marco Lilas", "canal": "WhatsApp", "categoria": "AP", "mes": 3},
        {"campanha": "Vacina", "canal": "SMS", "categoria": "AP", "mes": 4},
        {"campanha": "Prevencao Cardiometabolica", "canal": "Email", "categoria": "AC", "mes": 8},
    ])


def recomendar_melhor_campanha(
    cliente_id: int,
    clientes: pd.DataFrame,
    model: Pipeline,
    top_n: int = 5,
) -> pd.DataFrame:
    cliente = clientes.loc[clientes["cliente_id"].eq(cliente_id)].copy()
    if cliente.empty:
        raise ValueError(f"cliente_id {cliente_id} nao encontrado")

    opcoes = montar_opcoes_campanha()
    simulacoes = cliente.merge(opcoes, how="cross")

    cols_modelo = [
        "sexo", "cidade", "forma_pagamento_padrao", "marca_preferida",
        "campanha", "canal", "categoria", "cluster_cliente",
        "idade", "qtd_exames_12m", "ticket_medio", "dias_desde_ultimo_exame",
        "qtd_campanhas_12m", "taxa_conversao_whatsapp", "taxa_conversao_email",
        "taxa_conversao_sms", "taxa_conversao_geral", "afinidade_AC", "afinidade_RDI",
        "afinidade_AP", "mes",
    ]

    simulacoes["score_conversao"] = model.predict_proba(simulacoes[cols_modelo])[:, 1]
    return simulacoes.sort_values("score_conversao", ascending=False).head(top_n)[
        [
            "cliente_id", "idade", "sexo", "forma_pagamento_padrao", "marca_preferida",
            "cluster_cliente", "campanha", "canal", "categoria", "mes", "score_conversao",
        ]
    ]


def gerar_personas(clientes: pd.DataFrame) -> pd.DataFrame:
    resumo = clientes.groupby("cluster_cliente").agg(
        qtd_clientes=("cliente_id", "count"),
        idade_media=("idade", "mean"),
        ticket_medio=("ticket_medio", "mean"),
        qtd_exames_12m=("qtd_exames_12m", "mean"),
        afinidade_AC=("afinidade_AC", "mean"),
        afinidade_RDI=("afinidade_RDI", "mean"),
        afinidade_AP=("afinidade_AP", "mean"),
        taxa_whatsapp=("taxa_conversao_whatsapp", "mean"),
        taxa_email=("taxa_conversao_email", "mean"),
        taxa_sms=("taxa_conversao_sms", "mean"),
    ).reset_index()

    resumo["pct_base"] = resumo["qtd_clientes"] / resumo["qtd_clientes"].sum()

    def nomear_persona(row: pd.Series) -> str:
        afinidades = {"AC": row["afinidade_AC"], "RDI": row["afinidade_RDI"], "AP": row["afinidade_AP"]}
        melhor_cat = max(afinidades, key=afinidades.get)
        canal = max(
            {"WhatsApp": row["taxa_whatsapp"], "Email": row["taxa_email"], "SMS": row["taxa_sms"]},
            key={"WhatsApp": row["taxa_whatsapp"], "Email": row["taxa_email"], "SMS": row["taxa_sms"]}.get,
        )
        if row["idade_media"] >= 45 and melhor_cat == "RDI":
            return f"Preventivo imagem - {canal}"
        if melhor_cat == "AC":
            return f"Check-up laboratorial - {canal}"
        if row["qtd_exames_12m"] < 1.5:
            return f"Reativacao baixa frequencia - {canal}"
        return f"Afinidade {melhor_cat} - {canal}"

    resumo["persona"] = resumo.apply(nomear_persona, axis=1)
    return resumo.sort_values("pct_base", ascending=False)


def main() -> None:
    cfg = Config()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    clientes = gerar_clientes(cfg)
    clientes = segmentar_clientes(clientes, cfg)
    base = gerar_interacoes_campanha(clientes, cfg)
    personas = gerar_personas(clientes)
    model = treinar_modelo_propensao(base, cfg)

    recomendacoes = pd.concat(
        [recomendar_melhor_campanha(int(cid), clientes, model, top_n=3) for cid in clientes["cliente_id"].head(10)],
        ignore_index=True,
    )

    clientes.to_csv(out / "clientes_mocados.csv", index=False)
    base.to_csv(out / "base_campanhas_mocada.csv", index=False)
    personas.to_csv(out / "personas_clusters.csv", index=False)
    recomendacoes.to_csv(out / "recomendacoes_exemplo.csv", index=False)
    joblib.dump(model, out / "modelo_propensao.joblib")

    metadata = {
        "modelo": "LogisticRegression",
        "n_clientes": cfg.n_clientes,
        "n_interacoes_campanha": cfg.n_interacoes_campanha,
        "n_clusters": cfg.n_clusters,
        "arquivos_gerados": sorted(os.listdir(out)),
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nArquivos gerados em:", out.resolve())
    print("\nPersonas:")
    print(personas.head())
    print("\nRecomendacoes exemplo:")
    print(recomendacoes.head(10))


if __name__ == "__main__":
    main()
