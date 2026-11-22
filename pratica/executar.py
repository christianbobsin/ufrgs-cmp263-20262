#!/usr/bin/env python3
"""Auditoria, regressão e avaliação de energia sem notebooks."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV, GroupKFold, GroupShuffleSplit, RepeatedKFold,
    cross_validate, train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

BASE = Path(__file__).resolve().parent
TARGET = 'energy_mJ'
FEATURES = [f'arch_op_{src}_{dst}' for dst in range(1, 4) for src in range(dst)]
OPERATIONS = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
SCORING = {'mae': 'neg_mean_absolute_error', 'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    test_size: float = 0.20
    n_splits: int = 5
    n_repeats: int = 3
    n_jobs: int = 2
    division: str = 'composicao'


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def reconstruct_architecture(row):
    return '+'.join(
        '|' + '|'.join(f"{row[f'arch_op_{src}_{dst}']}~{src}" for src in range(dst)) + '|'
        for dst in range(1, 4)
    )


def load_dataset(path):
    df = pd.read_csv(path, keep_default_na=False)
    required = {'arch_idx', 'arch_str', 'board', TARGET, *FEATURES}
    if not required.issubset(df.columns):
        raise ValueError(f'Colunas ausentes: {sorted(required - set(df.columns))}')
    if df['arch_idx'].duplicated().any() or df['arch_str'].duplicated().any():
        raise ValueError('Há arquiteturas duplicadas; revisar a unidade de análise.')
    for col in FEATURES:
        if not set(df[col]).issubset(OPERATIONS):
            raise ValueError(f'Categorias inválidas ou ausentes em {col}.')
    df[TARGET] = pd.to_numeric(df[TARGET], errors='raise')
    if not (np.isfinite(df[TARGET]).all() and df[TARGET].gt(0).all()):
        raise ValueError('Energia não finita ou não positiva; investigar a aquisição.')
    if df['board'].eq('').any():
        raise ValueError('Metadado board ausente.')
    if not df.apply(reconstruct_architecture, axis=1).eq(df['arch_str']).all():
        raise ValueError('As features não reproduzem arch_str.')
    missing = df.isna().sum() + df.eq('').sum()
    audit = {
        'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'n_rows': len(df), 'n_columns': len(df.columns),
        'missing_by_column': {k: int(v) for k, v in missing.items()},
        'constant_columns': [c for c in df if df[c].nunique(dropna=False) <= 1],
        'duplicate_arch_idx': 0, 'duplicate_arch_str': 0,
        'architecture_reconstruction': True,
    }
    return df, audit


def composition_groups(df):
    """Mantém juntas permutações das mesmas quantidades de operações."""
    counts = pd.DataFrame({op: df[FEATURES].eq(op).sum(axis=1) for op in OPERATIONS})
    return counts.astype(str).agg('-'.join, axis=1)


def split_dataset(df, config):
    groups = composition_groups(df)
    if config.division == 'composicao':
        dev_idx, test_idx = next(GroupShuffleSplit(
            n_splits=1, test_size=config.test_size, random_state=config.seed,
        ).split(df[FEATURES], groups=groups))
        if set(groups.iloc[dev_idx]) & set(groups.iloc[test_idx]):
            raise AssertionError('Grupo compartilhado entre desenvolvimento e teste.')
    else:
        dev_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=config.test_size,
            random_state=config.seed, stratify=df['board'],
        )
    if set(dev_idx) & set(test_idx) or len(dev_idx) + len(test_idx) != len(df):
        raise AssertionError('Divisão inconsistente.')
    dev = df.iloc[dev_idx].copy()
    X, y, dev_groups = dev[FEATURES], dev[TARGET], groups.iloc[dev_idx]
    if config.division == 'composicao':
        if dev_groups.nunique() < config.n_splits:
            raise ValueError('Número de grupos insuficiente para os folds solicitados.')
        cv = []
        for repeat in range(config.n_repeats):
            cv.extend(GroupKFold(
                n_splits=config.n_splits, shuffle=True, random_state=config.seed + repeat,
            ).split(X, y, groups=dev_groups))
        for train, valid in cv:
            if set(dev_groups.iloc[train]) & set(dev_groups.iloc[valid]):
                raise AssertionError('Grupo compartilhado entre treino e validação.')
    else:
        cv = list(RepeatedKFold(
            n_splits=config.n_splits, n_repeats=config.n_repeats, random_state=config.seed,
        ).split(X, y))
    return dev_idx, test_idx, cv, groups


def record_setup(df, audit, config, out, dev_idx, test_idx, cv, groups):
    out.mkdir(parents=True, exist_ok=True)
    # Não misturar experimentos com dados ou protocolos diferentes na mesma pasta.
    manifest = {'config': asdict(config), 'target': TARGET, 'features': FEATURES,
                'data_sha256': audit['sha256']}
    prior_path = out / 'configuracao.json'
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())
        if prior != manifest:
            raise ValueError('Esta saída contém outro protocolo. Escolha outra pasta com --saida.')
    save_json(prior_path, manifest)
    save_json(out / 'auditoria_dataset.json', audit)
    save_json(out / 'ambiente.json', {
        'python': sys.version, 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ['numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn', 'joblib']},
    })
    partitions = pd.concat([
        df.iloc[idx][['arch_idx', 'board']].assign(
            grupo_composicao=groups.iloc[idx].to_numpy(), subconjunto=name,
            ordem_subset=np.arange(len(idx)),
        ) for name, idx in [('desenvolvimento', dev_idx), ('teste', test_idx)]
    ]).rename_axis('linha_csv').reset_index()
    partitions.to_csv(out / 'divisao_dados.csv', index=False)
    records = []
    dev = df.iloc[dev_idx]
    for fold_id, (train, valid) in enumerate(cv):
        assert not set(train) & set(valid)
        records.append(pd.DataFrame({
            'posicao_desenvolvimento': valid, 'arch_idx': dev.iloc[valid]['arch_idx'].to_numpy(),
            'grupo_composicao': groups.iloc[dev_idx].iloc[valid].to_numpy(),
            'repeticao': fold_id // config.n_splits + 1,
            'fold_validacao': fold_id % config.n_splits + 1,
        }))
    pd.concat(records).to_csv(out / 'folds_cv.csv', index=False)
    save_json(out / 'resumo_divisao.json', {
        'n_development': len(dev_idx), 'n_test': len(test_idx), 'n_cv_folds': len(cv),
        'groups_total': int(groups.nunique()),
        'groups_development': int(groups.iloc[dev_idx].nunique()),
        'groups_test': int(groups.iloc[test_idx].nunique()),
        'group_overlap': len(set(groups.iloc[dev_idx]) & set(groups.iloc[test_idx])),
        'group_order': OPERATIONS,
    })
    print(f"Base validada: {len(df)} arquiteturas; {groups.nunique()} grupos de composição.")
    print(f'Desenvolvimento: {len(dev_idx)}; teste reservado: {len(test_idx)}; folds: {len(cv)}.')
    print(pd.crosstab(partitions['board'], partitions['subconjunto']).to_string())


def exploratory_analysis(dev, out):
    fig_dir = out / 'figuras'
    fig_dir.mkdir(exist_ok=True)
    y = dev[TARGET]
    description = y.describe(percentiles=[.01, .05, .25, .50, .75, .95, .99])
    description.to_csv(out / 'energia_descritiva_desenvolvimento.csv')
    dev.groupby('board')[TARGET].agg(['count', 'mean', 'median', 'std', 'min', 'max']).to_csv(
        out / 'energia_por_placa_desenvolvimento.csv')
    q1, q3 = y.quantile([.25, .75])
    flagged = dev.loc[(y < q1 - 1.5 * (q3 - q1)) | (y > q3 + 1.5 * (q3 - q1))]
    flagged[['arch_idx', TARGET, 'board']].to_csv(out / 'extremos_iqr_desenvolvimento.csv', index=False)
    print(f'EDA no desenvolvimento: {len(flagged)} extremos sinalizados por IQR; nenhuma linha removida.')
    print(description.to_string())
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(y, bins=45, ax=axes[0])
    axes[0].set(xlabel='Energia por inferência (mJ)', ylabel='Arquiteturas', title='Desenvolvimento')
    sns.boxplot(data=dev, x='board', y=TARGET, ax=axes[1])
    axes[1].set(xlabel='Unidade de placa', ylabel='Energia por inferência (mJ)')
    axes[1].tick_params(axis='x', rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / 'energia_desenvolvimento.png', dpi=180)
    plt.close(fig)
    frequencies = pd.DataFrame({col: dev[col].value_counts() for col in FEATURES}).fillna(0).astype(int)
    frequencies.to_csv(out / 'frequencias_operacoes_desenvolvimento.csv')
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(frequencies, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set(xlabel='Conexão da célula', ylabel='Operação', title='Frequências no desenvolvimento')
    fig.tight_layout()
    fig.savefig(fig_dir / 'operacoes_desenvolvimento.png', dpi=180)
    plt.close(fig)


def make_models(config):
    models = {
        'baseline_mediana': DummyRegressor(strategy='median'),
        'ridge': Ridge(alpha=1.0),
        'knn': KNeighborsRegressor(n_neighbors=5, n_jobs=1),
        'arvore': DecisionTreeRegressor(random_state=config.seed, max_depth=10, min_samples_leaf=5),
        'floresta': RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                        random_state=config.seed, n_jobs=1),
        'svr': SVR(kernel='rbf', C=10.0, epsilon=0.1),
    }
    pipelines = {
        name: Pipeline([('encoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                        ('model', model)]) for name, model in models.items()
    }
    grids = {
        'ridge': {'model__alpha': [0.1, 1.0, 10.0, 100.0]},
        'knn': {'model__n_neighbors': [5, 11, 21], 'model__weights': ['uniform', 'distance']},
        'arvore': {'model__max_depth': [5, 10, 15], 'model__min_samples_leaf': [5, 15, 30]},
        'floresta': {'model__max_depth': [8, 15], 'model__min_samples_leaf': [3, 10, 20]},
        'svr': {'model__C': [1.0, 10.0, 100.0], 'model__epsilon': [0.1, 0.5],
                'model__gamma': ['scale', 0.1]},
    }
    return pipelines, grids


def compare_models(pipelines, X, y, cv, config, out):
    records = []
    for name, pipe in pipelines.items():
        print(f'Comparando {name}...', flush=True)
        scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=config.n_jobs,
                                return_train_score=True, error_score='raise')
        for i in range(len(cv)):
            records.append({
                'modelo': name, 'repeticao': i // config.n_splits + 1, 'fold': i % config.n_splits + 1,
                'mae_mJ': -scores['test_mae'][i], 'rmse_mJ': -scores['test_rmse'][i],
                'r2': scores['test_r2'][i], 'mae_treino_mJ': -scores['train_mae'][i],
                'tempo_fit_s': scores['fit_time'][i],
            })
    folds = pd.DataFrame(records)
    folds['gap_mae_mJ'] = folds['mae_mJ'] - folds['mae_treino_mJ']
    folds.to_csv(out / 'comparacao_scores_folds.csv', index=False)
    summary = folds.groupby('modelo').agg(
        mae_media_mJ=('mae_mJ', 'mean'), mae_desvio_mJ=('mae_mJ', 'std'),
        rmse_media_mJ=('rmse_mJ', 'mean'), r2_medio=('r2', 'mean'),
        mae_treino_media_mJ=('mae_treino_mJ', 'mean'), gap_mae_medio_mJ=('gap_mae_mJ', 'mean'),
        tempo_fit_medio_s=('tempo_fit_s', 'mean'),
    ).sort_values('mae_media_mJ')
    summary.to_csv(out / 'comparacao_resumo.csv')
    print(summary.to_string())
    return summary


def tune_models(pipelines, grids, summary, X, y, cv, config, out, audit):
    candidates = [name for name in summary.index if name != 'baseline_mediana'][:2]
    searches, records = {}, []
    for name in candidates:
        print(f'Ajustando {name}...', flush=True)
        search = GridSearchCV(pipelines[name], grids[name], scoring=SCORING, refit='mae',
                              cv=cv, n_jobs=config.n_jobs, error_score='raise', return_train_score=True)
        started = time.perf_counter()
        search.fit(X, y)
        searches[name] = search
        pd.DataFrame(search.cv_results_).to_csv(out / f'ajuste_{name}.csv', index=False)
        records.append({'modelo': name, 'mae_cv_mJ': float(-search.best_score_),
                        'parametros': search.best_params_, 'tempo_busca_s': time.perf_counter() - started})
    baseline_mae = float(summary.loc['baseline_mediana', 'mae_media_mJ'])
    best_name = max(searches, key=lambda name: searches[name].best_score_)
    if baseline_mae <= -searches[best_name].best_score_:
        name = 'baseline_mediana'
        winner = clone(pipelines[name]).fit(X, y)
        params, selected_mae = {'model__strategy': 'median'}, baseline_mae
    else:
        name, winner = best_name, searches[best_name].best_estimator_
        params, selected_mae = searches[name].best_params_, float(-searches[name].best_score_)
    selection = {'buscas': records, 'modelo_selecionado': name, 'parametros': params,
                 'mae_cv_mJ': selected_mae, 'data_sha256': audit['sha256'], 'config': asdict(config)}
    save_json(out / 'selecao_desenvolvimento.json', selection)
    joblib.dump(winner, out / 'modelo.joblib')
    print(f'Selecionado no desenvolvimento: {name}; MAE CV {selected_mae:.4f} mJ.')
    return winner


def interpret_model(winner, X, y, config, out):
    # Diagnóstico no desenvolvimento; não é estimativa de generalização.
    importance = permutation_importance(winner, X, y, scoring=SCORING['mae'],
                                        n_repeats=10, random_state=config.seed, n_jobs=config.n_jobs)
    frame = pd.DataFrame({'conexao': FEATURES, 'aumento_mae_medio_mJ': importance.importances_mean,
                          'desvio_permutacoes_mJ': importance.importances_std}).sort_values(
                              'aumento_mae_medio_mJ', ascending=False)
    frame.to_csv(out / 'importancia_permutacao_desenvolvimento.csv', index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(frame['conexao'], frame['aumento_mae_medio_mJ'], xerr=frame['desvio_permutacoes_mJ'])
    ax.invert_yaxis()
    ax.set(xlabel='Aumento da MAE após permutação (mJ)', title='Diagnóstico no desenvolvimento')
    fig.tight_layout()
    fig.savefig(out / 'figuras' / 'importancia_permutacao.png', dpi=180)
    plt.close(fig)


def evaluate_final(df, audit, out):
    if (out / 'avaliacao_final.json').exists():
        raise ValueError('Teste já avaliado nesta pasta. Consulte os resultados salvos.')
    selection_path = out / 'selecao_desenvolvimento.json'
    if not selection_path.exists() or not (out / 'modelo.joblib').exists():
        raise ValueError('Execute treinar antes de avaliar.')
    selection = json.loads(selection_path.read_text())
    if selection['data_sha256'] != audit['sha256']:
        raise ValueError('O dataset difere do usado para treinar.')
    config = Config(**selection['config'])
    dev_idx, test_idx, _, _ = split_dataset(df, config)
    winner = joblib.load(out / 'modelo.joblib')
    X_test, y_test = df.iloc[test_idx][FEATURES], df.iloc[test_idx][TARGET]
    pred = winner.predict(X_test)
    baseline = clone(make_models(config)[0]['baseline_mediana']).fit(
        df.iloc[dev_idx][FEATURES], df.iloc[dev_idx][TARGET])
    metrics = {}
    for name, predictions in [('selecionado', pred), ('baseline_mediana', baseline.predict(X_test))]:
        metrics[name] = {'mae_mJ': float(mean_absolute_error(y_test, predictions)),
                         'rmse_mJ': float(root_mean_squared_error(y_test, predictions)),
                         'r2': float(r2_score(y_test, predictions))}
    predictions = df.iloc[test_idx][['arch_idx', 'board', TARGET]].copy()
    predictions['predicao_mJ'] = pred
    predictions['residuo_mJ'] = y_test.to_numpy() - pred
    predictions['erro_absoluto_mJ'] = np.abs(predictions['residuo_mJ'])
    predictions.to_csv(out / 'predicoes_teste.csv', index=False)
    predictions.groupby('board')['erro_absoluto_mJ'].agg(['count', 'mean', 'median', 'max']).to_csv(
        out / 'erros_por_placa_teste.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_test, pred, s=9, alpha=.35)
    low, high = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    axes[0].plot([low, high], [low, high], '--', color='black')
    axes[0].set(xlabel='Energia medida (mJ)', ylabel='Energia prevista (mJ)')
    axes[1].scatter(pred, predictions['residuo_mJ'], s=9, alpha=.35)
    axes[1].axhline(0, linestyle='--', color='black')
    axes[1].set(xlabel='Energia prevista (mJ)', ylabel='Resíduo: medida − prevista (mJ)')
    fig.tight_layout()
    fig.savefig(out / 'figuras' / 'avaliacao_final.png', dpi=180)
    plt.close(fig)
    save_json(out / 'avaliacao_final.json', {'modelo_selecionado': selection['modelo_selecionado'],
              'n_test': len(test_idx), 'metricas': metrics, 'data_sha256': audit['sha256']})
    print(pd.DataFrame(metrics).T.to_string())


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, help_text in [('auditar', 'Validar, dividir e fazer EDA; não treina modelos.'),
                               ('treinar', 'Comparar cinco modelos e baseline, ajustar e interpretar.'),
                               ('avaliar', 'Avaliar o vencedor no teste reservado após fechar as escolhas.')]:
        sub = commands.add_parser(command, help=help_text)
        sub.add_argument('--dados', type=Path, default=BASE / 'dados' / 'hw_metrics_arch_features.csv')
        sub.add_argument('--saida', type=Path, default=BASE / 'resultados' / 'energia_composicao')
        if command != 'avaliar':
            sub.add_argument('--divisao', choices=['composicao', 'aleatoria'], default='composicao',
                             help='Composição mantém juntas permutações das mesmas operações.')
            sub.add_argument('--seed', type=int, default=42)
            sub.add_argument('--teste', type=float, default=0.20, help='Fração de grupos no protocolo composição.')
            sub.add_argument('--folds', type=int, default=5)
            sub.add_argument('--repeticoes', type=int, default=3)
            sub.add_argument('--jobs', type=int, default=2)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        df, audit = load_dataset(args.dados)
        out = args.saida.resolve()
        if args.command == 'avaliar':
            evaluate_final(df, audit, out)
        else:
            config = Config(args.seed, args.teste, args.folds, args.repeticoes, args.jobs, args.divisao)
            if not 0 < config.test_size < 1 or config.n_splits < 2 or config.n_repeats < 1 or config.n_jobs < 1:
                raise ValueError('Use 0 < teste < 1, folds >= 2, repetições >= 1 e jobs >= 1.')
            if args.command == 'treinar' and (out / 'avaliacao_final.json').exists():
                raise ValueError('Este experimento já foi avaliado no teste. Não reajuste usando esse teste.')
            dev_idx, test_idx, cv, groups = split_dataset(df, config)
            record_setup(df, audit, config, out, dev_idx, test_idx, cv, groups)
            dev = df.iloc[dev_idx]
            exploratory_analysis(dev, out)
            if args.command == 'treinar':
                pipelines, grids = make_models(config)
                summary = compare_models(pipelines, dev[FEATURES], dev[TARGET], cv, config, out)
                winner = tune_models(pipelines, grids, summary, dev[FEATURES], dev[TARGET], cv, config, out, audit)
                interpret_model(winner, dev[FEATURES], dev[TARGET], config, out)
                print('Treinamento concluído. O teste continua reservado; execute avaliar após fechar as decisões.')
        print(f'Artefatos: {out}')
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f'Erro: {exc}\n')


if __name__ == '__main__':
    main()
