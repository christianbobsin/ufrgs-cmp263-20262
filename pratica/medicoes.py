#!/usr/bin/env python3
"""Curvas de erro por orçamento de medições e busca complementar no desenvolvimento."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

import executar as original

BASE = Path(__file__).resolve().parent
TARGETS = {'energy_mJ': ('mJ', 1.0), 'latency_s': ('ms', 1000.0)}
MODEL_NAMES = ['baseline_mediana', 'ridge', 'knn', 'arvore', 'floresta', 'svr']
NAMES = {'baseline_mediana': 'Median / Mediana', 'ridge': 'Ridge', 'knn': 'k-NN',
         'arvore': 'Decision tree / Árvore', 'floresta': 'Random forest / Floresta', 'svr': 'SVR'}


def atomic_csv(frame, path):
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_development(data_path, split_path):
    """Usa a divisão registrada; devolve alvos apenas do desenvolvimento."""
    frame, audit = original.load_dataset(data_path)
    manifest = json.loads((split_path / 'configuracao.json').read_text())
    if audit['sha256'] != manifest['data_sha256']:
        raise ValueError('O hash do CSV difere da divisão de referência.')
    if manifest['config']['division'] != 'composicao':
        raise ValueError('Este experimento requer a divisão por composição.')
    membership = pd.read_csv(split_path / 'divisao_dados.csv')
    if len(membership) != len(frame) or not membership['linha_csv'].is_unique:
        raise ValueError('Registro de divisão inconsistente.')
    if not np.array_equal(np.sort(membership['linha_csv']), np.arange(len(frame))):
        raise ValueError('Índices da divisão não cobrem o CSV.')
    dev_rows = membership[membership['subconjunto'] == 'desenvolvimento'].sort_values('ordem_subset')
    test_rows = membership[membership['subconjunto'] == 'teste']
    if len(dev_rows) + len(test_rows) != len(frame):
        raise ValueError('Subconjunto desconhecido no registro de divisão.')
    if set(dev_rows['grupo_composicao']) & set(test_rows['grupo_composicao']):
        raise ValueError('Sobreposição entre grupos de desenvolvimento e teste.')
    dev = frame.iloc[dev_rows['linha_csv']].reset_index(drop=True)
    if not np.array_equal(dev['arch_idx'], dev_rows['arch_idx']):
        raise ValueError('Identificadores não correspondem ao registro de divisão.')
    groups = original.composition_groups(dev)
    if not np.array_equal(groups, dev_rows['grupo_composicao']):
        raise ValueError('Composições não correspondem ao registro de divisão.')
    for target in TARGETS:
        values = pd.to_numeric(dev[target], errors='raise')
        if not (np.isfinite(values).all() and values.gt(0).all()):
            raise ValueError(f'Alvo inválido no desenvolvimento: {target}.')
        dev[target] = values
    saved = pd.read_csv(split_path / 'folds_cv.csv')
    folds = []
    for (repeat, fold), entries in saved.groupby(['repeticao', 'fold_validacao']):
        valid = entries['posicao_desenvolvimento'].to_numpy(dtype=int)
        if len(valid) != len(set(valid)) or valid.min() < 0 or valid.max() >= len(dev):
            raise ValueError('Posições inválidas em um fold.')
        if not np.array_equal(dev.iloc[valid]['arch_idx'], entries['arch_idx']):
            raise ValueError('Identificadores de validação inconsistentes.')
        train = np.setdiff1d(np.arange(len(dev)), valid)
        if set(groups.iloc[train]) & set(groups.iloc[valid]):
            raise ValueError('Composição compartilhada entre treino e validação.')
        folds.append({'repeat': int(repeat), 'fold': int(fold), 'train': train, 'valid': valid})
    for repeat, entries in saved.groupby('repeticao'):
        if sorted(entries['posicao_desenvolvimento']) != list(range(len(dev))):
            raise ValueError(f'A repetição {repeat} não valida cada exemplo uma vez.')
    if len(folds) != manifest['config']['n_splits'] * manifest['config']['n_repeats']:
        raise ValueError('Número de folds diferente do manifesto.')
    return dev, folds, audit, manifest


def training_orders(folds, seed):
    return {(f['repeat'], f['fold']): np.random.default_rng(seed + 1000*f['repeat'] + f['fold']).permutation(f['train'])
            for f in folds}


def validate_budgets(budgets):
    if len(set(budgets)) != len(budgets) or any(n < 0 for n in budgets):
        raise ValueError('Orçamentos devem ser únicos e não negativos; 0 significa todo o treino do fold.')
    if not budgets:
        raise ValueError('Informe ao menos um orçamento.')


def configuration(args, audit, reference):
    return {'protocol': 'medicoes-v1', 'data_sha256': audit['sha256'],
            'split_sha256': {name: hashlib.sha256((args.divisao/name).read_bytes()).hexdigest()
                            for name in ['configuracao.json','divisao_dados.csv','folds_cv.csv']},
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'model_source_sha256': hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
            'features': original.FEATURES, 'targets': {k: {'unit':v[0], 'reporting_scale':v[1]} for k,v in TARGETS.items()},
            'budgets': args.orcamentos, 'models': args.modelos, 'seed': args.seed,
            'reference_config': reference['config'],
            'search': {'initial': args.inicial, 'budget': args.busca_orcamento, 'batch': args.lote,
                       'seeds': args.busca_seeds, 'strategies': ['aleatoria','guiada_floresta'],
                       'guide': 'floresta', 'guide_tuning': False, 'exploration_per_batch': 1},
            'packages': {name: importlib.metadata.version(name) for name in ['scikit-learn','numpy','pandas','joblib']}}


def prepare(args):
    validate_budgets(args.orcamentos)
    if args.saida.resolve() == args.divisao.resolve():
        raise ValueError('Use uma nova saída; os resultados originais devem ser preservados.')
    if args.inicial < 5 or args.busca_orcamento <= args.inicial or args.lote < 1 or args.busca_seeds < 1:
        raise ValueError('Busca requer inicial >= 5, orçamento > inicial, lote e seeds >= 1.')
    dev, folds, audit, reference = load_development(args.dados, args.divisao)
    if max([n for n in args.orcamentos if n] or [0]) > min(len(f['train']) for f in folds):
        raise ValueError('Há orçamento maior que o conjunto de treino de um fold.')
    if 'knn' in args.modelos and any(0 < n < 5 for n in args.orcamentos):
        raise ValueError('k-NN configurado com cinco vizinhos requer pelo menos cinco exemplos.')
    if args.busca_orcamento > len(dev):
        raise ValueError('Orçamento de busca maior que o desenvolvimento.')
    config = configuration(args, audit, reference)
    args.saida.mkdir(parents=True, exist_ok=True)
    file = args.saida/'configuracao.json'
    if file.exists() and json.loads(file.read_text()) != config:
        raise ValueError('A saída contém outra configuração. Escolha outra pasta.')
    original.save_json(file, config)
    original.save_json(args.saida/'escopo.json', {'development_rows':len(dev), 'folds':len(folds),
                        'final_test_evaluated':False, 'error_threshold_defined':False})
    orders = training_orders(folds, args.seed)
    path = args.saida/'membros_treino.csv'
    if not path.exists():
        entries=[]
        for f in folds:
            order = orders[(f['repeat'],f['fold'])]
            for budget in args.orcamentos:
                selected=order[:budget or len(order)]
                entries.append(pd.DataFrame({'repeticao':f['repeat'],'fold':f['fold'], 'orcamento':budget,
                                             'ordem_aquisicao':np.arange(len(selected)),
                                             'arch_idx':dev.iloc[selected]['arch_idx'].to_numpy()}))
        atomic_csv(pd.concat(entries, ignore_index=True),path)
    return dev, folds, orders


def curve_task(dev, f, order, budgets, model_name, target, seed):
    pipe = original.make_models(original.Config(seed=seed))[0][model_name]
    scale = TARGETS[target][1]
    records=[]
    with threadpool_limits(limits=1):
        for budget in budgets:
            positions=order[:budget or len(order)]
            model=clone(pipe)
            started=time.perf_counter()
            model.fit(dev.iloc[positions][original.FEATURES],dev.iloc[positions][target])
            seconds=time.perf_counter()-started
            truth=dev.iloc[f['valid']][target]
            predictions=model.predict(dev.iloc[f['valid']][original.FEATURES])
            train_predictions=model.predict(dev.iloc[positions][original.FEATURES])
            train_mae=mean_absolute_error(dev.iloc[positions][target], train_predictions)*scale
            mae=mean_absolute_error(truth,predictions)*scale
            records.append({'alvo':target,'unidade':TARGETS[target][0], 'modelo':model_name,
                            'repeticao':f['repeat'],'fold':f['fold'], 'orcamento':budget,
                            'n_medicoes':len(positions),'n_validacao':len(f['valid']),
                            'mae':mae,'rmse':root_mean_squared_error(truth,predictions)*scale,
                            'r2':r2_score(truth,predictions),'mae_treino':train_mae,
                            'gap_mae':mae-train_mae,'tempo_fit_s':seconds})
    return records


def plot_curves(summary,out):
    labels={'en':('Measured architectures used for training','Validation MAE'),
            'ptbr':('Arquiteturas medidas utilizadas no treinamento','MAE de validação')}
    for target in TARGETS:
        selected=summary[summary['alvo']==target]
        for language,(xlabel,ylabel) in labels.items():
            fig,ax=plt.subplots(figsize=(7.3,3.8))
            for name,frame in selected.groupby('modelo',sort=False):
                frame=frame.sort_values('n_medicoes_media')
                label=NAMES[name].split(' / ')[0 if language=='en' else -1]
                ax.plot(frame['n_medicoes_media'],frame['mae_media'],marker='o',label=label)
            ax.set(xscale='log',xlabel=xlabel,ylabel=f'{ylabel} ({TARGETS[target][0]})')
            ax.grid(alpha=.25)
            ax.legend(fontsize=8,ncol=2)
            fig.tight_layout()
            fig.savefig(out/f'curva_{target}_{language}.pdf',bbox_inches='tight')
            plt.close(fig)


def curves(args,dev,folds,orders):
    path=args.saida/'curvas_scores_folds.csv'
    prior=pd.read_csv(path) if path.exists() else pd.DataFrame()
    records=prior.to_dict('records')
    completed=set()
    if not prior.empty:
        for key,group in prior.groupby(['alvo','modelo','repeticao','fold']):
            if set(group['orcamento']) != set(args.orcamentos) or len(group)!=len(args.orcamentos):
                raise ValueError('Checkpoint de curvas incompleto ou duplicado.')
            completed.add(key)
    jobs=[]
    for target in TARGETS:
        for model in args.modelos:
            for f in folds:
                if (target,model,f['repeat'],f['fold']) not in completed:
                    jobs.append((target,model,f))
    print(f'Curvas: {len(jobs)} tarefas pendentes; {len(completed)} concluídas.',flush=True)
    stream=Parallel(n_jobs=args.jobs,return_as='generator_unordered')(
        delayed(curve_task)(dev,f,orders[(f['repeat'],f['fold'])],args.orcamentos,model,target,args.seed)
        for target,model,f in jobs)
    for index,result in enumerate(stream,1):
        records.extend(result)
        atomic_csv(pd.DataFrame(records),path)
        row=result[0]
        print(f"[{index}/{len(jobs)}] {row['alvo']} {row['modelo']} repetição {row['repeticao']} fold {row['fold']}",flush=True)
    scores=pd.DataFrame(records)
    summary=scores.groupby(['alvo','unidade','modelo','orcamento'],sort=False).agg(
        n_medicoes_media=('n_medicoes','mean'), n_medicoes_min=('n_medicoes','min'), n_medicoes_max=('n_medicoes','max'),
        n_folds=('mae','size'), mae_media=('mae','mean'), mae_desvio=('mae','std'),
        rmse_media=('rmse','mean'), r2_medio=('r2','mean'), mae_treino_media=('mae_treino','mean'),
        gap_mae_medio=('gap_mae','mean'), tempo_fit_medio_s=('tempo_fit_s','mean')).reset_index()
    if not (summary['n_folds']==len(folds)).all():
        raise ValueError('Há resultados ausentes.')
    atomic_csv(summary,args.saida/'curvas_resumo.csv')
    plot_curves(summary,args.saida)
    original.save_json(args.saida/'curvas_concluidas.json', {'n_fits':len(scores),'test_evaluated':False,
                       'error_threshold_defined':False,'fold_variability_is_confidence_interval':False})


def search_trace(dev,target,strategy,seed,initial,budget,batch,guide):
    """A política vê apenas alvos já consultados; o oráculo revela cada consulta."""
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(dev))
    observed=list(order[:initial])
    initial_positions=set(observed)
    remaining=[int(i) for i in order[initial:]]
    policy_model=original.make_models(original.Config(seed=seed))[0][guide]
    records=[]
    scale=TARGETS[target][1]
    incumbent=np.inf
    def record(position):
        nonlocal incumbent
        measured=float(dev.iloc[position][target])*scale
        incumbent=min(incumbent,measured)
        records.append({'alvo':target,'unidade':TARGETS[target][0],'estrategia':strategy,'seed':seed,
                        'n_medicoes':len(records)+1,'arch_idx':int(dev.iloc[position]['arch_idx']),
                        'valor_consultado':measured,'melhor_valor':incumbent})
    for position in observed:record(position)
    with threadpool_limits(limits=1):
        while len(observed)<budget:
            size=min(batch,budget-len(observed))
            if strategy=='aleatoria':
                proposals=remaining[:size]
            else:
                model=clone(policy_model).fit(dev.iloc[observed][original.FEATURES],dev.iloc[observed][target])
                # Uma consulta exploratória por lote; demais pela menor predição.
                exploration=int(rng.choice(remaining))
                candidates=[i for i in remaining if i!=exploration]
                predicted=model.predict(dev.iloc[candidates][original.FEATURES])
                ranked=np.argsort(predicted,kind='stable')
                proposals=[exploration]+[candidates[i] for i in ranked[:size-1]]
            for position in proposals:
                if position in observed or position in initial_positions:
                    raise AssertionError('Consulta repetida.')
                remaining.remove(position)
                observed.append(position)
                record(position)
    # O mínimo do pool é usado somente pelo avaliador, após a política terminar.
    optimum=float(dev[target].min())*scale
    for row in records:
        row['regret']=row['melhor_valor']-optimum
    return records


def searches(args,dev):
    path=args.saida/'busca_trajetorias.csv'
    prior=pd.read_csv(path) if path.exists() else pd.DataFrame()
    records=prior.to_dict('records')
    completed=set()
    if not prior.empty:
        for key,group in prior.groupby(['alvo','estrategia','seed']):
            if len(group)!=args.busca_orcamento or group['arch_idx'].duplicated().any():
                raise ValueError('Checkpoint de busca inválido.')
            completed.add(key)
    for target in TARGETS:
        for seed in range(args.seed,args.seed+args.busca_seeds):
            for strategy in ['aleatoria','guiada_floresta']:
                if (target,strategy,seed) in completed:continue
                result=search_trace(dev,target,strategy,seed,args.inicial,args.busca_orcamento,args.lote,'floresta')
                records.extend(result)
                atomic_csv(pd.DataFrame(records),path)
                print(f'Busca: {target}, {strategy}, seed {seed}: {len(result)} consultas.',flush=True)
    traces=pd.DataFrame(records)
    final=traces[traces['n_medicoes']==args.busca_orcamento].groupby(['alvo','unidade','estrategia']).agg(
        n_seeds=('seed','size'), melhor_valor_medio=('melhor_valor','mean'),
        melhor_valor_desvio=('melhor_valor','std'), regret_medio=('regret','mean')).reset_index()
    atomic_csv(final,args.saida/'busca_resumo.csv')
    for target in TARGETS:
        for language in ['en','ptbr']:
            fig,ax=plt.subplots(figsize=(7,3.3))
            for strategy,frame in traces[traces['alvo']==target].groupby('estrategia'):
                grouped=frame.groupby('n_medicoes')['regret'].mean()
                label={'aleatoria':('Random','Aleatória'),'guiada_floresta':('RF-guided','Guiada por floresta')}[strategy][language=='ptbr']
                ax.plot(grouped.index,grouped,label=label)
            ax.set(xlabel='Consulted architectures' if language=='en' else 'Arquiteturas consultadas',
                   ylabel=f"{'Mean regret' if language=='en' else 'Regret médio'} ({TARGETS[target][0]})")
            ax.grid(alpha=.25);ax.legend();fig.tight_layout()
            fig.savefig(args.saida/f'busca_{target}_{language}.pdf',bbox_inches='tight');plt.close(fig)
    original.save_json(args.saida/'busca_concluida.json',{'n_runs':len(completed) if traces.empty else traces.groupby(['alvo','estrategia','seed']).ngroups,
                       'pool':'development_only','test_evaluated':False,'accuracy_constraint':False,
                       'physical_measurements_performed':False})


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comando',choices=['planejar','curvas','busca','executar'])
    parser.add_argument('--dados',type=Path,default=BASE/'dados/hw_metrics_arch_features.csv')
    parser.add_argument('--divisao',type=Path,default=BASE/'resultados/energia_composicao')
    parser.add_argument('--saida',type=Path,default=BASE/'resultados/medicoes_v1')
    parser.add_argument('--orcamentos',type=int,nargs='+',default=[100,300,1000,3000,0],help='0 = todo o treino de cada fold.')
    parser.add_argument('--modelos',nargs='+',choices=MODEL_NAMES,default=MODEL_NAMES)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--inicial',type=int,default=20)
    parser.add_argument('--busca-orcamento',type=int,default=200)
    parser.add_argument('--lote',type=int,default=10)
    parser.add_argument('--busca-seeds',type=int,default=5)
    return parser


def main():
    parser=build_parser();args=parser.parse_args()
    if args.jobs<1:parser.error('--jobs deve ser positivo.')
    try:
        dev,folds,orders=prepare(args)
        print(f'Desenvolvimento: {len(dev)} arquiteturas; {len(folds)} folds. Teste permanece reservado.',flush=True)
        if args.comando in ['curvas','executar']:curves(args,dev,folds,orders)
        if args.comando in ['busca','executar']:searches(args,dev)
        print(f'Artefatos: {args.saida}',flush=True)
    except (ValueError,FileNotFoundError) as exc:
        parser.exit(2,f'Erro: {exc}\n')


if __name__=='__main__':
    main()
