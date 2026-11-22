"""Verifica orçamento, isolamento, unidades e consultas sem usar os dados reais."""
import argparse
from contextlib import redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import executar as old
import medicoes as experiment


class MeasurementChecks(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17)
        choices=np.array(list(product(old.OPERATIONS,repeat=6)))
        features=choices[rng.choice(len(choices),size=240,replace=False)]
        self.frame=pd.DataFrame(features,columns=old.FEATURES)
        self.frame['arch_idx']=np.arange(len(self.frame))
        self.frame['arch_str']=self.frame.apply(old.reconstruct_architecture,axis=1)
        self.frame['board']=[f'synthetic_{i%3}' for i in range(len(self.frame))]
        self.frame['energy_mJ']=10+(features=='nor_conv_3x3').sum(axis=1)*4+rng.uniform(0,1,len(self.frame))
        self.frame['latency_s']=self.frame['energy_mJ']/11.5

    def setup_directory(self,root):
        data=root/'data.csv';split=root/'original'
        self.frame.to_csv(data,index=False)
        frame,audit=old.load_dataset(data)
        config=old.Config(n_splits=2,n_repeats=2,n_jobs=1)
        dev,test,cv,groups=old.split_dataset(frame,config)
        old.record_setup(frame,audit,config,split,dev,test,cv,groups)
        return argparse.Namespace(dados=data,divisao=split,saida=root/'new',orcamentos=[10,40,0],
                                  modelos=['baseline_mediana','ridge'],seed=42,jobs=1,inicial=5,
                                  busca_orcamento=20,lote=4,busca_seeds=1)

    def test_nested_budgets_and_no_reserved_rows(self):
        with tempfile.TemporaryDirectory() as temporary,redirect_stdout(StringIO()):
            args=self.setup_directory(Path(temporary))
            dev,folds,orders=experiment.prepare(args)
            test_ids=set(pd.read_csv(args.divisao/'divisao_dados.csv').query("subconjunto=='teste'")['arch_idx'])
            for fold in folds:
                order=orders[(fold['repeat'],fold['fold'])]
                self.assertTrue(set(order[:10])<=set(order[:40])<=set(order))
                self.assertFalse(set(order)&set(fold['valid']))
                self.assertFalse(set(dev.iloc[order]['arch_idx'])&test_ids)
            experiment.curves(args,dev,folds,orders)
            summary=pd.read_csv(args.saida/'curvas_resumo.csv')
            self.assertEqual(set(summary['alvo']),set(experiment.TARGETS))
            self.assertTrue((summary['n_folds']==4).all())
            before=(args.saida/'curvas_scores_folds.csv').read_bytes()
            experiment.curves(args,dev,folds,orders)
            self.assertEqual(before,(args.saida/'curvas_scores_folds.csv').read_bytes())
            baseline=summary.query("modelo=='baseline_mediana'")
            for budget in [10,40,0]:
                energy=baseline.query("alvo=='energy_mJ' and orcamento==@budget")['mae_media'].iloc[0]
                latency=baseline.query("alvo=='latency_s' and orcamento==@budget")['mae_media'].iloc[0]
                self.assertAlmostEqual(latency,energy*1000/11.5,places=8)

    def test_search_budget_initial_pool_and_unseen_labels(self):
        config=old.Config()
        pipelines,grids=old.make_models(config)
        pipelines['floresta'].set_params(model__n_estimators=5)
        with patch.object(old,'make_models',return_value=(pipelines,grids)):
            guided=experiment.search_trace(self.frame,'energy_mJ','guiada_floresta',42,5,20,4,'floresta')
            random=experiment.search_trace(self.frame,'energy_mJ','aleatoria',42,5,20,4,'floresta')
            self.assertEqual(len(guided),20)
            self.assertEqual(len({r['arch_idx'] for r in guided}),20)
            self.assertEqual([r['arch_idx'] for r in guided[:5]],[r['arch_idx'] for r in random[:5]])
            altered=self.frame.copy()
            initial=[r['arch_idx'] for r in guided[:5]]
            altered.loc[~altered['arch_idx'].isin(initial),'energy_mJ']+=1000
            other=experiment.search_trace(altered,'energy_mJ','guiada_floresta',42,5,20,4,'floresta')
            # O primeiro lote depois da inicialização não pode depender de alvos ocultos.
            self.assertEqual([r['arch_idx'] for r in guided[:9]],[r['arch_idx'] for r in other[:9]])
            self.assertTrue(all(guided[i]['melhor_valor']<=guided[i-1]['melhor_valor'] for i in range(1,20)))
            self.assertTrue(all(r['regret']>=0 for r in guided))


if __name__=='__main__':
    unittest.main()
