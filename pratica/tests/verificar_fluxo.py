"""Verificação com dados sintéticos: não usa nem avalia o dataset real."""
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import executar as exp


class WorkflowCheck(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        choices = np.array(list(product(exp.OPERATIONS, repeat=6)))
        operations = choices[rng.choice(len(choices), size=240, replace=False)]
        self.df = pd.DataFrame(operations, columns=exp.FEATURES)
        self.df['arch_idx'] = np.arange(len(self.df))
        self.df['board'] = [f'synthetic_{i % 3}' for i in range(len(self.df))]
        self.df[exp.TARGET] = 10 + (operations == 'nor_conv_3x3').sum(axis=1) * 4 + rng.uniform(0, 1, len(self.df))
        self.df['arch_str'] = self.df.apply(exp.reconstruct_architecture, axis=1)
        self.config = exp.Config(n_splits=2, n_repeats=2, n_jobs=1)

    def independent_compositions(self, positions):
        return {tuple(sorted(Counter(row).items()))
                for row in self.df.iloc[positions][exp.FEATURES].to_numpy()}

    def test_group_isolation_and_repetitions(self):
        dev, test, cv, groups = exp.split_dataset(self.df, self.config)
        self.assertFalse(set(dev) & set(test))
        self.assertEqual(len(dev) + len(test), len(self.df))
        self.assertFalse(self.independent_compositions(dev) & self.independent_compositions(test))
        self.assertEqual(len(cv), 4)
        for repeat in range(2):
            validations = []
            for train, valid in cv[repeat * 2:(repeat + 1) * 2]:
                self.assertFalse(self.independent_compositions(dev[train]) &
                                 self.independent_compositions(dev[valid]))
                validations.extend(valid)
            self.assertEqual(sorted(validations), list(range(len(dev))))

    def test_training_and_separate_evaluation(self):
        with tempfile.TemporaryDirectory(prefix='energia-synthetic-check-') as tmp, redirect_stdout(StringIO()):
            root = Path(tmp)
            path = root / 'synthetic.csv'
            out = root / 'outputs'
            self.df.to_csv(path, index=False)
            df, audit = exp.load_dataset(path)
            dev, test, cv, groups = exp.split_dataset(df, self.config)
            exp.record_setup(df, audit, self.config, out, dev, test, cv, groups)
            exp.exploratory_analysis(df.iloc[dev], out)
            X, y = df.iloc[dev][exp.FEATURES], df.iloc[dev][exp.TARGET]
            pipelines, grids = exp.make_models(self.config)
            pipelines['floresta'].set_params(model__n_estimators=5)
            grids = {name: {key: values[:1] for key, values in grid.items()}
                     for name, grid in grids.items()}
            summary = exp.compare_models(pipelines, X, y, cv, self.config, out)
            self.assertEqual(set(summary.index), set(pipelines))
            winner = exp.tune_models(pipelines, grids, summary, X, y, cv, self.config, out, audit)
            exp.interpret_model(winner, X, y, self.config, out)
            # A avaliação carrega o modelo persistido, sem receber winner em memória.
            exp.evaluate_final(df, audit, out)
            predictions = pd.read_csv(out / 'predicoes_teste.csv')
            self.assertEqual(set(predictions['arch_idx']), set(df.iloc[test]['arch_idx']))
            with self.assertRaises(ValueError):
                exp.evaluate_final(df, audit, out)
            self.assertTrue((out / 'importancia_permutacao_desenvolvimento.csv').is_file())


if __name__ == '__main__':
    unittest.main()
