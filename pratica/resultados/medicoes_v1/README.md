# Resultados do protocolo medicoes-v1

Execução com dados reais do benchmark, somente no desenvolvimento: **900 ajustes de curvas** e **20 trajetórias de busca**. Não foram realizadas novas medições físicas nem avaliação do teste reservado.

- `configuracao.json`: parâmetros, hashes das fontes, versões e divisão usada.
- `escopo.json`: conjunto usado e ausência de limiar de erro/teste avaliado.
- `membros_treino.csv`: arquiteturas de cada subconjunto aninhado, compartilhado entre alvos e modelos.
- `curvas_scores_folds.csv`: métricas individuais; energia em mJ e erros de latência em ms.
- `curvas_resumo.csv`: médias/desvios por orçamento e modelo.
- `curva_*_en.pdf`, `curva_*_ptbr.pdf`: curvas com os seis modelos.
- `busca_trajetorias.csv`: cada consulta, valor revelado, melhor custo e regret.
- `busca_resumo.csv`, `busca_*_en.pdf`, `busca_*_ptbr.pdf`: resumo/gráficos da busca.
- `curvas_concluidas.json`, `busca_concluida.json`: conclusão das execuções.

Orçamento `0` nos CSVs de curvas significa **todo o treino do fold**, não ausência de medições. São 100, 300, 1.000, 3.000 ou o conjunto completo daquele treino, cuja média é 9.335,2 exemplos. Desvios entre folds são descritivos e não intervalos de confiança.

Comando, em `pratica/`: `.venv/bin/python medicoes.py executar`. Os checkpoints permitem retomada. Não misture outra configuração nesta pasta; use `--saida` para outro protocolo. Modelo guia da busca foi fixado previamente: floresta aleatória, 20 observações iniciais, consultas em lotes de dez, orçamento total 200 e cinco seeds.

Resumo comentado: `../../docs/resultados_medicoes.md`. As duas versões do artigo estão em `../../../manuscript/`.
