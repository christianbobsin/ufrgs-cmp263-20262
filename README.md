# Predição de energia e latência de arquiteturas no nRF5340-DK

Trabalho de Christian Regina Bobsin e Isabel da Costa Damião, UFRGS. O estudo investiga a relação entre o número de arquiteturas medidas disponíveis para treinamento e o erro de predição da energia por inferência. A latência é uma segunda tarefa de regressão com os mesmos atributos e partições. Uma simulação de busca aleatória versus guiada por floresta complementa os experimentos.

São utilizadas as seis operações categóricas das conexões da célula NATS-Bench. O protocolo agrupa arquiteturas pelas contagens de operações, mantendo composições iguais na mesma partição. A base possui 14.281 arquiteturas: 11.669 no desenvolvimento e 2.612 no teste, distribuídas em 149 e 38 grupos, respectivamente.

Repositório público: https://github.com/christianbobsin/ufrgs-cmp263-20262.

## Estrutura

```text
article/
  artigo-ptbr.pdf                 # Artigo em português
  artigo.pdf                       # Artigo em inglês
pratica/
  executar.py                      # Auditoria, comparação, ajuste e avaliação de energia
  medicoes.py                      # Curvas por orçamento e simulação de busca
  requirements.txt
  requirements.lock.txt
  dados/                           # CSV usado pelos scripts
  tests/                           # Verificações com dados sintéticos
  resultados/
    energia_composicao/            # Fluxo de energia e avaliação final salvos
    medicoes_v1/                    # Curvas e trajetórias de busca salvas
data/
  DATASHEET.md                     # Origem, campos, unidades, uso e limites dos dados
  raw/nats-bench/export/            # Export original
  processed/nats-bench/             # Base com atributos da arquitetura
README.md
```

## Modelos presentes

Todos os regressores usam codificação one-hot ajustada dentro do pipeline de treinamento. Identificadores, placa, energia, latência e outras medições não entram como atributos preditores.

| Modelo | Configuração inicial |
|---|---|
| Ridge | Regularização `alpha=1` |
| k-NN | Cinco vizinhos, pesos uniformes |
| Árvore de decisão | Profundidade máxima 10, mínimo de cinco exemplos por folha |
| Floresta aleatória | 200 árvores, mínimo de três exemplos por folha |
| SVR | Kernel RBF, `C=10`, `epsilon=0.1`, `gamma=scale` |
| Baseline | Mediana do alvo no treinamento |

No fluxo de energia, SVR e floresta passaram por busca em grade. O SVR selecionado usa `C=100`, `epsilon=0.1` e `gamma=scale`; foi reajustado no desenvolvimento completo e salvo em `pratica/resultados/energia_composicao/modelo.joblib`. A floresta usada para guiar a busca complementar mantém configuração fixa, sem ajuste pelo teste.

## Experimentos e resultados disponíveis

1. **Fluxo de energia:** auditoria, partições, 15 folds agrupados (cinco folds em três repetições), comparação dos modelos, ajuste de SVR/floresta, seleção, importância por permutação e avaliação final. As predições finais salvas registram MAE de 1,090996 mJ, RMSE de 1,782726 mJ e R² de 0,992114 para o SVR, nas 2.612 arquiteturas de teste.
2. **Curvas por orçamento:** energia e latência, seis configurações incluindo baseline, cinco orçamentos (100, 300, 1.000, 3.000 e todo o treinamento de cada fold) e 15 folds, totalizando 900 ajustes. Os membros dos subconjuntos, scores e configurações estão salvos.
3. **Busca complementar:** seleção aleatória e guiada por floresta, dois alvos, cinco seeds pareadas, 200 consultas por execução incluindo 20 iniciais, totalizando 20 trajetórias. As consultas revelam medições existentes; não são novas medições físicas.

As curvas e a busca usam somente desenvolvimento. A avaliação final cobre apenas energia. A comparação inicial e as curvas são execuções distintas: k-NN registra MAE de 6,90 e 6,88 mJ, respectivamente. Os arquivos de cada execução foram preservados.

## Preparar o ambiente local

Clone o projeto e execute os comandos em Linux/bash, com Python 3.13 disponível:

```bash
git clone https://github.com/christianbobsin/ufrgs-cmp263-20262.git
cd ufrgs-cmp263-20262
```

Crie o ambiente isolado na raiz do repositório:

```bash
python3.13 -m venv pratica/.venv
pratica/.venv/bin/python -m pip install -r pratica/requirements.lock.txt
```

As dependências diretas estão em `requirements.txt`; o arquivo `requirements.lock.txt` fixa também as transitivas. Os registros de preparação indicam Python 3.13.15; `resultados/energia_composicao/ambiente.json` registra Python 3.14.2 junto à etapa final. Esses registros não garantem compatibilidade com qualquer versão futura. O modelo persistido requer ambiente compatível para carregamento.

Confira as opções dos comandos e execute os testes sintéticos:

```bash
pratica/.venv/bin/python pratica/executar.py --help
pratica/.venv/bin/python pratica/medicoes.py --help
pratica/.venv/bin/python pratica/tests/verificar_fluxo.py
pratica/.venv/bin/python pratica/tests/verificar_medicoes.py
```

Os testes usam dados sintéticos e não repetem a avaliação do dataset real. Para consultar o estudo sem executar modelos, leia os CSVs/JSONs e figuras em `pratica/resultados/` e os PDFs em `article/`.

## Reproduzir os experimentos

Use novos diretórios de saída para manter os resultados entregues. O treinamento pode ser demorado; as regressões usam CPU. A configuração padrão usa dois processos, com uma thread por ajuste no experimento de curvas.

### 1. Auditoria e treinamento de energia

```bash
pratica/.venv/bin/python pratica/executar.py auditar --saida pratica/resultados/reproducao_energia
pratica/.venv/bin/python pratica/executar.py treinar --saida pratica/resultados/reproducao_energia
```

A auditoria salva o hash do CSV, as partições, os folds e os descritivos. O treinamento compara os regressores, ajusta os dois melhores e salva seleção, modelo e importância por permutação.

### 2. Curvas de energia/latência e busca complementar

```bash
pratica/.venv/bin/python pratica/medicoes.py planejar --divisao pratica/resultados/reproducao_energia --saida pratica/resultados/reproducao_medicoes
pratica/.venv/bin/python pratica/medicoes.py curvas --divisao pratica/resultados/reproducao_energia --saida pratica/resultados/reproducao_medicoes
pratica/.venv/bin/python pratica/medicoes.py busca --divisao pratica/resultados/reproducao_energia --saida pratica/resultados/reproducao_medicoes
```

O parâmetro `--divisao` aponta para a auditoria produzida na etapa anterior, preservando as mesmas fronteiras de composição. O comando `executar` de `medicoes.py` reúne curvas e busca. Os checkpoints permitem retomar tarefas já concluídas no mesmo protocolo; parâmetros diferentes exigem outra pasta.

### 3. Avaliação final de energia

Após congelar as decisões de desenvolvimento:

```bash
pratica/.venv/bin/python pratica/executar.py avaliar --saida pratica/resultados/reproducao_energia
```

Esse comando carrega o modelo salvo e produz métricas e predições no teste reservado. A pasta original já avaliada bloqueia novo treinamento e avaliação. O teste original já foi examinado: repetir essa etapa é reprodução, não uma nova avaliação independente. Não use o teste para escolher hiperparâmetros ou orçamentos.

## Dados e limites

A origem, a atribuição ao dataset NATS-Bench-MCU v2.0, a licença declarada CC BY 4.0 e o dicionário das 23 colunas estão em [data/DATASHEET.md](data/DATASHEET.md). As conclusões se restringem às arquiteturas implantadas com sucesso no espaço e hardware estudados. Não foi definido um limiar de erro aceitável para declarar um orçamento suficiente. Não há teste final de latência, comprovação de transferência para outros dispositivos ou interpretação causal da importância por permutação.

Artigos: [português](article/artigo-ptbr.pdf) e [inglês](article/artigo.pdf).
