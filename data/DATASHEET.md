# Datasheet — NATS-Bench-MCU e dados dos experimentos

## Origem e atribuição

Fonte: Sebastian Zimmermann, René Groh e Andreas M. Kist, **NATS-Bench-MCU**, versão 2.0, Zenodo, 2026. DOI: https://doi.org/10.5281/zenodo.21644470. Licença declarada para o dataset: **Creative Commons Attribution 4.0 International (CC BY 4.0)**, conforme os metadados DataCite conferidos no projeto de origem. Essa licença não atribui automaticamente uma licença ao código ou aos artigos deste trabalho.

O export recebido descreve 15.625 tentativas no nRF5340-DK, 14.281 implantações bem-sucedidas e 1.344 falhas. Somente os sucessos entram nas regressões. O export contém métricas resumidas por arquitetura, associadas a três unidades de placa. Segundo a descrição do depósito, a aquisição empregou Nordic Power Profiler Kit II a 100 kHz. Não foram realizadas novas medições neste trabalho.

A associação do export ao depósito foi informada pelos responsáveis pelo projeto e é consistente com sua documentação; não foi realizada nova comparação byte a byte com download remoto. Firmware, séries temporais brutas completas, repetições físicas e condições detalhadas de aquisição não foram inspecionados.

## Arquivos e transformação

- `data/raw/nats-bench/export/`: CSVs originais e documentação do export.
- `data/processed/nats-bench/hw_metrics_arch_features.csv`: base original acrescida das seis operações categóricas extraídas de `arch_str`, preservando os valores originais.
- `pratica/dados/hw_metrics_arch_features.csv`: cópia usada diretamente pelos scripts.
- SHA-256 da base expandida: `bfa3154b18d048b49462a59159b3e7c7f1d486803cc17e89b372f76724eac725`.

## Dicionário e uso dos dados

Arquivo principal: `pratica/dados/hw_metrics_arch_features.csv`. Origem: `data/raw/nats-bench/export/hw_metrics.csv`, expandido sem alterar as células das colunas originais. São 14.281 arquiteturas e 23 colunas.

## Features iniciais

`arch_op_i_j` contém a operação na conexão do nó de origem `i` ao nó de destino `j`, conforme os grupos de `arch_str`: cada grupo separado por `+` corresponde a um destino, e `~i` identifica a origem. Trata-se da topologia de uma célula, não da lista completa de camadas da rede.

| Coluna | Conexão |
|---|---|
| `arch_op_0_1` | 0 → 1 |
| `arch_op_0_2` | 0 → 2 |
| `arch_op_1_2` | 1 → 2 |
| `arch_op_0_3` | 0 → 3 |
| `arch_op_1_3` | 1 → 3 |
| `arch_op_2_3` | 2 → 3 |

Categorias observadas em todas as conexões: `none`, `skip_connect`, `nor_conv_1x1`, `nor_conv_3x3`, `avg_pool_3x3`. `none` é uma categoria válida (ausência de conexão), não um dado faltante. A arquitetura pode ser reconstruída exatamente com as seis features.

## Alvo e demais colunas

Descrições e unidades conforme o README do exportador. A auditoria do script verifica as propriedades efetivas da cópia.

| Coluna | Significado | Uso inicial |
|---|---|---|
| `energy_mJ` | Energia média por inferência [mJ] | Alvo principal; excluído das features de latência |
| `arch_idx` | Índice NATS-Bench TSS | Identificação e registro das divisões |
| `arch_str` | String original da topologia | Auditoria e rastreabilidade |
| `board` | Unidade nRF5340-DK da medição | Análise de erros; estratificação somente no protocolo aleatório |
| `int8_kb` | Tamanho do arquivo TFLite INT8 [KB] | Excluído das entradas do experimento inicial |
| `rom_kb` | ROM total no dispositivo [KB] | Excluído |
| `ram_kb` | RAM total no dispositivo [KB] | Excluído; constante na base |
| `latency_s` | Latência média por inferência [s] | Alvo secundário nas curvas; excluído das features; erros relatados em ms |
| `current_uA` | Corrente média durante a inferência [µA] | Excluído; medição ligada ao alvo |
| `power_mW` | Potência média durante a inferência [mW] | Excluído; medição ligada ao alvo |
| `g_model_kb` | Tamanho do símbolo `g_model` [KB] | Excluído |
| `arena_used_kb` | Arena usada em execução [KB] | Excluído |
| `arena_size_kb` | Arena alocada estaticamente [KB] | Excluído; constante na base |
| `arena_used_pct` | Percentual de arena usada [%] | Excluído |
| `operators` | Número de operadores TFLM | Excluído do experimento inicial |
| `rom_categories_json` | Composição da ROM por categoria [KB] | Excluído |
| `ram_categories_json` | Composição da RAM por categoria [KB] | Excluído; constante na base |

Tamanho do modelo e contagem de operadores podem motivar um experimento adicional se estiverem disponíveis no momento da previsão pretendida. Essa mudança deve ser documentada como outra configuração de features, mantendo a comparação inicial baseada somente na célula.

Não adicionar automaticamente `accuracies.csv` ou `failures.csv` ao treinamento: o primeiro descreve outro tipo de desempenho e o segundo contém casos sem energia medida.

## Protocolo por orçamento

`medicoes.py` ajusta energia e latência separadamente, com as mesmas seis operações e subconjuntos. Latência é ajustada em segundos; MAE/RMSE são convertidos para ms no relatório. Nenhum dos alvos entra como feature do outro. Um par energia/latência da mesma arquitetura não é contado como duas implantações.
