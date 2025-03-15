# Transcription Metrics

Esse repositório contém scripts para calcular métricas de erro em transcrições automáticas de áudio, especificamente **Character Error Rate (CER)** e **Word Error Rate (WER)**. 

Os scripts permitem a avaliação de modelos de transcrição comparando suas saídas com um texto de referência. Ambos aceitam parâmetros via terminal para serem flexíveis e reutilizáveis com diferentes arquivos CSV.

## Requisitos

Antes de executar os scripts, instale as dependências necessárias:

```bash
pip install pandas numpy torch torchmetrics num2words seaborn matplotlib chardet
```

## Scripts Disponíveis

### 1. `cer.py` - Character Error Rate (CER)

O **CER** mede a taxa de erro por caractere entre a transcrição automática e o texto de referência.

#### Uso:

```bash
python cer.py <arquivo_csv> <coluna_ground_truth> <coluna_transcricao> <arquivo_saida>
```

#### Exemplo:

```bash
python cer.py metadata.csv "ground-truth" "whisper" resultado_cer.csv
```

Isso calculará o CER para cada linha e salvará os resultados no `resultado_cer.csv`.

### 2. `wer.py` - Word Error Rate (WER)

O **WER** mede a taxa de erro por palavra na transcrição, considerando substituições, inserções e deleções.

#### Uso:

```bash
python wer.py <arquivo_csv> <coluna_ground_truth> <coluna_transcricao> <arquivo_saida>
```

#### Exemplo:

```bash
python wer.py metadata.csv "ground-truth" "whisper" resultado_wer.csv
```

Isso calculará o WER para cada linha e salvará os resultados no `resultado_wer.csv`.

## Visualização dos Resultados

Ambos os scripts geram um gráfico de violino para visualizar a distribuição dos erros. Certifique-se de executar os scripts em um ambiente que suporte gráficos, como Jupyter Notebook ou uma IDE que exiba `matplotlib`.

