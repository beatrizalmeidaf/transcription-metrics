import pandas as pd
import re
import torch
import argparse
from num2words import num2words
from torchmetrics.text import CharErrorRate
import matplotlib.pyplot as plt
import seaborn as sns
import chardet

# Definindo o alfabeto e verificando se existe uma GPU disponível
alphabet = 'abcdefghijklmnopqrstuvwxyzáéíóúâêîôûãõàèìòùç '
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função de normalização do texto
def normalize(phrase):
    phrase = phrase.lower()
    
    new_words = []
    for word in phrase.split():
        word = re.sub(r"\d+[%]", lambda x: x.group() + " por cento", word)
        word = re.sub(r"%", "", word)
        word = re.sub(r"\d+[o]{1}", lambda x: num2words(x.group()[:-1], to='ordinal', lang='pt_BR'), word)
        ref = word
        word = re.sub(r"\d+[a]{1}", lambda x: num2words(x.group()[:-1], to='ordinal', lang='pt_BR'), word)
        if word != ref:
            segs = word.split(' ')
            word = ''
            for seg in segs:
                word += seg[:-1] + 'a' + ' '
            word = word[:-1]

        if any(i.isdigit() for i in word):
            segs = re.split(r"[?.!\s]", word)
            word = ''
            for seg in segs:
                if seg.isnumeric():
                    seg = num2words(seg, lang='pt_BR')
                word += seg + ' '
            word = word[:-1]
        new_words.append(word)
    
    phrase = ' '.join(new_words)
    
    for c in phrase:
        if c not in alphabet:
            phrase = phrase.replace(c, '')

    return phrase

# Função para calcular o CER entre duas strings
def calculate_cer(ground_truth, transcription):
    cer = CharErrorRate()
    ground_truth = normalize(ground_truth)
    transcription = normalize(transcription)
    cer.update([transcription], [ground_truth])
    return cer.compute().item()

# Configurar argumentos do terminal
parser = argparse.ArgumentParser(description='Calcula o CER entre duas colunas de um CSV.')
parser.add_argument('file_path', type=str, help='Caminho do arquivo CSV de entrada')
parser.add_argument('ground_truth_col', type=str, help='Nome da coluna de referência')
parser.add_argument('transcription_col', type=str, help='Nome da coluna de transcrição')
parser.add_argument('output_file', type=str, help='Nome do arquivo CSV de saída')
args = parser.parse_args()

# Detectar a codificação do arquivo
with open(args.file_path, 'rb') as f:
    result = chardet.detect(f.read())
encoding_detected = result['encoding']
print(f"Encoding detectado: {encoding_detected}")

# Ler o arquivo com a codificação detectada
df = pd.read_csv(args.file_path, encoding=encoding_detected)
print(df.head())

# Calculando o CER linha por linha
df['CER'] = df.apply(lambda row: calculate_cer(row[args.ground_truth_col], row[args.transcription_col]), axis=1)

# Exibindo o DataFrame com os valores de CER
print(df[[args.ground_truth_col, args.transcription_col, 'CER']])

# Calculando um CER geral
total_chars = 0
total_errors = 0
for index, row in df.iterrows():
    ground_truth = normalize(row[args.ground_truth_col])
    total_chars += len(ground_truth)
    total_errors += int(df['CER'][index] * len(ground_truth))

cer_geral = total_errors / total_chars
print(f"CER Geral: {cer_geral:.4f}")

# Salvar o arquivo atualizado
df.to_csv(args.output_file, index=False, encoding='utf-8')

# Plotando um gráfico de violino do CER
plt.figure(figsize=(10, 6))
sns.violinplot(x=df['CER'])
plt.title('Distribuição do CER')
plt.xlabel('Character Error Rate (CER)')
plt.show()
