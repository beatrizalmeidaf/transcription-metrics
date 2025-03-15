import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from num2words import num2words
import chardet

# Definição do alfabeto permitido
alphabet = 'abcdefghijklmnopqrstuvwxyzáéíóúâêîôûãõàèìòùç '

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

# Função para calcular WER
def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
    
    for i in range(len(ref_words) + 1):
        d[i, 0] = i
    for j in range(len(hyp_words) + 1):
        d[0, j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                substitution = d[i - 1, j - 1] + 1
                insertion = d[i, j - 1] + 1
                deletion = d[i - 1, j] + 1
                d[i, j] = min(substitution, insertion, deletion)

    wer = d[len(ref_words), len(hyp_words)] / max(1, len(ref_words))
    return wer

# Configurar argumentos do terminal
parser = argparse.ArgumentParser(description='Calcula o WER entre duas colunas de um CSV.')
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

# Ler o arquivo 
df = pd.read_csv(args.file_path, encoding=encoding_detected)

# Calculando o WER para cada linha
df['WER'] = df.apply(lambda row: calculate_wer(normalize(row[args.ground_truth_col]), normalize(row[args.transcription_col])), axis=1)

# Cálculo do WER geral (média dos valores)
wer_geral = df['WER'].mean()
print(f"Word Error Rate (WER) Geral: {wer_geral:.4f}")

# Salvar os resultados no arquivo CSV atualizado
df.to_csv(args.output_file, index=False, encoding='utf-8')

# Gráfico de violino para visualizar a distribuição do WER
plt.figure(figsize=(8, 6))
sns.violinplot(y=df['WER'], color="skyblue")
plt.title("Distribuição do WER")
plt.ylabel("WER")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

print("Cálculo de WER finalizado e salvo no CSV!")
