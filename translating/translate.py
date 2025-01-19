import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import pandas as pd
from openai import OpenAI
import torch
from datasets import Dataset
import numpy as np
import time
from tqdm import tqdm
import re
import json
from json_repair import repair_json

# Declaração de funções
def clean_and_extract_portuguese(text):
    try:
        repaired_text = repair_json(text)
        data = json.loads(repaired_text)
        return data.get('Portuguese', None)
    except json.JSONDecodeError:
        return None

def detect_foreign_chars(text):
    foreign_pattern = r"[\u4e00-\u9fff]|[\u3040-\u30ff]|[\uac00-\ud7af]"
    return re.findall(foreign_pattern, str(text))

def count_words(text):
    return len(str(text).split())

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F1E0-\U0001F1FF"
        "\U0001F300-\U0001F5FF"
        "\U0001F600-\U0001F64F"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "]", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def process_dataset(dataset):
    dataset['text'] = dataset['text'].map(remove_emojis)
    return dataset

# Configurações iniciais
print("\nINICIANDO TRADUÇÃO DE AMH PARA PORTUGUÊS\n")
gpt_key = ""

print(torch.version.cuda)
print(torch.cuda.is_available())

if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
else:
    print("Nenhuma GPU CUDA foi detectada.")

# Leitura do dataset
print("Lendo dataset...")
dataset = pd.read_csv("amh.csv")
dataset = dataset.head(2)
print(len(dataset))
print("Dataset lido com sucesso!")

# Preparação para tradução
results = dataset.copy()
translations = []
times = []
print("Traduzindo textos...")
client = OpenAI(api_key=gpt_key)

start_total = time.time()
with tqdm(total=len(dataset), desc="Translating...", unit="texto") as pbar:
    for i, text in enumerate(dataset["text"], start=1):
        try:
            start = time.time()
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um assistente especializado em traduções de texto para o português. "
                            "Adapte os textos traduzidos ao estilo das redes sociais, garantindo fluidez, "
                            "informalidade e uso de gírias quando apropriado. Sempre mantenha o tom natural e "
                            "contextualizado, sem adicionar explicações ou notas extras."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"""
                            Traduza o texto abaixo do Amharic para o português, adaptando o vocabulário para o estilo típico das redes sociais.
                            Use gírias, abreviações e expressões informais comuns nessas plataformas, evitando traduções literais.
                            Mantenha o tom original e certifique-se de que a tradução soe natural e adequada ao contexto das redes sociais.

                            ATENÇÃO:  
                             - Não adicione explicações, justificativas, notas ou opiniões.  
                             - Retorne apenas a tradução solicitada.  
                             - Não inclua nenhum texto adicional além da tradução.
                             - Não use construções formais como "o(a)", que prejudicam a fluidez e informalidade da frase. Substitua por formas naturais e adaptadas ao estilo das redes sociais.  

                            Retorne a tradução no formato JSON com as seguintes chaves: "Amharic" para o texto original e "Portuguese" para a tradução, seguindo os exemplos abaixo:
                            {{"Amharic": "አደገኛ ጊዜያት...", "Portuguese": "momentos perigosos..."}}
                            {{"Amharic": "{text}", "Portuguese":
                            """
                        ),
                    }
                ]
            )
            response = completion.choices[0].message.content
            end = time.time()
            duration = end - start
            times.append(duration)
            translations.append(response)
        except Exception as e:
            print(f"Erro ao traduzir texto {i}: {e}")
            translations.append(None)
            times.append(None)
        pbar.update(1)

# Pós-processamento
end_total = time.time()
total_duration = end_total - start_total
print(f"Tempo total: {total_duration} segundos")
print("Textos traduzidos com sucesso!")
print("Salvando resultados...")

results['translation'] = translations
results['inference_time'] = times
results['inference_total_time'] = total_duration
results['inference_average_time'] = np.mean([t for t in times if t is not None])

results = results[['id', 'text', 'translation', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'inference_time', 'inference_total_time', 'inference_average_time']]
dataset = Dataset.from_pandas(results)

token = ''
print("Enviando resultados para o Hugging Face...")
dataset.push_to_hub("fernandabufon/amh_to_pt_json_gpt", token=token)

df = results.copy()
df['translated_text'] = df['translation'].apply(clean_and_extract_portuguese)
df = df.reset_index(drop=True)
df = df.rename(columns={'translation': 'generation'})
df = df[['id', 'text', 'translated_text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'generation', 'inference_time', 'inference_total_time', 'inference_average_time']]

dataset = Dataset.from_pandas(df)
dataset.push_to_hub("fernandabufon/amh_to_pt_gpt", token=token, private=True)

texts = df[['text', 'translated_text']]
texts['foreign_chars'] = texts['translated_text'].apply(detect_foreign_chars)
texts['has_foreign_chars'] = texts['foreign_chars'].apply(lambda x: len(x) > 0)

foreign_lines = texts[texts['has_foreign_chars']]
df = df[~df.index.isin(foreign_lines.index)]

df_filtered = df.copy()
df_filtered['text_word_count'] = df_filtered['text'].apply(count_words)
df_filtered['translation_word_count'] = df_filtered['translated_text'].apply(count_words)
df_filtered['word_count_diff'] = df_filtered['text_word_count'] - df_filtered['translation_word_count']

threshold = 25
df_filtered['large_diff_flag'] = df_filtered['word_count_diff'].abs() > threshold
significant_differences = df_filtered[df_filtered['large_diff_flag']]
df = df[~df.index.isin(significant_differences.index)]

df.reset_index(drop=True, inplace=True)
df = df[['id', 'translated_text', 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']]
df = df.rename(columns={'translated_text': 'text'})

df = process_dataset(df)
dataset = Dataset.from_pandas(df)
dataset.push_to_hub("SEMEVAL-11/amh2pt_v2", token=token, private=True)
