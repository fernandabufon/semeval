import pandas as pd
import ollama
import torch
from datasets import Dataset
from huggingface_hub import create_repo
import numpy as np
from huggingface_hub import HfApi
import time
import numpy as np
from tqdm import tqdm


print(torch.version.cuda)  # Verifica a versão do CUDA suportada pelo PyTorch
print(torch.cuda.is_available())  # Confirma se o CUDA está disponível

# Verifica se CUDA está disponível
print("CUDA disponível:", torch.cuda.is_available())

# Verifica a versão do CUDA
print("Versão do CUDA:", torch.version.cuda)

# Nome da GPU detectada
if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
else:
    print("Nenhuma GPU CUDA foi detectada.")

print("Lendo dataset...")   
dataset = pd.read_csv("eng.csv")
print("Dataset lido com sucesso!")


results = dataset.copy()

prompt_style = "You are an expert translator specializing in accurate and natural translations between English and Brazilian Portuguese.Translate the following text into Brazilian Portuguese, ensuring the meaning, tone, idiomatic expressions, slang, and any profanity are preserved. Do not add explanations or additional information; return only the translated text. Make sure the translation reflects the author's original style and intent: text"
translations = []
times = []
print("Traduzindo textos...")

start_total = time.time()
with tqdm(total=len(dataset), desc="Translating...", unit="texto") as pbar:
    for i, text in enumerate(dataset["text"], start=1):
        start = time.time()
        response = ollama.generate(
            model='qwq:32b',
            prompt = (
        f"You are an expert translator specializing in accurate and natural translations between English and Brazilian Portuguese. "
        f"Translate the following text into Brazilian Portuguese, ensuring the meaning, tone, idiomatic expressions, slang, and any profanity are preserved. "
        f"Do not add explanations or additional information; return only the translated text. Make sure the translation reflects the author's original style and intent: {text}"
    )

        )
        end = time.time()
        duration = end - start
        times.append(duration)
        translations.append(response['response'])  # Verifique se 'response' retorna corretamente
        pbar.update(1)



end_total = time.time()
total_duration = end_total - start_total
print(f"Tempo total: {total_duration} segundos")
print("Textos traduzidos com sucesso!")
print("Salvando resultados...")
results['translation'] = translations
results['inference_time'] = times
results['inference_total_time']   = total_duration
results['inference_average_time'] = np.mean(times)
results['prompt_style'] = prompt_style


# print(results.head())
# print(results["translation"].iloc[0])

token = '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'

results = results[['id', 'text', 'translation', 'Anger', 'Fear',  'Joy',  'Sadness',  'Surprise', 'prompt_style', 'inference_time', 'inference_total_time', 'inference_average_time']]

print(results.head())


dataset = Dataset.from_pandas(results)

print("Enviando resultados para o hf...")

# api = HfApi()

# repo_id="fernandabufon/eng_to_pt"
# # Verificar se o repositório já existe
# try:
#     api.repo_info(repo_id, token=token)
#     print(f"O repositório '{repo_id}' já existe.")
# except Exception as e:
#     print(f"Repositório '{repo_id}' não encontrado. Criando um novo repositório.")
#     create_repo(repo_id, private=True, token=token, repo_type="dataset")

# print(f"Repositório criado: {repo_id}")



dataset.push_to_hub("fernandabufon/eng_to_pt_qwq-32b", token=token)