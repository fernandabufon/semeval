import pandas as pd
from datasets import load_dataset

token = ""

data = load_dataset('SEMEVAL-11/translated_track_a', token=token)

pt = pd.read_csv('ptbr.csv')


geral = pd.DataFrame()  
print("Iniciando processamento dos idiomas...")
for split in data.keys():  
    print(f"\nProcessando idioma: {split}")

    df = data[split].to_pandas().reset_index(drop=True)  
    emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']

    df[emotions] = df[emotions].astype(int)
    pt[emotions] = pt[emotions].astype(int)
    df[emotions] = df[emotions].replace(-1, 0)

    df_filtered = df.merge(pt, on=emotions, suffixes=('_orig', '_pt'))
    print(f"Linhas após merge: {len(df_filtered)}")

    if df_filtered.empty:
        print(f"Nenhuma correspondência encontrada para o idioma {split}.")
        continue

    df_filtered['idioma'] = split
    df_filtered['nota'] = None  
    
    df_filtered['comb_emocao'] = df_filtered[emotions].astype(str).agg('-'.join, axis=1)
    print(f"Total de combinações únicas de emoções: {df_filtered['comb_emocao'].nunique()}")

    unique_samples = (
        df_filtered.groupby('comb_emocao')
        .sample(n=1, random_state=42) 
        .reset_index(drop=True)
    )
    print(f"Linhas após seleção de combinações únicas: {len(unique_samples)}")

    if len(unique_samples) > 10:
        unique_samples = unique_samples.sample(10, random_state=42)
        print(f"Reduzido para 10 amostras para o idioma {split}.")
    elif len(unique_samples) < 10:
        print(f"Apenas {len(unique_samples)} combinações únicas disponíveis para o idioma {split}. Preenchendo com amostras adicionais.")
        remaining = df_filtered.loc[~df_filtered.index.isin(unique_samples.index)]
        additional_samples = remaining.sample(10 - len(unique_samples), random_state=42)
        unique_samples = pd.concat([unique_samples, additional_samples], ignore_index=True)
        print(f"Amostras adicionais adicionadas. Total de 10 para o idioma {split}.")

    geral = pd.concat([geral, unique_samples], ignore_index=True)
    print(f"Total acumulado de linhas no dataset final: {len(geral)}")

print("\nProcessamento finalizado!")
print(f"Total de idiomas processados: {len(data.keys())}")
print(f"Total de linhas no dataset final: {len(geral)}")

geral.to_csv('geral.csv', index=False)