import pandas as pd


geral = pd.read_csv('geral.csv')
geral = geral.sample(frac=1)
i = 0
for index, row in geral.iterrows():
    if pd.notna(row['nota']):  # Verifica se a nota já existe
      continue
    print(f"Texto 1: {row['text_pt']}")
    print(f"Texto 2: {row['text_orig']}")
    print(f"Emoções: Anger - {row['anger']}, Disgust - {row['disgust']}, Fear - {row['anger']}, Joy - {row['joy']}, Sadness - {row['sadness']}, Surprise - {row['surprise']}")
    nota = input("De 0 a 5, quanto o texto 2 aparenta ser brasileiro? (Use como exemplo o texto 1): ")
    
    if nota == 'sair':
        print(f"Você avaliou {i} textos")
        break
    geral.to_csv('geral.csv', index=False)
    i = i+1
    # Atualiza o valor no DataFrame diretamente usando o índice
    geral.at[index, 'nota'] = float(nota)
    geral.to_csv('geral.csv', index=False)
    print()

print(f"\nAgora falta só {geral['nota'].isna().sum()}")

if geral.isna().sum().sum() == 0:
    print("Avaliação completa!")
    mean = geral.groupby('idioma')['nota'].mean().sort_values(ascending=False)
    print("Nota média por idioma:")
    print(mean)