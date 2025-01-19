import pandas as pd
from datasets import load_dataset, DatasetDict, Dataset


def load(dataset):
  token = ""
  dataset = load_dataset(dataset, token=token)
  df = pd.DataFrame(dataset['train'])
  return df



datasets = [
    "hau_to_pt_processed",
    "arq_to_pt_processed",
    "chn_to_pt_processed",
    "deutsch_to_pt_processed",
    "eng_to_pt_processed",
    "esp_to_pt_processed",
    "amh_to_pt_processed",
    "ron_to_pt_processed",
    "rus_to_pt_processed",
    "ukr_to_pt_processed",
]

for dataset in datasets:
  print(f"Dataset: {dataset}")
  df = load(f"fernandabufon/{dataset}")
  print("Salvando dataset")
  df.to_csv(f"data_trackb\{dataset}.csv", index=False)

dfs = [
    "hau",
    "arq",
    "chn",
    "deu",
    "eng",
    "esp",
    "amh",
    "ron",
    "rus",
    "ukr"
]

# for name in dfs:
#   print(f"Dataset: {name}")
#   df = pd.read_csv(f"{name}.csv")
#   print("Salvando dataset")
#   df.to_csv(f"{name}", index=False)

data_names = []
for a, b in zip(datasets, dfs):
  print(f"Dataset: {a}, {b}")
  track_a = pd.read_csv(f"data_trackb\{a}.csv")
  track_b = pd.read_csv(f"data_trackb\{b}.csv")

  track_a = track_a.drop_duplicates(subset=['text'])
  track_b = track_b.drop_duplicates(subset=['text'])

  track_a = track_a[['text', 'translated_text']]
  print(track_a.shape)
  print(track_b.shape)
  merged = pd.merge(track_a, track_b, on='text', how='inner')
  print(merged.shape)
  merged['text'] = merged['translated_text']
  merged = merged.drop(columns=['translated_text'])
  print(merged.columns)
  merged.to_csv(f"{b}_track_b.csv", index=False)
  data_names.append(f"{b}_track_b.csv")
  print(f"Arquivo salvo em {b}_track_b.csv")
  print("------------------------------------------------------")



required_columns = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]


dataset_dict = DatasetDict()


for dataset_name in data_names:
    dataset = pd.read_csv(dataset_name)

    dataset = Dataset.from_pandas(dataset)

    if "__index_level_0__" in dataset.column_names:
        dataset= dataset.remove_columns(["__index_level_0__"])

    for column in required_columns:
        if column not in dataset.column_names:
            dataset = dataset.map(lambda x: {**x, column: -1})

    split_name = dataset_name.split("/")[-1].split("_")[0]

    dataset_dict[split_name] = dataset

print(dataset_dict)

test = pd.Series(dataset_dict['eng']['disgust']).value_counts()
print(test)

token = ""

# dataset_dict.push_to_hub("SEMEVAL-11/translated_track_b", private=True, token=token)