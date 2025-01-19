from datasets import load_dataset, DatasetDict

token = "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"


datasets_with_versions = [
    "SEMEVAL-11/vmw2pt_v3",
    "SEMEVAL-11/hau2pt_v3",
    "SEMEVAL-11/chn2pt_v3",
    "SEMEVAL-11/esp2pt_v3",
    "SEMEVAL-11/arq2pt_v3",
    "SEMEVAL-11/afr2pt_v3",
    "SEMEVAL-11/oromo2pt_v3",
    "SEMEVAL-11/ary2pt_v3",
    "SEMEVAL-11/somali2pt_v3",
    "SEMEVAL-11/deutsch2pt_v3",
    "SEMEVAL-11/eng2pt_v3",
    "fernandabufon/ukr2pt_v3",
    "fernandabufon/rus2pt_v3",
    "fernandabufon/ron2pt_v3",
    "fernandabufon/hin2pt_v3",
    "fernandabufon/amh2pt_v3",
]
dataset_dict = DatasetDict()

required_columns = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]

for dataset_name in datasets_with_versions:
    dataset = load_dataset(dataset_name, token=token)

    if "__index_level_0__" in dataset["train"].column_names:
        dataset["train"] = dataset["train"].remove_columns(["__index_level_0__"])

    for column in required_columns:
        if column not in dataset["train"].column_names:
            dataset["train"] = dataset["train"].map(lambda x: {**x, column: -1})

    split_name = dataset_name.split("/")[-1].split("2")[0]

    dataset_dict[split_name] = dataset["train"]

print(dataset_dict)


# dataset_dict.push_to_hub("SEMEVAL-11/translated_track_a", private=True, token=token)