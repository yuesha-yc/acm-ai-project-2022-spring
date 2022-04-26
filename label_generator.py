import pandas as pd
import os

id_to_label_dict = {}
label_to_id_dict = {}

data_path = os.getcwd() + "/data/humpback-whale-identification/train"
label_path = os.getcwd() + "/data/humpback-whale-identification/train.csv"

df = pd.read_csv(label_path)

ids = df["Id"]
for id in ids:
    if id not in id_to_label_dict:
        new_label = len(id_to_label_dict)
        id_to_label_dict[id] = new_label
        label_to_id_dict[new_label] = id


def id_to_label(id):
    if id in id_to_label_dict:
        return id_to_label_dict[id]
    else:
        raise ValueError("The specified id doesn't exist")


def label_to_id(label):
    if label in id_to_label_dict:
        return id_to_label_dict[label]
    else:
        raise ValueError("The specified label doesn't exist")
