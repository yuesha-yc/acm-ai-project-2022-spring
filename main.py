import os

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train

import pandas as pd

import label_generator


def generateDataset():
    data_path = os.getcwd() + "/data/humpback-whale-identification/train"
    label_path = os.getcwd() + "/data/humpback-whale-identification/train.csv"

    df = pd.read_csv(label_path)

    train = df.iloc[0:20000]
    test = df.iloc[20001:25000]

    return train, test


def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS,
                       "batch_size": constants.BATCH_SIZE}

    # TODO: Add GPU support. This line of code might be helpful.
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!
    train, test = generateDataset()
    # print(train)
    # print(test)

    train_dataset = StartingDataset(train)
    val_dataset = StartingDataset(test)

    # print('test')
    # print(train_dataset[13761])

    model = StartingNetwork()
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=constants.N_EVAL,
    )


if __name__ == "__main__":
    main()
