import kagglehub
import shutil
import os
import pandas as pd

def download_dataset():
    # Download the dataset
    download_path = kagglehub.dataset_download("victordibia/signverod")
    # Move to target path
    dataPath = "data/raw/signverod_dataset/"
    for file_name in os.listdir(download_path):
        full_file_name = os.path.join(download_path, file_name)
        if os.path.isfile(full_file_name) or os.path.isdir(full_file_name):
            shutil.move(full_file_name, dataPath)

    print(f"Dataset files moved to: {dataPath}")

    # Fix and merge dataset. See: https://www.kaggle.com/code/alexhorduz/fixing-signverod-dataset
    trainDF = pd.read_csv(dataPath + 'train.csv')
    testDF = pd.read_csv(dataPath + 'test.csv')
    mapping = pd.read_csv(dataPath + 'image_ids.csv')
    trainDF.loc[trainDF.index > 4309, 'image_id'] += 2133
    trainDF.loc[trainDF.index > 4309, 'id'] += 4737
    trainDF.iloc[4307:4316]
    testDF.loc[testDF.index > 809, 'image_id'] += 2133
    testDF.loc[testDF.index > 809, 'id'] += 4737
    testDF.iloc[806:820]
    mapping.loc[mapping.index > 2132, 'id'] += 2133
    mapping.iloc[2130:2140]
    testIDS = set(testDF['id'])
    trainIDS = set(trainDF['id'])
    duplicated = testIDS.intersection(trainIDS)
    trainDF.loc[trainDF['id'] == 26, :]
    testDF.loc[testDF['id'] == 26, :]
    data = pd.concat([trainDF, testDF]).drop_duplicates().sort_values(['id'])

    # Save the fixed version of the dataset
    save_path = "data/raw/fixed_dataset/"
    os.makedirs(save_path, exist_ok=True)
    data.to_csv(save_path + "full_data.csv", index=False)
    mapping.to_csv(save_path + "updated_image_ids.csv", index=False)