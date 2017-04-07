import os
import pickle
import numpy as np


def gen_dataset(dat_path):
    if not os.path.isfile(dat_path):
        print("\nGenerating Dataset...")
        folders = os.listdir("_Data")
        label_dic = [folder for folder in folders if os.path.isdir(os.path.join("_Data", folder))]
        folders_path = [os.path.join("_Data", folder) for folder in label_dic]
        x, y = [], []
        for i, folder in enumerate(folders_path):
            for txt in os.listdir(folder):
                with open(os.path.join(folder, txt), "r", encoding="utf-8") as file:
                    try:
                        x.append(file.read().strip().split())
                        y.append(i)
                    except Exception as err:
                        print(err)
        np.save(os.path.join("_Data", "LABEL_DIC.npy"), label_dic)
        with open(dat_path, "wb") as file:
            pickle.dump((x, y), file)
        print("Done")
