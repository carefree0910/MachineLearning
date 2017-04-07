import os
import pickle

from SkRun import run

if not os.path.isfile("dataset.dat"):
    print("Processing data...")
    rs, labels = [], []
    data_folder = "_Data"
    for i, folder in enumerate(os.listdir(data_folder)):
        for txt_file in os.listdir(os.path.join(data_folder, folder)):
            with open(os.path.join(data_folder, folder, txt_file), "r", encoding="utf-8") as file:
                try:
                    rs.append(file.readline().split())
                    labels.append(i)
                except UnicodeDecodeError as err:
                    print(err)
    with open("dataset.dat", "wb") as file:
        pickle.dump((rs, labels), file)
    print("Done")

print("Running Naive Bayes written by myself...")
os.system("python _NB.py")

print("Running Naive Bayes in sklearn...")
run("Naive Bayes")

print("Running LinearSVM in sklearn")
run("SVM")
