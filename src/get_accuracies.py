import glob
import json
import numpy as np

files = glob.glob("accuracies/*.json")

def print_latex():
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            print(f"Name: {data['name']}")
            conf = data['conf']
            for i in range(len(conf)):
                for j in range(len(conf[i])):
                    entry = conf[i][j]
                    print(f"{entry} ", end="")
                    if j < len(conf[i]) - 1:
                        print("& ", end="")
                print("\\\\ ", end="")
            print()

def print_jsons():
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            data['conf'] = np.array(data['conf'])
            print(f"Name: {data['name']}")
            print(f"Test Accuracy: {data['test']}")
            print(f"Epochs: {data['epochs']}")
            print(f"Confusion Matrix:")
            conf = data['conf']
            print(conf)
            if conf.shape == (2, 2):
                print("Sensitivity: {}".format(conf[1, 1] / (conf[1, 1] + conf[1, 0])))
                print("Specificity: {}".format(conf[0, 0] / (conf[0, 0] + conf[0, 1])))
            print()

print_latex()
print_jsons()
