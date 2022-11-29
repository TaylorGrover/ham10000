import glob
import json
import numpy as np

files = glob.glob("*.json")
for file in files:
    with open(file, "r") as f:
        data = json.load(f)
        data['conf'] = np.array(data['conf'])
        print(f"Name: {data['name']}")
        print(f"Test Accuracy: {data['test']}")
        print(f"Confusion Matrix:")
        print(data['conf'])
        print()
