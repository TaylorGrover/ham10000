"""
Get statistics from metadata
"""
import numpy as np
import os
from pathlib import Path
import pandas as pd

root_path = "data/all_images"

def copy_images(df):
    classes = set(df.dx)
    counts = dict(zip(classes, [0 for i in range(len(classes))]))
    for row in np.array(df):
        name = row[1] + ".jpg"
        img_path = os.path.join(root_path, name)
        counts[row[2]] += 1
        dest_path = f"sample_images/{row[2]}_{counts[row[2]]}.jpg"
        os.system(f"cp {img_path} {dest_path}")

dataframe = pd.read_csv("data/csvs/HAM10000_metadata.csv")
samples = dataframe.groupby("dx").apply(lambda x: x.sample(5)).reset_index(drop=True)
copy_images(samples)
