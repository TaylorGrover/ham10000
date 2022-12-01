"""
Filesystem manipulation
"""
import glob
import numpy as np
import os
from pathlib import Path
import pandas as pd
import shutil

multiclass_path_root = "../data/images_multiclass"
binary_path_root = "../data/images_binary"

def copy_images(df, root_path):
    classes = set(df.dx)
    counts = dict(zip(classes, [0 for i in range(len(classes))]))
    for row in np.array(df):
        name = row[1] + ".jpg"
        img_path = os.path.join(root_path, name)
        counts[row[2]] += 1
        dest_path = f"sample_images/{row[2]}_{counts[row[2]]}.jpg"
        os.system(f"cp {img_path} {dest_path}")


def fix_dir_structure(df, root_path):
    classes = set(df.dx)
    for label in classes:
        dirpath = os.path.join(root_path, label)
        if not os.path.isdir(dirpath):
            os.mkdir(dirpath)
    for row in np.array(df):
        name = row[1] + ".jpg"
        img_path = os.path.join(root_path, name)
        dest_path = os.path.join(root_path, row[2], name)
        os.system(f"mv {img_path} {dest_path}")

def make_train_test_val_dirs(root_path):
    """
    Make 80/10/10%  train/test/val split in dir for keras DataGen
    """
    image_files = glob.glob(os.path.join(root_path, "train", "*/*.jpg"))
    n_val_test = int(0.2 * len(image_files))
    val_test_files = np.random.choice(image_files, n_val_test, replace=False)
    train_files = list(set(image_files) - set(val_test_files))
    n_val = int(0.5 * len(val_test_files))
    val_files = np.random.choice(val_test_files, n_val, replace=False)
    test_files = list(set(val_test_files) - set(val_files))
    print(train_files[:5])
    return train_files, test_files, val_files

#dataframe = pd.read_csv("../data/csvs/HAM10000_metadata.csv")
train, test, val = make_train_test_val_dirs(multiclass_path_root)
