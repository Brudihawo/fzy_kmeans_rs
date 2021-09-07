"""Generate example data for testing."""
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

N_SAMPLES = 100
N_DIMS = 2
N_CLASSES = 3

out_path = os.path.abspath(os.path.dirname(__file__) + "/../files")

if not os.path.exists(out_path):
    os.mkdir(out_path)

dfs = []
for cls in range(N_CLASSES):
    tmp = pd.DataFrame()
    locs = np.random.uniform(low=0.0, high=1.0, size=(N_DIMS))
    stds = np.random.uniform(low=0.0, high=0.15, size=(N_DIMS))
    for i, (std, loc) in enumerate(zip(stds, locs)):
        tmp[f"feature_{i}"] = np.random.normal(
            loc=loc, scale=std, size=(N_SAMPLES))
    tmp["cls"] = [cls for i in range(N_SAMPLES)]
    dfs.append(tmp)

df = pd.concat(dfs)
if "--plot" in sys.argv:
    sns.pairplot(df, hue="cls")
    plt.show()

df.to_csv(out_path + "/sample_data.csv", sep=";", index=False,
          columns=[f"feature_{i}" for i in range(N_DIMS)])

df.to_csv(out_path + "/ground_truth.csv", sep=";", index=False)
