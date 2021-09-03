import os
import pandas as pd
from matplotlib import pyplot as plt

out_path = os.path.abspath(os.path.dirname(__file__) + "/../files")

prediction = pd.read_csv(out_path + "/predicted_classes.csv", sep=";",
                         names=["feature_1", "feature_2", "cls"], dtype="float")
ground_truth = pd.read_csv(out_path + "/ground_truth.csv", sep=";", dtype="float",
                           names=["feature_1", "feature_2", "cls"], skiprows=1)

cluster_centers = pd.read_csv(out_path + "/clusters.csv", sep=";", dtype="float",
                              names=["feature_1", "feature_2"])
fig, ax = plt.subplots(2, 1)

for pred_class in prediction["cls"].unique():
    tmp = prediction[prediction["cls"] == pred_class]
    ax[0].scatter(tmp["feature_1"], tmp["feature_2"])

for ground_class in ground_truth["cls"].unique():
    tmp = ground_truth[ground_truth["cls"] == ground_class]
    ax[1].scatter(tmp["feature_1"], tmp["feature_2"])

ax[0].scatter(cluster_centers["feature_1"], cluster_centers["feature_2"], color="red", s=50, marker="x")
ax[0].set_title("Prediction")
ax[1].set_title("Ground Truth")

plt.show()
