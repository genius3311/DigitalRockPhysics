import numpy as np
import tifffile
from utilities import two_point_correlation
import pandas as pd
from tqdm import tnrange
import cv2


# File Directories
sample_dic = {
    "orig_input": "shale_bicubic.thresholded.tif",  # 输入图像路径
    "out_direc": "./",  # 计算结果保存路径
    # "seed_min": 43,
    # "seed_max": 64
}

data_dic = sample_dic
orig_img = tifffile.imread(data_dic["orig_input"])
out_direc = data_dic["out_direc"]
# seed_min, seed_max = data_dic["seed_min"], data_dic["seed_max"]

# Data Loading
# pore_phase = orig_img.min()
# grain_phase = orig_img.max()
pore_phase = orig_img.max()
grain_phase = orig_img.min()

######### Compute two_point_correlation for pore phase ########

two_point_correlation_pore_phase_orig = {}
for i, direc in enumerate(["x", "y", "z"]):
    two_point_direc = two_point_correlation(orig_img, i, var=pore_phase)
    two_point_correlation_pore_phase_orig[direc] = two_point_direc


direc_correlation_pore_phase_orig = {}
for direc in ["x", "y", "z"]:
    direc_correlation_pore_phase_orig[direc] = np.mean(np.mean(two_point_correlation_pore_phase_orig[direc], axis=0), axis=0)
print(direc_correlation_pore_phase_orig["x"].shape)

covariance_orig_df = pd.DataFrame(direc_correlation_pore_phase_orig)
covariance_orig_df.to_csv(out_direc+"orig_pph_bicubic_shale.csv", sep = ",", index=False)

covariances_orig_df_backload = pd.read_csv(out_direc+"orig_pph_bicubic_shale.csv")
covariances_orig_df_backload.head()


######### Compute two_point_correlation for grain phase ########

two_point_correlation_grain_phase_orig = {}
for i, direc in enumerate(["x", "y", "z"]):
    two_point_direc = two_point_correlation(orig_img, i, var=grain_phase)
    two_point_correlation_grain_phase_orig[direc] = two_point_direc

direc_correlation_grain_phase_orig = {}
for direc in ["x", "y", "z"]:
    direc_correlation_grain_phase_orig[direc] = np.mean(np.mean(two_point_correlation_grain_phase_orig[direc], axis=0), axis=0)
print(direc_correlation_grain_phase_orig["x"].shape)

covariance_orig_df = pd.DataFrame(direc_correlation_grain_phase_orig)
covariance_orig_df.to_csv(out_direc+"orig_gph_bicubic_shale.csv", sep = ",", index=False)

covariances_orig_df_backload = pd.read_csv(out_direc+"orig_gph_bicubic_shale.csv")
covariances_orig_df_backload.head()
