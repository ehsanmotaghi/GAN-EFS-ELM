import pandas as pd
import os

# Replace the two points (..) with the file path.

df_39 = pd.read_csv('../original_tt_iot_net_combine_39.csv')
# df_39.shape   # (472162, 39)
# df_39 for fs has alot of processing time --> by simulator --> undersampling: 10430

## No-GAN, No-EFS
df_res_39 = pd.read_csv('../No-GAN_No-EFS(resampled_39).csv')
# df_res_39.shape   # (3174, 39)

## No-GAN, Yes-EFS
df_res_fs_14 = pd.read_csv('../No-GAN_Yes-EFS(resampled_39_efs_14).csv')
# df_res_fs_14.shape   # (3174, 16)

## Yes-GAN, No-EFS
df_gan_39 = pd.read_csv('../Yes-GAN_No-EFS(cgan_balanced_39).csv')
# df_gan_39.shape   # (10430, 39)

## Yes-GAN, Yes-EFS
df_gan_fs_12 = pd.read_csv('../Yes-GAN_Yes-EFS(cgan_balanced_39_efs_12).csv')
# df_gan_fs_12.shape   # (10430, 14)