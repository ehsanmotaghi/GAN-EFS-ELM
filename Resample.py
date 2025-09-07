import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

data3 = pd.read_csv('../original_tt_iot_net_combine_39.csv')
features = data3.drop(['label', 'type'], axis =1)
features.shape   # (472162, 37)

# Replace two points (..) with the file path.

##########
# Check the count of category class
counts = data3.value_counts('type')
print(counts)
# let's see the distribution of our target category by bar chart
plt.figure(figsize=(10,5))
plt.xticks(rotation=45, ha='right')
ax = sns.countplot(x = 'type', data = data3, order=data3['type'].value_counts(ascending=False).index, palette = 'Set2')
for container in ax.containers:
    ax.bar_label(container)
ax.set_title("ToN_IoT Count of Normal and Attack Types")
##########

# 1. Calculate class counts in 'type' column
class_counts = data3['type'].value_counts()
normal_count = class_counts.get('normal', 0)

# 2. Calculate percentage of each class compared to 'normal' class
percent_of_normal = (class_counts / normal_count) * 100
print('Class counts:')
print(class_counts)
print('\nPercentage of normal class:')
print(percent_of_normal)

# 3. Downsample each class to the size of the smallest class
min_class_count = class_counts.min()
equal_sampled_data3 = data3.groupby('type').apply(lambda x: x.sample(min_class_count, random_state=42)).reset_index(drop=True)

print(f'Equal sampled data shape: {equal_sampled_data3.shape}')
print('New class counts:')
print(equal_sampled_data3['type'].value_counts())

# Calculate percent_of_normal for the balanced dataset
equal_class_counts = equal_sampled_data3['type'].value_counts()
equal_normal_count = equal_class_counts.get('normal', 0)
equal_percent_of_normal = (equal_class_counts / equal_normal_count) * 100
print('\nPercentage of normal class in balanced dataset:')
print(equal_percent_of_normal)

# Resample equal_sampled_data3 based on percent_of_normal
N = equal_class_counts.get('normal', 0)
resampled_frames = []
for cls, pct in percent_of_normal.items():
    num_samples = int((pct / 100) * N)
    cls_rows = equal_sampled_data3[equal_sampled_data3['type'] == cls]
    if len(cls_rows) > 0 and num_samples > 0:
        sampled = cls_rows.sample(min(num_samples, len(cls_rows)), random_state=42)
        resampled_frames.append(sampled)
resampled_data3 = pd.concat(resampled_frames).reset_index(drop=True)

print(f'After resampling based on percent_of_normal: {resampled_data3.shape}')
print(resampled_data3["type"].value_counts())
resampled_data3.dtypes
resampled_data3['type'].unique()
resampled_data3['label'].unique()

##########
# Check the count of category class
counts = resampled_data3.value_counts('type')
print(counts)
# let's see the distribution of our target category by bar chart
plt.figure(figsize=(10,5))
plt.xticks(rotation=45, ha='right')
ax = sns.countplot(x = 'type', data = resampled_data3, order=resampled_data3['type'].value_counts(ascending=False).index, palette = 'Set2')
for container in ax.containers:
    ax.bar_label(container)
ax.set_title("ToN_IoT_resampled Count of Normal and Attack Types")
##########

resampled_data3.to_csv('../No-GAN_No-EFS(resampled_39).csv', index=False)
