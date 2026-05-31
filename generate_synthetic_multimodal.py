import os
import numpy as np
import pandas as pd

os.makedirs('data/processed', exist_ok=True)

# Image modality: 784 features (28x28), binary label
n_img = 200
img_features = np.random.randn(n_img, 784)
img_labels = (img_features[:, 0] + img_features[:, 100] > 0).astype(int)
df_img = pd.DataFrame(img_features, columns=[f'feature_{i}' for i in range(784)])
df_img['label'] = img_labels
df_img.to_csv('data/processed/processed_images.csv', index=False)

# Audio modality: 128 features, binary label
n_aud = 200
aud_features = np.random.randn(n_aud, 128)
aud_labels = (aud_features[:, 0] - aud_features[:, 50] > 0).astype(int)
df_aud = pd.DataFrame(aud_features, columns=[f'feature_{i}' for i in range(128)])
df_aud['label'] = aud_labels
df_aud.to_csv('data/processed/processed_audios.csv', index=False)

# Video modality: 1024 features (32x32), binary label
n_vid = 200
vid_features = np.random.randn(n_vid, 1024)
vid_labels = (vid_features[:, 0] + vid_features[:, 500] < 0).astype(int)
df_vid = pd.DataFrame(vid_features, columns=[f'feature_{i}' for i in range(1024)])
df_vid['label'] = vid_labels
df_vid.to_csv('data/processed/processed_videos.csv', index=False)

print('Synthetic multimodal datasets generated:')
print(' - data/processed/processed_images.csv')
print(' - data/processed/processed_audios.csv')
print(' - data/processed/processed_videos.csv')
