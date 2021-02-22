import torch
from tfrecord.torch.dataset import TFRecordDataset

tfrecord_path = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/tfrecords/train/0.tfrecord"
index_path = "/mnt/291d3084-ca91-4f28-8f33-ed0b64be0a8c/akshay/kitti/index/train/0_tfrecord.index"
description = {"image": "byte", "label": "float"}
dataset = TFRecordDataset(tfrecord_path, index_path)
loader = torch.utils.data.DataLoader(dataset, batch_size=32)

data = next(iter(loader))
print(data['x_dm'].dtype)
print(data['x_cam'].dtype)