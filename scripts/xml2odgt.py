import json
import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import cv2
import random
import string
from IPython import embed

dataset_path = '/home/xiongfeng/dataset/data_training/'
dac_all = open('/home/xiongfeng/dataset/dac_all.odgt', 'w')

tmp = dict()
for path in tqdm(glob.glob(os.path.join(dataset_path, '*'))):
    classes_name = path.replace(dataset_path, '')
    for gt_file in glob.glob(os.path.join(path, '*.xml')):
        gt_img = gt_file.replace('.xml', '.jpg')
        tree = ET.ElementTree(file=gt_file)
        b=np.zeros(4, dtype=float)
        for elem in tree.iter():
            if(elem.tag == 'xmin'):
                b[0] = float(elem.text)
            if(elem.tag == 'xmax'):
                b[1] = float(elem.text)
            if(elem.tag == 'ymin'):
                b[2] = float(elem.text)
            if(elem.tag == 'ymax'):
                b[3] = float(elem.text)
        gt = [b[0], b[2], (b[1] - b[0] + 1), (b[3] - b[2] + 1)]

        gt_area = max(0, b[1] - b[0] + 1) * max(0, b[3] - b[2] + 1)

        extra = {'ignore': 0}
        tag = gt_img.split('/')[-2]
        gt_info = [{'extra': extra, 'box': gt, 'tag': 'foreground'}]
        tmp['fpath'] = gt_img
        tmp['gtboxes'] = gt_info
        dac_all.write(json.dumps(tmp) + '\n')


### random split  ###

with open('/home/xiongfeng/dataset/dac_all.odgt', 'r') as f:
    files = f.readlines()

train_dataset = open('/home/xiongfeng/dataset/dac_train.odgt', 'w')
val_dataset = open('/home/xiongfeng/dataset/dac_val.odgt', 'w')

split_ratio = 0.9
num_data = len(files)

random.shuffle(files)
for i, item in tqdm(enumerate(files)):
    item_info = json.loads(item)
    if i < split_ratio * num_data:
        train_dataset.write(json.dumps(item_info) + '\n')
    else:
        val_dataset.write(json.dumps(item_info) + '\n')
