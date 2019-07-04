import json
import os
import os.path as osp
from tqdm import tqdm
import struct
import imghdr
from IPython import embed

def get_image_size(fname):
    '''Determine the image type of fhandle and return its size.
    from draco'''
    with open(fname, 'rb') as fhandle:
        head = fhandle.read(24)
        if len(head) != 24:
            return
        if imghdr.what(fname) == 'png':
            check = struct.unpack('>i', head[4:8])[0]
            if check != 0x0d0a1a0a:
                return
            width, height = struct.unpack('>ii', head[16:24])
        elif imghdr.what(fname) == 'gif':
            width, height = struct.unpack('<HH', head[6:10])
        elif imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0)  # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception:  # IGNORE:W0703
                return
        else:
            return
        return width, height

classes_originID = {'foreground':1}

# source path is odgt format, target format is coco format
def convert(source_path, target_path):
    with open(source_path, 'r') as f:
        lines = f.readlines()
    wt = open(target_path, 'w')

    categories = [{'supercategory': 'foreground', 'id': 1, 'name': 'foreground'}]
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": categories}

    img_id = 1
    box_id = 1
    for line in tqdm(lines):
        gt = json.loads(line.strip())

        # image
        filename = gt['fpath']
        width, height = get_image_size(filename)
        image = dict(file_name=filename, height=height, width=width, id=img_id)
        json_dict['images'].append(image)

        # annotation
        for box in gt['gtboxes']:
            category_id = classes_originID[box['tag']]
            ignore = 0
            if 'ignore' in box['extra'].keys():
                if box['extra']['ignore'] == 1:
                    ignore = 1
            b = box['box']
            ann = {'area': b[2] * b[3], 'iscrowd': 0, 'image_id': img_id,
                   'bbox': b, 'category_id': category_id,
                   'id': box_id, 'ignore': ignore, 'segmentation': []}
            json_dict['annotations'].append(ann)
            box_id += 1
        img_id += 1

    wt.write(json.dumps(json_dict))
    print('save in {}'.format(target_path))
    wt.close()


if __name__ == '__main__':
    odgt_path = {'train': '/home/xiongfeng/dataset/dac_train.odgt',
                 'val': '/home/xiongfeng/dataset/dac_val.odgt'}

    save_dir = '/home/xiongfeng/dataset/'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    for dset in ['train', 'val']:
        convert(odgt_path[dset], osp.join(save_dir, 'dac_' + dset + '.json'))
