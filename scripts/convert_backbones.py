import argparse
import re
from collections import OrderedDict

import torch
from IPython import embed

def convert(in_file, out_file):
    checkpoint = torch.load(in_file)
    if 'state_dict' in checkpoint:
        in_state_dict = checkpoint.pop('state_dict')
    else:
        in_state_dict = checkpoint
    out_state_dict = OrderedDict()
    for k, v in in_state_dict.items():
        key = k
        if k.split('.')[0] == 'module':
            new_key = k[7:]
        else:
            new_key = k
        out_state_dict[new_key] = v
    torch.save(out_state_dict, out_file)


def main():
    parser = argparse.ArgumentParser(description='Upgrade model version')
    parser.add_argument('in_file', help='input checkpoint file')
    parser.add_argument('out_file', help='output checkpoint file')
    args = parser.parse_args()
    convert(args.in_file, args.out_file)


if __name__ == '__main__':
    main()