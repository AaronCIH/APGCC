# ref: https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/issues/8
# cmd: python pre_process_label.py src_path dataset output_path
from scipy.io import loadmat
import os
import argparse
from tqdm import tqdm
import json

def get_points(root_path, mat_path):
    m = loadmat(os.path.join(root_path, mat_path))
    return m['image_info'][0][0][0][0][0]

def get_image_list(root_path, sub_path):
    images_path = os.path.join(root_path, sub_path, 'images')
    images = [os.path.join(images_path, im) for im in
    os.listdir(os.path.join(root_path, images_path)) if 'jpg' in im]
    return images

def get_gt_from_image(image_path):
    gt_path = os.path.dirname(image_path.replace('images', 'ground_truth'))
    gt_filename = os.path.basename(image_path)
    gt_filename = 'GT_{}'.format(gt_filename.replace('jpg', 'mat'))
    return os.path.join(gt_path, gt_filename)

def ShanghaiTech(root_path, part_name, output_path):
    if part_name not in ['A', 'B']:
        raise NotImplementedError('Supplied dataset part does not exist')

    dataset_splits = ['train', 'test']
    for split in dataset_splits:
        part_folder = 'part_{}'.format(part_name)
        if part_name == 'A':
            sub_path = os.path.join(part_folder, '{}'.format(split))
        elif part_name == 'B':
            sub_path = os.path.join(part_folder, '{}_data'.format(split))
        out_sub_path = os.path.join(part_folder, '{}_data'.format(split))

        images = get_image_list(root_path, sub_path=sub_path)
        try:
            os.makedirs(os.path.join(output_path, out_sub_path))
        except FileExistsError:
            print('Warning, output path already exists, overwriting')

        list_file = []
        for image_path in images:
            gt_path = get_gt_from_image(image_path)
            gt = get_points(root_path, gt_path)

            # for each image, generate a txt file with annotations
            new_labels_file = os.path.join(output_path, out_sub_path,
                                            os.path.basename(image_path).replace('jpg', 'txt'))
            with open(new_labels_file, 'w') as fp:
                for p in gt:
                    fp.write('{} {}\n'.format(p[0], p[1]))
            list_file.append((image_path, new_labels_file))

        # generate file with listing
        with open(os.path.join(output_path, part_folder,'{}.list'.format(split)), 'w') as fp:
            for item in list_file:
                fp.write('{} {}\n'.format(item[0], item[1]))

def NWPU(root_path, output_path):
    dataset_splits = ['train.txt', 'val.txt', 'test.txt']
    for split in dataset_splits:
        with open(os.path.join(root_path, split)) as f:
            images = f.read().split('\n')[:-1]

        if split != 'test.txt':
            out_folder = os.path.join(output_path, split.split('.')[0])    # ./Dataset/NWPU/PointData/train
            print("DataFolder:", os.path.join(root_path, split), len(images))
            try:
                os.makedirs(out_folder)
            except FileExistsError:
                print('Warning, output path already exists, overwriting')

        list_file = []
        for image_data in tqdm(images):
            iid, l, s = image_data.split(' ')  # 0001 1 1
            img_path = os.path.join(root_path, 'images', iid+'.jpg') # ./Dataset/NWPU/images/0001.jpg

            if split != 'test.txt':
                # open label.json
                with open(os.path.join(root_path, 'jsons', iid+'.json')) as f: # ./Dataset/NWPU/jsons/0001.json
                    img_info = json.load(f)
                gt = img_info['points']
                
                # for each image, generate a txt file with annotations
                new_labels_file = os.path.join(out_folder, iid+'.txt')  # ./Dataset/NWPU/PointData/train/0001.txt
                with open(new_labels_file, 'w') as fp:
                    for p in gt:
                        fp.write('{} {}\n'.format(p[0], p[1]))
                list_file.append((img_path, new_labels_file, l, s))
            else:
                list_file.append((img_path, 'Nan', l, s))

        # generate file with listing
        with open(os.path.join(output_path, '{}.list'.format(split.split('.')[0])), 'w') as fp: # ./Dataset/NWPU/PointData/train.list
            for item in list_file:
                fp.write('{} {} {} {}\n'.format(item[0], item[1], item[2], item[3]))

def build_datalabel(root_path, dataset, output_path):
    if dataset == 'SHHA':
        ShanghaiTech(root_path, 'A', output_path)
    elif dataset == 'SHHB':
        ShanghaiTech(root_path, 'B', output_path)
    elif dataset == 'NWPU':
        NWPU(root_path, output_path)
    else:
        raise NotImplemented

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root_path", help="Root path for shanghai dataset")
    parser.add_argument("dataset", choices=['SHHA', 'SHHB', 'NWPU'], help="dataset name: ('SHHA', 'SHHB', 'NWPU')")
    parser.add_argument("output_path", help="Path to store results")
    args = parser.parse_args()
    build_datalabel(args.root_path, args.dataset, args.output_path)
