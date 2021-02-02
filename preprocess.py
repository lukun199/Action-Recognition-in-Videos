from __future__ import print_function, division
import os
import sys
import cv2
import argparse
import json
import tqdm
import h5py
import skimage.io
import os
from collections import Counter
import torch
import platform
import numpy as np
import torchvision

from encoder import Res3d

glob_cnt = 0
blob_class = 0
abnormal_length = 0


def class_process(dir_path, dst_dir_path, class_name, f):
    global glob_cnt
    global blob_class
    global abnormal_length
    class_path = os.path.join(dir_path, class_name)
    dst_path = dst_dir_path
    if not os.path.exists(dst_path): os.makedirs(dst_path)

    for idx, file_name in enumerate(os.listdir(class_path)):
        assert '.avi' in file_name
        video_file_path = os.path.join(class_path, file_name)
        video_cap = cv2.VideoCapture(video_file_path)
        total_frams = video_cap.get(7)
        if total_frams > (16 + 8):
            selected_frames = np.linspace(4, total_frams - 4, 16).astype(int)
        else:
            abnormal_length += 1
            print('------------------ABNORMAL---------------', video_file_path)
            continue

        list_thisFrame = list()
        for idx_frame in selected_frames:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
            is_done, frame_this = video_cap.read()
            assert is_done
            frame_this = cv2.cvtColor(frame_this, cv2.COLOR_BGR2RGB)
            frame_this = cv2.resize(frame_this, (160, 160))
            list_thisFrame.append(frame_this[np.newaxis, :])

        f.create_dataset('c%03d_' % blob_class + class_name + '_%03d' % idx, data=np.vstack(list_thisFrame))
        glob_cnt += 1

        print('the %5d th: ' % glob_cnt, 'c%03d_' % blob_class + class_name + '_%03d' % idx)

    blob_class += 1


def flip_key(key):
    # c000_sadsa_000
    return key[:-3] + '9' + key[-2:]


def small_extract_imgs_feat(opt_in, in_pth):

    encoder = Res3d()
    encoder.to(opt_in.device)
    encoder.eval()

    dataset = h5py.File(in_pth + 'small_images_val.h5', 'r')

    with h5py.File(os.path.join(in_pth, 'processed_val/small_feats_val.h5')) as file_fc_feats:

        for idx, key in enumerate(dataset.keys()):
            frames = dataset[key].value  # frames: [16, H, W, C] in RGB
            frame_flip = np.flip(frames,2)

            small_extract_bboxes_rcnn(key, frames, opt_in, in_pth)
            small_extract_bboxes_rcnn(flip_key(key), frame_flip, opt_in, in_pth)

            with torch.no_grad():
                frames = encoder.preprocess(frames)
                frames = frames.to(opt.device)
                fc_feats = encoder(frames)

                frame_flip = encoder.preprocess(frame_flip)
                frame_flip = frame_flip.to(opt.device)
                fc_feats_flip= encoder(frame_flip)

            #file_conv_feats.create_dataset(key, data=conv_feats.cpu().half().numpy())
            file_fc_feats.create_dataset(key, data=fc_feats.cpu().half().numpy())
            file_fc_feats.create_dataset(flip_key(key), data=fc_feats_flip.cpu().half().numpy())
            #file_conv_feats.create_dataset(flip_key(key), data=conv_feats_flip.cpu().half().numpy())
            #file_fc_feats.create_dataset(flip_key(key), data=fc_feats_flip.cpu().float().numpy())

            if idx % 10 == 0:
                print('-----------%d-------------' % idx)


def small_extract_bboxes_rcnn(key, frames, opt_in, in_pth):

    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    detector.to(opt_in.device)
    detector.eval()

    with h5py.File(os.path.join(in_pth, 'processed_val/small_bboxes_val.h5')) as file_bboxes:

        lib_pth = []
        with torch.no_grad():
            for each_frame in frames:

                # ToTensor()
                each_frame = torch.from_numpy(each_frame/255.).permute(2,0,1).float().to(opt_in.device)
                bboxes = detector([each_frame])[0]

                # select the biggest bbox.
                condition = np.array(bboxes['scores'].cpu()>0.7) & np.array(bboxes['labels'].cpu()==1)
                human_idx = np.argwhere(condition == True).flatten()

                if human_idx.size >0:
                    tuple_positions = bboxes['boxes'].cpu().numpy().take(human_idx,axis=0)
                    tuple_pos_four = list(zip(*tuple_positions))
                    ptf = list(map(lambda  x: int(min(x)), tuple_pos_four[:2])) + \
                                list(map(lambda x: int(max(x)), tuple_pos_four[2:]))
                    lib_pth.append(ptf)
                    #print(ptf)
                else: lib_pth.append([0,0,160,160])

        each_coordinate = list(zip(*lib_pth))
        each_coordinate = list(map(lambda x:int(sum(x)/len(x)), each_coordinate))

        file_bboxes.create_dataset(key, data= np.array(each_coordinate))
        # file_conv_feats.create_dataset(flip_key(key), data=conv_feats_flip.cpu().half().numpy())
        #file_fc_feats.create_dataset(flip_key(key), data=save_fc_feats_flip.cpu().float().numpy())

        print('------bboxes of-----%s----:  '%key, each_coordinate)

if __name__ == '__main__':

    # process video

    dir_path = '/home1/lk/Videos/Datasets/Pose/val_data_hmdb51'
    dst_dir_path = '../dataset/'
    """
    dir_path = '../dataset/video_data/'
    dst_dir_path = '../dataset/'
    """

    with h5py.File(os.path.join(dst_dir_path, 'small_images_val.h5'), 'w') as file_image:
        for class_name in os.listdir(dir_path):
            class_process(dir_path, dst_dir_path, class_name, file_image)

    print('----------------ABNORNAM--ONES-------------', abnormal_length)


    # print(1/0)

    # generate features.
    parser = argparse.ArgumentParser()

    if 'Windows' in platform.system():

        parser.add_argument('--resnet101_file', type=str,
                            default=r'E:\DOC\RESEARCH\MASTER\Semantic\Codes\resnet101.pth')
        opt = parser.parse_args()
        opt.device = torch.device('cpu')

        small_extract_imgs_feat_rcnn(opt, in_pth=dst_dir_path)

    else:

        parser.add_argument('--resnet101_file', type=str,
                            default='/home1/lk/Videos/Codes/reference/resnet101.pth')
        opt = parser.parse_args()
        opt.device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

        small_extract_imgs_feat(opt, in_pth=dst_dir_path)

        print('dataset done!')


