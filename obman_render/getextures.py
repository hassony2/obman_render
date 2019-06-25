from copy import deepcopy
import os
import random

import cv2
import numpy as np


def get_all_textures_imagenet(root='/sequoia/data3/datasets/imagenet',
                              file_path='assets/textures/imagenet_train.txt'):

    with open(file_path) as f:
        lines = f.readlines()

    all_textures = []
    for line in lines:
        all_textures.append(line.strip())

    all_textures = [os.path.join(root, texture) for texture in all_textures]
    return all_textures


def get_all_textures_shapenet(
        split_folder='assets/textures/shapenet',
        shapenet_folder='/sequoia/data2/dataset/shapenet/ShapeNetCore.v2',
        split='train'):
    split_textures = os.path.join(split_folder,
                                  'textures_{}.txt'.format(split))
    print(split_textures)
    with open(split_textures) as tex_f:
        lines = tex_f.readlines()
    all_textures = [
        img_path.strip().replace(
            '/sequoia/data2/dataset/shapenet/ShapeNetCore.v2', shapenet_folder)
        for img_path in lines
    ]
    print('Got {} textures fot split {} from shapenet'.format(
        len(all_textures), split))
    return all_textures


def get_high_res_hands(
        folder_bought_template='misc/ten24_TexturesHands_R_L_RL_{:04d}/',
        folder_inhouse_template='misc/may_2018_inhouse_mpi_hands_{:04d}/',
        res=512,
        split='train',
        hand_type='BOTH',
        texture_types=['in_house', 'bought']):
    """
    Args:
        hand_type (str): [BOTH|RIGHT|LEFT]
    """
    hand_types = ['BOTH', 'RIGHT', 'LEFT']
    all_hand_paths = []
    if 'bought' in texture_types:
        folder = folder_bought_template.format(res)
        assert hand_type in hand_types, 'hand_type {} not in {}'.format(
            hand_type, hand_types)
        all_hands = sorted(os.listdir(folder))
        hand_paths = [
            os.path.join(folder, hand) for hand in all_hands
            if hand_type in hand
        ]
        if split == 'test':
            all_hand_paths.extend(hand_paths[0])
        elif split == 'val':
            all_hand_paths.extend([hand_paths[1]])
        else:
            all_hand_paths.extend(hand_paths[2:])
    if 'in_house' in texture_types:
        folder = folder_inhouse_template.format(res)
        assert hand_type in hand_types, 'hand_type {} not in {}'.format(
            hand_type, hand_types)
        all_hands = sorted(os.listdir(folder))
        hand_paths = [
            os.path.join(folder, hand) for hand in all_hands
            if hand_type in hand
        ]
        if split == 'test':
            all_hand_paths.extend(hand_paths[0:2])
        elif split == 'val':
            all_hand_paths.extend([hand_paths[2]])
        else:
            all_hand_paths.extend(hand_paths[3:])

    return all_hand_paths


def get_overlaped(body_path, hand_path, tmp_folder='misc/tmp'):
    img_hands = cv2.imread(hand_path)
    img_body = cv2.imread(body_path)
    assert img_hands.shape == img_body.shape, 'Hand texture and body texture have != shapes {} and {}'.format(
        img_hands.shape, img_body.shape)
    img_overlap = deepcopy(img_body)
    hand_ids = np.where(np.logical_not(img_hands == np.array([0, 0, 0])))
    img_overlap[hand_ids] = img_hands[hand_ids]
    img_name = '{:030x}.jpg'.format(random.randrange(16**30))
    img_path = os.path.join(tmp_folder, img_name)
    os.makedirs(tmp_folder, exist_ok=True)
    cv2.imwrite(img_path, img_overlap)
    return img_path


def get_all_textures_bodywithands(folder='assets/textures/bodywithands',
                                  split='train'):
    split_folder = os.path.join(folder, split)
    all_textures = [
        os.path.join(split_folder, tex) for tex in os.listdir(split_folder)
    ]
    print('Got {} body+hand textrures for split {}'.format(
        len(all_textures), split))
    return all_textures


def get_all_textures_jpg(
        folder='/sequoia/data1/gvarol/home/github/surreal/datageneration/smpl_data/',
        split='train'):
    if split == 'train' or split == 'val':
        all_female = os.path.join(folder, 'textures', 'female_train.txt')
        all_male = os.path.join(folder, 'textures', 'male_train.txt')
    else:
        all_female = os.path.join(folder, 'textures', 'female_test.txt')
        all_male = os.path.join(folder, 'textures', 'male_test.txt')

    all_textures = []
    with open(all_female) as f:
        lines = f.readlines()
        if split == 'train':
            lines = lines[::4] + lines[1::4] + lines[2::4]
        if split == 'val':
            lines = lines[3::4]
    for line in lines:
        all_textures.append(line.strip())
    with open(all_male) as f:
        lines = f.readlines()
        if split == 'train':
            lines = lines[::4] + lines[1::4] + lines[2::4]
        if split == 'val':
            lines = lines[3::4]
    for line in lines:
        all_textures.append(line.strip())
    all_textures = [os.path.join(folder, texture) for texture in all_textures]
    return all_textures
