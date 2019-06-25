import csv
import json
from functools import lru_cache
import os
import pickle

import numpy as np
from tqdm import tqdm

from obman_render.grasps.splitutils import read_split


def get_inv_hand_pca(mano_path='assets/models/MANO_RIGHT.pkl'):
    with open(mano_path, 'rb') as f:
        hand_r = pickle.load(f, encoding='latin1')
    hands_pca_r = hand_r['hands_components']
    inv_hand_pca = np.linalg.inv(hands_pca_r)
    return inv_hand_pca


def read_grasp_file(
        split_path='assets/grasps/shapenet_grasps_split.csv',
        filepath='assets/grasps/shapenet_grasps/meshnet_02876657_1071fa4cddb2da2fc8724d5673a063a6_scale_125.0.json',
        split='train',
        mano_path='assets/models/MANO_RIGHT.pkl',
        filter_classes=None):
    inv_hand_pca = get_inv_hand_pca(mano_path=mano_path)
    split_data = read_split(split_path)
    grasp_info = []
    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        for row_idx, row in enumerate(reader):
            if row_idx > 0:
                class_id = row['category_id']
                sample_id = row['object_id']
                hand_pose = np.array(eval(row['mano_pose']))
                # Get 45 hand joints axis-angle rotations
                hand_axisang = hand_pose[3:]
                # Get hand translation
                hand_trans = np.array(eval(row['trans']))
                assert (len(hand_pose) == 48)
                pose = np.array(eval(row['pose']))
                if (class_id, sample_id) in split_data[split]:
                    grasp_info.append({
                        'class_id':
                        class_id,
                        'sample_id':
                        sample_id,
                        'pose':
                        pose,
                        'hand_pose':
                        hand_pose,
                        'pca_pose':
                        np.array(hand_axisang).dot(inv_hand_pca),
                        'hand_global_rot':
                        hand_pose[:3],
                        'hand_trans':
                        hand_trans
                    })
    return grasp_info


def grasp_wrong(g,
                angle=94,
                indicies=[2, 5, 8, 11, 15],
                offsets=[10.5, 6.5, 8, 2.2, 0]):
    dofs = np.degrees(g['dofs'])
    medial_jnts = dofs[indicies] + offsets
    return np.any(medial_jnts > angle)


@lru_cache(maxsize=128)
def read_grasp_folder(
        filter_angle=94,
        grasp_nb=2,
        grasp_folder='assets/grasps/shapenet_grasps',
        mano_path='assets/models/MANO_RIGHT.pkl',
        obj_models='shapenet',
        split_path='assets/grasps/shapenet_grasps_splits.csv',
        split='train',
        shapenet_root='/sequoia/data2/dataset/shapenet/ShapeNetCore.v2',
        use_cache=False, ):
    """
    Args:
        grasp_nb: number of GRAsps to keep for each sample-scale pair
    """
    split_data = read_split(split_path)
    inv_hand_pca = get_inv_hand_pca(mano_path=mano_path)
    grasp_files = os.listdir(grasp_folder)
    grasp_info = []
    cache_folder = 'misc/{}_cache/'.format(obj_models)
    os.makedirs(cache_folder, exist_ok=True)
    cache_path = os.path.join(
        cache_folder, '2018_10_01_{}_filter_angle_{}_grasp_nb{}.pkl'.format(
            split, filter_angle, grasp_nb))
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, 'rb') as p_f:
            grasp_info = pickle.load(p_f)
        print('Loaded cache info from {}'.format(cache_path))
    else:
        grasp_files = [
            grasp_file for grasp_file in grasp_files if '.json' in grasp_file
        ]
        for grasp_file in tqdm(grasp_files):
            grasp_path = os.path.join(grasp_folder, grasp_file)
            with open(grasp_path, 'r') as json_f:
                raw_line = json_f.read()
                grasps = json.loads(raw_line)
                grasp_list = grasps['grasps']
                for idx, grasp in enumerate(grasp_list[:grasp_nb]):
                    if not grasp_wrong(grasp, angle=filter_angle):
                        if obj_models == 'shapenet':
                            object_cat = grasps['object_cat']
                            object_id = grasps['object_id']
                            shapenet_template = os.path.join(
                                shapenet_root,
                                '{}/{}/models/model_normalized.obj')
                            obj_full_path = shapenet_template.format(
                                object_cat, object_id)
                            object_scale = grasps['object_scale']
                            sample_key = (object_cat, object_id)
                        if (split_data is None
                                or sample_key in split_data[split]):
                            grasp_info.append({
                                'obj_path':
                                obj_full_path,
                                'class_id':
                                object_cat,
                                'sample_id':
                                object_id,
                                'sample_scale':
                                object_scale,
                                'pose':
                                grasp['pose'],
                                'hand_pose':
                                grasp['mano_pose'],
                                'hand_trans':
                                grasp['mano_trans'][0],
                                'pca_pose':
                                np.array(
                                    grasp['mano_pose'][3:]).dot(inv_hand_pca),
                                'hand_global_rot':
                                grasp['mano_pose'][:3],
                                'grasp_quality':
                                grasp['quality'],
                                'grasp_epsilon':
                                grasp['epsilon'],
                                'grasp_volume':
                                grasp['volume']
                            })
        with open(cache_path, 'wb') as p_f:
            pickle.dump(grasp_info, p_f)
        print('Wrote cache info to {}'.format(cache_path))
    return grasp_info


if __name__ == '__main__':
    # grasp_info = read_grasp_file()
    grasp_info_train = read_grasp_folder()
    print(len(grasp_info_train))
    grasp_info_test = read_grasp_folder(split='test')
    print(len(grasp_info_test))
    grasp_info_val = read_grasp_folder(split='val')
    print(len(grasp_info_val))
