import os
import random
import pickle
import sys

import bpy
from sacred import Experiment
import cv2
import numpy as np
from mathutils import Matrix

root = '.'
sys.path.insert(0, root)
mano_path = os.environ.get('MANO_LOCATION', None)
if mano_path is None:
    raise ValueError('Environment variable MANO_LOCATION not defined'
                     'Please follow the README.md instructions')
sys.path.insert(0, os.path.join(mano_path, 'webuser'))

from obman_render import (mesh_manip, render, texturing, conditions, imageutils,
                         camutils, coordutils, depthutils)
from smpl_handpca_wrapper import load_model as smplh_load_model
from serialization import load_model

ex = Experiment('generate_dataset')

@ex.config
def exp_config():
    # Path to folder where to render
    results_root = 'results'
    # in ['train', 'test', 'val']
    split = 'train'
    # Number of frames to render
    frame_nb = 1
    # Idx of first frame
    frame_start = 0
    # Min distance to camera
    z_min = 0.5
    # Max distance to camera
    z_max = 0.8
    # Zoom to increase resolution of textures
    texture_zoom = 1
    # combination of [imagenet|lsun|pngs|jpgs|with|4096]
    texture_type = ['bodywithands']
    # Render full bodys and save body annotation
    render_body = False
    high_res_hands = False
    # Combination of [black|white|imagenet|lsun]
    background_datasets = ['imagenet', 'lsun']
    # Paths to background datasets
    lsun_path = '/sequoia/data2/gvarol/datasets/LSUN/data/img'
    imagenet_path = '/sequoia/data3/datasets/imagenet'
    # Lighting ambiant mean
    ambiant_mean = 0.7
    # Lighting ambiant add
    ambiant_add = 0.5

    # hand params
    pca_comps = 6
    # Pose params are uniform in [-hand_pose_var, hand_pose_var]
    hand_pose_var = 2
    # Path to fit folder
    smpl_data_path = 'assets/SURREAL/smpl_data/smpl_data.npz'
    mano_path = mano_path
    smpl_model_path = os.path.join(mano_path, 'models', 'SMPLH_female.pkl')
    mano_right_path = os.path.join(mano_path, 'models', 'MANO_RIGHT.pkl')


@ex.automain
def run(_config, results_root, split, frame_nb, frame_start, z_min, z_max,
        texture_zoom, texture_type, render_body, high_res_hands,
        background_datasets, lsun_path, imagenet_path, ambiant_mean, ambiant_add,
        hand_pose_var, pca_comps, smpl_data_path, smpl_model_path,
        mano_right_path):
    print(_config)
    scene = bpy.data.scenes['Scene']
    # Clear default scene cube
    bpy.ops.object.delete()

    # Set results folders
    folder_meta = os.path.join(results_root, 'meta')
    folder_rgb = os.path.join(results_root, 'rgb')
    folder_segm = os.path.join(results_root, 'segm')
    folder_temp_segm = os.path.join(results_root, 'tmp_segm')
    folder_depth = os.path.join(results_root, 'depth')
    folders = [
        folder_meta, folder_rgb, folder_segm, folder_temp_segm, folder_depth
    ]
    # Create results directories
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # Load smpl2mano correspondences
    right_smpl2mano = np.load('assets/models/smpl2righthand_verts.npy')
    left_smpl2mano = np.load('assets/models/smpl2lefthand_verts.npy')

    # Load SMPL+H model
    ncomps = 2 * pca_comps  # 2x6 for 2 hands and 6 PCA components
    smplh_model = smplh_load_model(
        smpl_model_path, ncomps=ncomps, flat_hand_mean=False)
    camutils.set_camera()

    backgrounds = imageutils.get_image_paths(
        background_datasets, split=split, lsun_path=lsun_path,
        imagenet_path=imagenet_path)
    print('Got {} backgrounds'.format(len(backgrounds)))

    # Get full body textures
    body_textures = imageutils.get_bodytexture_paths(
        texture_type, split=split, lsun_path=lsun_path,
        imagenet_path=imagenet_path)
    print('Got {} body textures'.format(len(body_textures)))

    # Get high resolution hand textures
    if high_res_hands:
        hand_textures = imageutils.get_hrhand_paths(texture_type, split=split)
        print('Got {} high resolution hand textures'.format(
            len(hand_textures)))
    print('Finished loading textures')

    # Load smpl info
    smpl_data = np.load(smpl_data_path)

    smplh_verts, faces = smplh_model.r, smplh_model.f
    smplh_obj = mesh_manip.load_smpl()
    # Smooth the edges of the body model
    bpy.ops.object.shade_smooth()

    # Set camera rendering params
    scene.render.resolution_x = 256
    scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100

    # Get camera info
    cam_calib = np.array(camutils.get_calib_matrix())
    cam_extr = np.array(camutils.get_extrinsic())

    scs, materials, sh_path = texturing.initialize_texture(
        smplh_obj, texture_zoom=texture_zoom, tmp_suffix='tmp')

    sides = ['right', 'left']

    # Create object material if none is present
    print('Starting loop !')

    for i in range(frame_nb):
        frame_idx = i + frame_start
        np.random.seed(frame_idx)
        random.seed(frame_idx)
        tmp_files = []  # Keep track of temporary files to delete at the end

        # Sample random hand poses
        side = random.choice(sides)
        hand_pose = None
        hand_pose_offset = 3

        smplh_verts, posed_model, meta_info = mesh_manip.randomized_verts(
            smplh_model,
            smpl_data,
            ncomps=ncomps,
            hand_pose=hand_pose,
            z_min=z_min,
            z_max=z_max,
            side=side,
            hand_pose_offset=hand_pose_offset,
            pose_var=hand_pose_var,
            random_shape=True,
            random_pose=True,
            split=split)
        mesh_manip.alter_mesh(smplh_obj, smplh_verts.tolist())

        hand_info = coordutils.get_hand_body_info(
            posed_model,
            render_body=render_body,
            side='right',
            cam_extr=cam_extr,
            cam_calib=cam_calib,
            right_smpl2mano=right_smpl2mano,
            left_smpl2mano=left_smpl2mano)
        hand_infos = {**hand_info, **meta_info}

        frame_prefix = '{:08}'.format(frame_idx)
        camutils.set_camera()
        camera_name = 'Camera'
        # Randomly pick background
        bg_path = random.choice(backgrounds)
        depth_path = os.path.join(folder_depth, frame_prefix)
        tmp_segm_path = render.set_cycle_nodes(
            scene, bg_path, segm_path=folder_temp_segm, depth_path=depth_path)
        tmp_files.append(tmp_segm_path)
        tmp_depth = depth_path + '{:04d}.exr'.format(1)
        tmp_files.append(tmp_depth)
        # Randomly pick clothing texture
        tex_path = random.choice(body_textures)

        # Replace high res hands
        if high_res_hands:
            old_state = random.getstate()
            old_np_state = np.random.get_state()
            hand_path = random.choice(hand_textures)
            tex_path = texturing.get_overlaped(tex_path, hand_path)
            tmp_files.append(tex_path)
            # Restore previous seed state to not interfere with randomness
            random.setstate(old_state)
            np.random.set_state(old_np_state)

        sh_coeffs = texturing.get_sh_coeffs(
            ambiant_mean=ambiant_mean, ambiant_max_add=ambiant_add)
        texturing.set_sh_coeffs(scs, sh_coeffs)

        # Update body+hands image
        tex_img = bpy.data.images.load(tex_path)
        for part, material in materials.items():
            material.node_tree.nodes['Image Texture'].image = tex_img

        # Render
        img_path = os.path.join(folder_rgb, '{}.jpg'.format(frame_prefix))
        scene.render.filepath = img_path
        scene.render.image_settings.file_format = 'JPEG'
        bpy.ops.render.render(write_still=True)

        camutils.check_camera(camera_name=camera_name)
        segm_img = cv2.imread(tmp_segm_path)[:, :, 0]
        if render_body:
            keep_render = True
        else:
            keep_render = conditions.segm_condition(
                segm_img, side=side, use_grasps=False)
        depth, depth_min, depth_max = depthutils.convert_depth(tmp_depth)

        hand_infos['depth_min'] = depth_min
        hand_infos['depth_max'] = depth_max
        hand_infos['bg_path'] = bg_path
        hand_infos['sh_coeffs'] = sh_coeffs
        hand_infos['body_tex'] = tex_path

        # Clean residual files
        if keep_render:
            # Write depth image
            final_depth_path = os.path.join(folder_depth,
                                            '{}.png'.format(frame_prefix))
            cv2.imwrite(final_depth_path, depth)

            # Save meta
            meta_pkl_path = os.path.join(folder_meta,
                                         '{}.pkl'.format(frame_prefix))
            with open(meta_pkl_path, 'wb') as meta_f:
                pickle.dump(hand_infos, meta_f)

            # Write segmentation path
            segm_save_path = os.path.join(folder_segm,
                                          '{}.png'.format(frame_prefix))
            cv2.imwrite(segm_save_path, segm_img)
            ex.log_scalar('generated.idx', frame_idx)
        else:
            os.remove(img_path)
        for filepath in tmp_files:
            os.remove(filepath)
    print('DONE')
