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

from obman_render.grasps.grasputils import read_grasp_folder
from obman_render import (mesh_manip, render, texturing, conditions, imageutils,
                         camutils, coordutils, depthutils)
from obman_render.blenderobj import load_obj_model, delete_obj_model
from serialization import load_model
from smpl_handpca_wrapper import load_model as smplh_load_model

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
    # Combination of [black|white|imagenet|lsun]
    background_datasets = ['imagenet', 'lsun']
    # Paths to background datasets
    lsun_path = '/sequoia/data2/gvarol/datasets/LSUN/data/img'
    imagenet_path = '/sequoia/data3/datasets/imagenet'
    obj_tex_datasets = ['shapenet']
    # Lighting ambiant mean
    ambiant_mean = 0.7
    # Lighting ambiant add
    ambiant_add = 0.5
    # Grasp params
    grasp_nb = 2
    grasp_folder = 'assets/grasps/shapenet_grasps'
    grasp_split_path = 'assets/grasps/shapenet_grasps_splits.csv'
    random_obj_textures = True
    shapenet_root = '/sequoia/data2/dataset/shapenet/ShapeNetCore.v2'
    # Minimum ratio of object visibility
    min_obj_ratio = 0.4

    smpl_data_path = 'assets/SURREAL/smpl_data/smpl_data.npz'
    mano_path = 'assets'
    smpl_model_path = os.path.join(mano_path, 'models', 'SMPLH_female.pkl')
    mano_right_path = os.path.join(mano_path, 'models', 'MANO_RIGHT.pkl')


@ex.automain
def run(results_root, split, frame_nb, frame_start, z_min, z_max, texture_zoom,
        texture_type, render_body, background_datasets, ambiant_mean,
        ambiant_add, grasp_folder, grasp_split_path, min_obj_ratio, _config,
        obj_tex_datasets, random_obj_textures, grasp_nb, lsun_path,
        smpl_data_path, smpl_model_path, mano_right_path, shapenet_root,
        imagenet_path):
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
    folder_rgb_hand = os.path.join(results_root, 'rgb_hand')
    folder_rgb_obj = os.path.join(results_root, 'rgb_obj')
    folder_depth_hand = os.path.join(results_root, 'depth_hand')
    folder_depth_obj = os.path.join(results_root, 'depth_obj')
    folders.extend([folder_rgb_hand, folder_rgb_obj])

    # Create results directories
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

    # Load smpl2mano correspondences
    right_smpl2mano = np.load('assets/models/smpl2righthand_verts.npy')

    # Load SMPL+H model and grasp infos
    ncomps = 45
    grasp_info = read_grasp_folder(
        grasp_folder=grasp_folder,
        shapenet_root=shapenet_root,
        split_path=grasp_split_path,
        split=split,
        filter_angle=94,
        grasp_nb=grasp_nb,
        mano_path=mano_right_path,
        obj_models='shapenet',
        use_cache=True)

    print('Loaded grasp info for {} grasps'.format(len(grasp_info)))
    smplh_model = smplh_load_model(
        smpl_model_path, ncomps=2 * ncomps, flat_hand_mean=True)
    mano_model = load_model(mano_right_path)
    mano_mesh = bpy.data.meshes.new('Mano')
    mano_mesh.from_pydata(list(np.array(mano_model.r)), [], list(mano_model.f))
    mano_obj = bpy.data.objects.new('Mano', mano_mesh)
    bpy.context.scene.objects.link(mano_obj)
    mano_obj.hide_render = True

    print('Loaded mano model')
    camutils.set_camera()

    backgrounds = imageutils.get_image_paths(
        background_datasets, split=split, lsun_path=lsun_path,
        imagenet_path=imagenet_path)
    print('Got {} backgrounds'.format(len(backgrounds)))

    # Get full body textures
    body_textures = imageutils.get_bodytexture_paths(
        texture_type, split=split, lsun_path=lsun_path, imagenet_path=imagenet_path)
    print('Got {} body textures'.format(len(body_textures)))

    obj_textures = imageutils.get_image_paths(
        obj_tex_datasets,
        split=split,
        shapenet_folder=shapenet_root,
        lsun_path=lsun_path,
        imagenet_path=imagenet_path)
    print('Got {} object textures'.format(len(obj_textures)))

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

    # Create object material if none is present
    print('Starting loop !')

    for i in range(frame_nb):
        frame_idx = i + frame_start
        np.random.seed(frame_idx)
        random.seed(frame_idx)
        tmp_files = []  # Keep track of temporary files to delete at the end

        grasp = random.choice(grasp_info)

        if 'mano_trans' in grasp:
            mano_model.trans[:] = [val for val in grasp['mano_trans']]
        else:
            mano_model.trans[:] = grasp['hand_trans']
        mano_model.pose[:] = grasp['hand_pose']
        mesh_manip.alter_mesh(mano_obj, mano_model.r.tolist())

        smplh_verts, posed_model, meta_info = mesh_manip.randomized_verts(
            smplh_model,
            smpl_data,
            ncomps=2 * ncomps,
            z_min=z_min,
            z_max=z_max,
            side='right',
            hand_pose=grasp['pca_pose'],
            hand_pose_offset=0,
            random_shape=False,
            random_pose=True,
            split=split)

        # Center mesh on center_idx
        mesh_manip.alter_mesh(smplh_obj, smplh_verts.tolist())

        # Load object
        obj_path = grasp['obj_path']
        obj = load_obj_model(obj_path)
        obj_scale = float(grasp['sample_scale']) / 1000
        obj.scale = (obj_scale, obj_scale, obj_scale)
        obj.rotation_euler = (0, 0, 0)
        bpy.ops.object.shade_smooth()

        model_name = obj.name
        obj_mesh = bpy.data.meshes[model_name]
        obj_scs = []

        # Create object material if none is present
        materials_tmp = []
        if len(obj_mesh.materials) == 0:
            mat = bpy.data.materials.new(name='{}_mat'.format(obj_mesh.name))
            bpy.ops.object.material_slot_add()
            obj.material_slots[0].material = mat
        for mat_idx, obj_mat in enumerate(obj_mesh.materials):
            materials_tmp.append(obj_mat)
            if random_obj_textures:
                obj_texture = random.choice(obj_textures)
                generated_uv = True
            else:
                obj_texture = os.path.join(
                    os.path.dirname(obj_path), 'texture.jpg')
                generated_uv = False
            obj_sh_path = texturing.add_obj_texture(
                obj_mat,
                obj_texture,
                sh_path,
                down_scale=texture_zoom,
                tmp_suffix='tmp',
                generated_uv=generated_uv)
            tmp_files.append(obj_sh_path)
            tmp_files.append(obj_sh_path.replace('.osl', '.oso'))
            obj_scs.append(obj_mat.node_tree.nodes['Script'])
            obj_scs[-1].update()

        # Apply transform to object
        rigid_transform = coordutils.get_rigid_transform_posed_mano(
            posed_model, mano_model)
        mano_obj.matrix_world = Matrix(rigid_transform)

        obj_transform = rigid_transform.dot(obj.matrix_world)
        obj.matrix_world = Matrix(obj_transform)
        obj.scale = (obj_scale, obj_scale, obj_scale)

        hand_info = coordutils.get_hand_body_info(
            posed_model,
            render_body=render_body,
            side='right',
            cam_extr=cam_extr,
            cam_calib=cam_calib,
            right_smpl2mano=right_smpl2mano)
        frame_prefix = '{:08}'.format(frame_idx)

        # Save object info
        hand_info['affine_transform'] = obj_transform.astype(np.float32)
        if random_obj_textures:
            hand_info['obj_texture'] = obj_texture
        hand_info['obj_path'] = obj_path
        hand_info['obj_scale'] = obj_scale

        # Save grasp info
        for label in [
                'sample_id', 'class_id', 'pca_pose', 'grasp_quality',
                'grasp_epsilon', 'grasp_volume', 'hand_trans',
                'hand_global_rot', 'hand_pose'
        ]:
            hand_info[label] = grasp[label]

        hand_infos = {**hand_info, **meta_info}

        camutils.set_camera()
        camera_name = 'Camera'
        # Randomly pick background
        bg_path = random.choice(backgrounds)

        # Setup depth and segmentation rendering
        depth_path = os.path.join(folder_depth, frame_prefix)
        tmp_segm_path = render.set_cycle_nodes(
            scene, bg_path, segm_path=folder_temp_segm, depth_path=depth_path)
        tmp_files.append(tmp_segm_path)
        tmp_depth = depth_path + '{:04d}.exr'.format(1)
        tmp_files.append(tmp_depth)
        # Randomly pick clothing texture
        tex_path = random.choice(body_textures)

        # Spherical harmonic lighting
        sh_coeffs = texturing.get_sh_coeffs(
            ambiant_mean=ambiant_mean, ambiant_max_add=ambiant_add)
        texturing.set_sh_coeffs(scs, sh_coeffs)
        texturing.set_sh_coeffs(obj_scs, sh_coeffs)

        # Update body+hands image
        tex_img = bpy.data.images.load(tex_path)
        for part, material in materials.items():
            material.node_tree.nodes['Image Texture'].image = tex_img

        # Render
        img_path = os.path.join(folder_rgb, '{}.jpg'.format(frame_prefix))
        scene.render.filepath = img_path
        scene.render.image_settings.file_format = 'JPEG'
        bpy.ops.render.render(write_still=True)

        # Render obj only
        obj_img_path = os.path.join(folder_rgb_obj,
                                    '{}.jpg'.format(frame_prefix))
        smplh_obj.hide_render = True
        scene.render.filepath = obj_img_path
        obj_depth_path = os.path.join(folder_depth_obj, frame_prefix)
        tmp_segm_obj_path = render.set_cycle_nodes(
            scene,
            bg_path,
            segm_path=folder_temp_segm,
            depth_path=obj_depth_path)
        tmp_obj_depth = obj_depth_path + '{:04d}.exr'.format(1)
        tmp_files.append(tmp_obj_depth)
        tmp_files.append(tmp_segm_obj_path)
        bpy.ops.render.render(write_still=True)

        # Render hand only
        hand_img_path = os.path.join(folder_rgb_hand,
                                     '{}.jpg'.format(frame_prefix))
        smplh_obj.hide_render = False
        obj.hide_render = True
        scene.render.filepath = hand_img_path
        hand_depth_path = os.path.join(folder_depth_hand, frame_prefix)
        tmp_segm_hand_path = render.set_cycle_nodes(
            scene,
            bg_path,
            segm_path=folder_temp_segm,
            depth_path=hand_depth_path)

        tmp_hand_depth = hand_depth_path + '{:04d}.exr'.format(1)
        tmp_files.append(tmp_hand_depth)
        tmp_files.append(tmp_segm_hand_path)
        bpy.ops.render.render(write_still=True)

        # Delete objects
        delete_obj_model(obj)

        camutils.check_camera(camera_name=camera_name)
        segm_img = cv2.imread(tmp_segm_path)[:, :, 0]
        if render_body:
            keep_render = True
        else:
            keep_render = conditions.segm_condition(
                segm_img, side='right', use_grasps=True)
        depth, depth_min, depth_max = depthutils.convert_depth(tmp_depth)

        hand_infos['depth_min'] = depth_min
        hand_infos['depth_max'] = depth_max
        hand_infos['bg_path'] = bg_path
        hand_infos['sh_coeffs'] = sh_coeffs
        hand_infos['body_tex'] = tex_path
        # Concatenate depth as rgb
        hand_depth, hand_depth_min, hand_depth_max = depthutils.convert_depth(
            tmp_hand_depth)
        obj_depth, obj_depth_min, obj_depth_max = depthutils.convert_depth(
            tmp_obj_depth)
        depth = np.stack([depth, hand_depth, obj_depth], axis=2)
        hand_infos['hand_depth_min'] = hand_depth_min
        hand_infos['hand_depth_max'] = hand_depth_max
        hand_infos['obj_depth_min'] = obj_depth_min
        hand_infos['obj_depth_max'] = obj_depth_max

        # Concatenate segm as rgb
        obj_segm = cv2.imread(tmp_segm_obj_path)[:, :, 0]
        hand_segm = cv2.imread(tmp_segm_hand_path)[:, :, 0]
        keep_render_obj, obj_ratio = conditions.segm_obj_condition(
            segm_img, obj_segm, min_obj_ratio=min_obj_ratio)
        keep_render = keep_render and keep_render_obj
        hand_infos['obj_visibility_ratio'] = obj_ratio
        segm_img = np.stack([segm_img, hand_segm, obj_segm], axis=2)

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
            os.remove(obj_img_path)
            os.remove(hand_img_path)
        for filepath in tmp_files:
            os.remove(filepath)

        # Remove materials
        for material in materials_tmp:
            material.user_clear()
            bpy.data.materials.remove(material)
    print('DONE')
