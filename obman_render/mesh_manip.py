from copy import deepcopy
import random

import numpy as np


def alter_mesh(obj, verts):
    import bmesh
    import bpy
    from mathutils import Vector
    bpy.context.scene.objects.active = obj
    mesh = bpy.context.object.data

    bm = bmesh.new()

    # convert the current mesh to a bmesh (must be in edit mode)
    bpy.ops.object.mode_set(mode='EDIT')
    bm.from_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')  # return to object mode

    for v, bv in zip(verts, bm.verts):
        bv.co = Vector(v)

    # make the bmesh the object's mesh
    bm.to_mesh(mesh)
    bm.select_flush(True)
    bm.free()  # always do this when finished


def load_body_data(smpl_data, gender='female', idx=0, n_sh_bshapes=10):
    """
    Loads MoSHed pose data from CMU Mocap (only the given idx is loaded), and loads all CAESAR shape data.
    Args:
        smpl_data: Files with *trans, *shape, *pose parameters
        gender: female | male. CAESAR data has 2K male, 2K female shapes
        idx: index of the mocap sequence
        n_sh_bshapes: number of shape blendshapes (number of PCA components)
    """
    # create a dictionary with key the sequence name and values the pose and trans
    cmu_keys = []
    for seq in smpl_data.files:
        if seq.startswith('pose_'):
            cmu_keys.append(seq.replace('pose_', ''))

    name = sorted(cmu_keys)[idx % len(cmu_keys)]

    cmu_parms = {}
    for seq in smpl_data.files:
        if seq == ('pose_' + name):
            cmu_parms[seq.replace('pose_', '')] = {
                'poses': smpl_data[seq],
                'trans': smpl_data[seq.replace('pose_', 'trans_')]
            }

    # load all SMPL shapes
    fshapes = smpl_data['%sshapes' % gender][:, :n_sh_bshapes]

    return (cmu_parms, fshapes, name)


def load_smpl(template='assets/models/basicModel_{}_lbs_10_207_0_v1.0.2.fbx',
              gender='f'):
    """
    Loads smpl model, deleted armature and renames mesh to 'Body'
    """
    import bpy
    filepath = template.format(gender)
    bpy.ops.import_scene.fbx(
        filepath=filepath, axis_forward='Y', axis_up='Z', global_scale=100)
    obname = '{}_avg'.format(gender)
    ob = bpy.data.objects[obname]
    ob.parent = None

    # Delete armature
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['Armature'].select = True
    bpy.ops.object.delete(use_global=False)

    # Rename mesh
    bpy.data.meshes['Untitled'].name = 'Body'
    return ob


def random_global_rotation():
    """
    Creates global random rotation in axis-angle rotation format.
    """
    # 1. We will pick random axis: random azimuth and random elevation in spherical coordinates.
    # 2. We will pick random angle.
    # Random azimuth
    randazimuth = np.arccos(2 * np.random.rand(1) - 1)
    # Random elevation
    randelevation = 2 * np.pi * np.random.rand(1)
    # Random axis in cartesian coordinate (this already has norm 1)
    randaxis = [
        np.cos(randelevation) * np.cos(randazimuth),
        np.cos(randelevation) * np.sin(randazimuth),
        np.sin(randelevation)
    ]
    # Random angle
    randangle = 2 * np.pi * np.random.rand(1)
    # Construct axis-angle vector
    randaxisangle = randangle * randaxis

    return np.squeeze(randaxisangle)


def randomized_verts(model,
                     smpl_data,
                     ncomps=12,
                     pose_var=2,
                     hand_pose=None,
                     hand_pose_offset=3,
                     z_min=0.5,
                     z_max=0.8,
                     center_idx=40,
                     shape_val=2,
                     random_shape=False,
                     random_pose=False,
                     body_rot=True,
                     side='right',
                     split='train'):
    """
    Args:
        model: SMPL+H chumpy model
        smpl_data: 72-dim SMPL pose parameters from CMU and 10-dim shape parameteres from CAESAR
        center_idx: hand root joint on which to center, 25 for left hand
            40 for right
        z_min: min distance to camera in world coordinates
        z_max: max distance to camera in world coordinates
        ncomps: number of principal components used for both hands
        hand_pose: pca coeffs of hand pose
        hand_pose_offset: 3 is hand_pose contains global rotation
            0 if only pca coeffs are provided
    """

    if side == 'left':
        center_idx = 25
    else:
        center_idx = 40
    # Load smpl
    if split == 'test':
        cmu_idx = random.choice(list(range(4000, 4700)))
    elif split == 'val':
        cmu_idx = random.choice(list(range(4700, 5338)))
    else:
        cmu_idx = random.choice(list(range(0, 4000)))

    cmu_parms, fshapes, name = load_body_data(smpl_data, idx=cmu_idx)
    pose_data = cmu_parms[name]['poses']
    nframes = pose_data.shape[0]
    randframe = np.random.randint(nframes)

    # Init with zero trans
    model.trans[:] = np.zeros(model.trans.size)

    # Set random shape param
    if random_shape:
        randshape = random.choice(fshapes)
        model.betas[:] = randshape
    else:
        randshape = np.zeros(model.betas.shape)
    #model.betas[:] = np.random.uniform(
    #    low=-shape_val, high=shape_val, size=model.betas.shape)

    # Random body pose (except hand)
    randpose = np.zeros(model.pose.size)
    if random_pose:
        body_idx = 72
        randpose[:body_idx] = pose_data[randframe]

    # Overwrite global rotation with uniform random rotation
    if body_rot:
        randpose[0:3] = random_global_rotation()
    else:
        randpose[0:3] = [-np.pi/2, 0, 0]

    hand_comps = int(ncomps / 2)
    hand_idx = 66
    if hand_pose is not None:
        if side == 'left':
            randpose[hand_idx:hand_idx + hand_comps:] = hand_pose[
                hand_pose_offset:]
            left_rand = hand_pose[hand_pose_offset:]
        elif side == 'right':
            randpose[hand_idx + hand_comps:] = hand_pose[hand_pose_offset:]
            right_rand = hand_pose[hand_pose_offset:]
    else:
        # Alter right hand
        right_rand = np.random.uniform(
            low=-pose_var, high=pose_var, size=(hand_comps, ))
        randpose[hand_idx:hand_idx + hand_comps:] = right_rand

        # Alter left hand
        left_rand = np.random.uniform(
            low=-pose_var, high=pose_var, size=(hand_comps, ))
        randpose[hand_idx + hand_comps:] = left_rand

    model.pose[:] = randpose

    # Center on the hand
    rand_z = random.uniform(z_min, z_max)
    trans = np.array(
        [model.J_transformed[center_idx, :].r[i] for i in range(3)])
    # Offset in z direction
    trans[2] = trans[2] + rand_z
    model.trans[:] = -trans

    new_verts = model.r
    if side == 'right':
        hand_pose = right_rand
    else:
        hand_pose = left_rand
    meta_info = {
        'z': rand_z,
        'trans': (-trans).astype(np.float32),
        'pose': randpose.astype(np.float32),
        'shape': randshape.astype(np.float32),
        'mano_pose': hand_pose.astype(np.float32)
    }
    return new_verts, model, meta_info


def load_obj(filename_obj, normalization=True, texture_size=4):
    """
    Load Wavefront .obj file.
    This function only supports vertices (v x x x) and faces (f x x x).
    """

    # load vertices
    vertices = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype('float32')

    # load faces
    faces = []
    for line in open(filename_obj).readlines():
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype('int32') - 1

    # normalize into a unit cube centered zero
    if normalization:
        vertices -= vertices.min(0)[None, :]
        vertices /= np.abs(vertices).max()
        vertices *= 2
        vertices -= vertices.max(0)[None, :] / 2

    return vertices, faces
