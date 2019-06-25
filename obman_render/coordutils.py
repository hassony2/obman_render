import numpy as np
import numpy.matlib


def get_hand_body_info(posed_model,
                       render_body=False,
                       side='right',
                       right_smpl2mano=None,
                       left_smpl2mano=None,
                       cam_extr=None,
                       cam_calib=None):
    coords_3d = posed_model.J_transformed.r
    verts_3d = posed_model.r
    # Graspit is always a right hand
    side = 'right'
    if side == 'right':
        # Right hand coordinates
        root_3d = coords_3d[21]
        fingers_3d = coords_3d[37:52]
        tips_3d = get_tips(verts_3d, side='right')

        # Right hand vertices
        hand_verts_3d = verts_3d[right_smpl2mano]
    else:
        # Left hand coordinates
        root_3d = coords_3d[20]
        fingers_3d = coords_3d[22:37]
        tips_3d = get_tips(verts_3d, side='left')

        # Left hand vertices
        hand_verts_3d = verts_3d[left_smpl2mano]
    # Get reordered 3d positions
    h_3d = np.concatenate(
        [np.expand_dims(root_3d, 0), fingers_3d, tips_3d])

    # Matches to idx in order Root, Thumb, Index, Middle, Ring, Pinky
    # With numbering increasing for each finger from base to tips
    idxs = [
        0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8,
        9, 20
    ]
    h_3d = h_3d[np.array(idxs)]

    if render_body:
        # Get 3d and 2d coordinates for body
        full_body_3d = coords_3d[:22]
        full_body_2d = get_coords_2d(
            full_body_3d, cam_extr=cam_extr, cam_calib=cam_calib)

    # Get 2d coordinates for hand
    coords_2d = get_coords_2d(
        h_3d, cam_extr=cam_extr, cam_calib=cam_calib)
    # Save information
    hand_info = {
        'side': side,
        'coords_2d': coords_2d.transpose().astype(np.float32),
        'coords_3d': h_3d.astype(np.float32),
        'verts_3d': hand_verts_3d.astype(np.float32)
    }
    if render_body:
        hand_info['full_body_3d'] = full_body_3d.astype(np.float32)
        hand_info['full_body_2d'] = full_body_2d.transpose().astype(
            np.float32)
    return hand_info


def get_coords_2d(coords3d, cam_extr, cam_calib):
    coords3d_hom = np.concatenate(
        [coords3d, np.ones((coords3d.shape[0], 1))], 1)
    coords3d_hom = coords3d_hom.transpose()
    coords2d_hom = np.dot(cam_calib, np.dot(cam_extr, coords3d_hom))
    coords_2d = coords2d_hom / coords2d_hom[2, :]
    coords_2d = coords_2d[:2, :]
    return coords_2d


def get_tips(verts, side='right'):
    # In order Thumb, Index, Middle, Ring, Pinky
    right_tip_idxs = [6206, 5795, 5905, 6016, 6134]
    left_tip_idxs = [2745, 2334, 2444, 2555, 2672]
    if side == 'right':
        tip_idxs = right_tip_idxs
    elif side == 'left':
        tip_idxs = left_tip_idxs
    else:
        raise ValueError('Side sould be in [right, left], got {}'.format(side))

    tips = verts[np.array(tip_idxs)]
    return tips


def get_rigid_transform(A, B):
    cenA = np.mean(A, 0)  # 3
    cenB = np.mean(B, 0)  # 3
    N = A.shape[0]  # 24
    H = np.dot((B - np.matlib.repmat(cenB, N, 1)).transpose(),
               (A - np.matlib.repmat(cenA, N, 1)))

    [U, S, V] = np.linalg.svd(H)
    R = np.dot(U, V)  # matlab returns the transpose: .transpose()
    if np.linalg.det(R) < 0:
        U[:, 2] = -U[:, 2]
        R = np.dot(U, V.transpose())
    t = np.dot(-R, cenA.transpose()) + cenB.transpose()
    return R, t


def get_rigid_transform_posed_mano(posed_model, mano_model):
    rigid_transform = get_rigid_transform(mano_model.J_transformed[1:].r,
                                          posed_model.J_transformed[37:].r)

    # Concatenate rotation and translation
    rigid_transform = np.asarray(
        np.concatenate((rigid_transform[0], np.matrix(rigid_transform[1]).T),
                       axis=1))
    rigid_transform = np.concatenate((rigid_transform, np.array([[0, 0, 0,
                                                                  1]])))
    return rigid_transform
