def segm_condition(segm_img, pixel_nb=100, use_grasps=False, side='right'):
    # When grasps, object should be visible
    if use_grasps:
        if (segm_img == 100).sum() > pixel_nb:
            object_ok = True
        else:
            object_ok = False
    else:
        object_ok = True
    if side == 'right':
        if (segm_img == 22).sum() > pixel_nb:
            return True and object_ok
        if (segm_img == 24).sum() > pixel_nb:
            return True and object_ok
        else:
            return False
    elif side == 'left':
        if (segm_img == 21).sum() > pixel_nb:
            return True and object_ok
        if (segm_img == 23).sum() > pixel_nb:
            return True and object_ok
        else:
            return False
    else:
        raise ValueError('Side should be [left|right], got {}'.format(side))


def segm_obj_condition(segm_img, obj_segm, obj_idx=100, min_obj_ratio=0.7):
    obj_vis_ratio = (segm_img == obj_idx).sum() / (obj_segm == obj_idx).sum()
    if obj_vis_ratio > min_obj_ratio:
        return True, obj_vis_ratio
    else:
        return False, obj_vis_ratio
