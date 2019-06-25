import os

from obman_render import getextures


def get_all_lsun(folder='/sequoia/data2/gvarol/datasets/LSUN/data/img'):
    train_list_path = os.path.join(folder, 'train_img.txt')
    with open(train_list_path) as f:
        lines = f.readlines()
    bg_names = [os.path.join(folder, line.strip()) for line in lines]
    return bg_names


def get_image_paths(
        background_datasets,
        split='train',
        shapenet_folder='/sequoia/data2/dataset/shapenet/ShapeNetCore.v2',
        shapenet_split_folder='assets/textures/shapenet',
        lsun_path='/sequoia/data2/gvarol/datasets/LSUN/data/img',
        imagenet_path='/sequoia/data3/datasets/imagenet'):
    valid_datasetnames = ['imagenet', 'lsun', 'white', 'black', 'shapenet']
    erroneous_datasetnames = set(background_datasets) - set(valid_datasetnames)
    if len(erroneous_datasetnames):
        raise ValueError('{} not in {}'.format(erroneous_datasetnames,
                                               valid_datasetnames))
    backgrounds = []
    if 'imagenet' in background_datasets:
        if split == 'test':
            backgrounds.extend(getextures.get_all_textures_imagenet(imagenet_path)[0::3])
        elif split == 'val':
            backgrounds.extend(getextures.get_all_textures_imagenet(imagenet_path)[1::3])
        else:
            backgrounds.extend(getextures.get_all_textures_imagenet(imagenet_path)[2::3])
    # Get LSUN backgrounds
    if 'lsun' in background_datasets:
        if split == 'test':
            backgrounds.extend(get_all_lsun(folder=lsun_path)[0::3])
        elif split == 'val':
            backgrounds.extend(get_all_lsun(folder=lsun_path)[1::3])
        else:
            backgrounds.extend(get_all_lsun(folder=lsun_path)[2::3])
    if 'white' in background_datasets:
        backgrounds.append('assets/backgrounds/white.jpg')
    if 'black' in background_datasets:
        backgrounds.append('assets/backgrounds/black.png')
    if 'shapenet' in background_datasets:
        backgrounds.extend(
            getextures.get_all_textures_shapenet(
                split_folder=shapenet_split_folder,
                shapenet_folder=shapenet_folder,
                split=split))
    return backgrounds


def get_bodytexture_paths(
        texture_types,
        split='train',
        shapenet_split_folder='assets/textures/shapenet',
        shapenet_folder='/sequoia/data2/dataset/shapenet/ShapeNetCore.v2',
        lsun_path='/sequoia/data2/gvarol/datasets/LSUN/data/img',
        imagenet_path='/sequoia/data3/datasets/imagenet'):
    textures = []
    if 'jpgs' in texture_types:
        textures.extend(getextures.get_all_textures_jpg(split=split))
    if 'bodywithands' in texture_types:
        textures.extend(getextures.get_all_textures_bodywithands(split=split))
    if '4096' in texture_types:
        textures.append('misc/mpi_body_4096/sample4096.png')
    if 'lsun' in texture_types:
        if split == 'test':
            textures.extend(get_all_lsun(folder=lsun_path)[:50000])
        elif split == 'val':
            textures.extend(get_all_lsun(folder=lsun_path)[50000:100000])
        else:
            textures.extend(get_all_lsun(folder=lsun_path)[100000:200000])
    if 'shapenet' in texture_types:
        textures.extend(
            getextures.get_all_textures_shapenet(
                split_folder=shapenet_split_folder,
                shapenet_folder=shapenet_folder,
                split=split))
    if 'imagenet' in texture_types:
        if split == 'test':
            textures.extend(getextures.get_all_textures_imagenet(imagenet_path)[:50000])
        elif split == 'val':
            textures.extend(
                getextures.get_all_textures_imagenet(imagenet_path)[50000:100000])
        else:
            textures.extend(
                getextures.get_all_textures_imagenet(imagenet_path)[100000:200000])
    if 'white' in texture_types:
        textures.append('assets/backgrounds/white.jpg')
    return textures


def get_hrhand_paths(texture_type, split='train'):
    if texture_type == ['jpgs']:
        all_hands = getextures.get_high_res_hands(
            res=512, texture_types=['in_house'], split=split)
    elif texture_type == ['pngs']:
        all_hands = getextures.get_high_res_hands(
            res=1024, texture_types=['in_house'], split=split)
    elif texture_type == ['4096']:
        all_hands = getextures.get_high_res_hands(
            res=4096, texture_types=['in_house'], split=split)
    else:
        raise NotImplementedError(
            'high_res_hands not available yet for texture_type {}'.format(
                texture_type))
    return all_hands
