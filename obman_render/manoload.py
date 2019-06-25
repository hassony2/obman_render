import zipfile
import os
import pickle


def get_dim_poses(ncomps=45, split='train'):
    zip_path = 'misc/MANO_training_scans_alignments/REFITS___MANO_POSES___parms_for_______shapeYES_priorNOO___ini/handsOnly_REGISTRATIONS_r_lm___parms_for___ncomps_{}.zip'.format(
        ncomps)
    zipinfo = zipfile.ZipFile(zip_path)
    pkl_files = sorted([
        name for name in zipinfo.namelist()
        if '.pkl' in name and 'mirrored' not in name
    ])
    train_subjects = [
        '01', '04', '06', '09', '10', '13', '15', '24', '25', '26', '27', '32',
        '33', '34', '35', '36', '37', '38', '39', '49', '50'
    ]
    val_subjects = ['17', '28', '29', '40', '42']
    test_subjects = ['18', '30', '31', '41', '43']
    if split == 'train':
        subjects = train_subjects
    elif split == 'val':
        subjects = val_subjects
    elif split == 'test':
        subjects = test_subjects
    pkl_files_subjects = [
        name for name in pkl_files if name[53:55] in subjects
    ]
    hand_poses = [
        pickle.load(zipinfo.open(pkl_file), encoding='latin')['pose'][3:]
        for pkl_file in pkl_files_subjects
    ]
    return hand_poses
