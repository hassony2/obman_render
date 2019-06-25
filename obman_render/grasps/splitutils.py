import csv
from collections import defaultdict
import random
import os
from tqdm import tqdm


def make_modelnet_split(root='/sequoia/data2/dataset/ModelNet/graspmodels',
                        test_val_ratio=0.15,
                        save_path='misc/2018_10_01_grasp_split_modelnet.csv'):
    model_root = os.path.join(root, 'models')
    classes_root = os.path.join(root, 'selected')

    all_models = {}
    for category in tqdm(os.listdir(model_root)):
        print(category)
        model_list_path = os.path.join(classes_root, '{}.txt'.format(category))
        with open(model_list_path) as m_f:
            lines = [line.strip() for line in m_f.readlines()]
        all_models[category] = lines

    with open(save_path, 'w', newline='') as csv_write_f:
        csv_writer = csv.writer(
            csv_write_f,
            delimiter=' ',
            quotechar='|',
            quoting=csv.QUOTE_MINIMAL)
        for cat, model_list in tqdm(sorted(all_models.items()), desc='cat'):
            model_infos = []
            for model in tqdm(sorted(model_list), desc='model'):
                print(model)
                # model_path = os.path.join(model_root, cat, model,
                #                           '{}.off'.format(model))
                model_infos.append([cat, model])
            random.shuffle(model_infos)
            test_nb = int(len(model_infos) * test_val_ratio)
            test_samples = model_infos[:test_nb]
            val_samples = model_infos[test_nb:2 * test_nb]
            train_samples = model_infos[2 * test_nb:]
            for cat, model in test_samples:
                csv_writer.writerow([cat, model, 'test'])
            for cat, model in val_samples:
                csv_writer.writerow([cat, model, 'val'])
            for cat, model in train_samples:
                csv_writer.writerow([cat, model, 'train'])


def make_split(filepath='/sequoia/data2/dataset/shapenet/shapenet_select.csv',
               test_val_ratio=0.15,
               save_path='misc/grasp_split_all_graspable.csv'):
    if os.path.exists(save_path):
        raise ValueError('Grasp split already exists at {}'.format(save_path))
    with open(save_path, 'w', newline='') as csv_write_f:
        csv_writer = csv.writer(
            csv_write_f,
            delimiter=' ',
            quotechar='|',
            quoting=csv.QUOTE_MINIMAL)
        with open(filepath, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row_idx, row in enumerate(csvreader):
                if row_idx > 0:
                    class_path = row[1]
                    class_id = os.path.basename(class_path)
                    samples = sorted(os.listdir(class_path))
                    random.shuffle(samples)
                    test_nb = int(len(samples) * test_val_ratio)
                    test_samples = samples[:test_nb]
                    val_samples = samples[test_nb:2 * test_nb]
                    train_samples = samples[2 * test_nb:]
                    for test_sample in test_samples:
                        csv_writer.writerow([class_id, test_sample, 'test'])
                    for val_sample in val_samples:
                        csv_writer.writerow([class_id, val_sample, 'val'])
                    for train_sample in train_samples:
                        csv_writer.writerow([class_id, train_sample, 'train'])


def read_split(split_path='misc/grasp_split.csv'):
    samples = defaultdict(list)
    with open(split_path, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=' ', quotechar='"')
        for row_idx, row in enumerate(csvreader):
            if row_idx > 0:
                samples[row[2]].append((row[0], row[1]))
    return samples


if __name__ == '__main__':
    random.seed(0)
    make_modelnet_split()
    samples = read_split('misc/2018_10_01_grasp_split_modelnet.csv')
    import pdb
    pdb.set_trace()
