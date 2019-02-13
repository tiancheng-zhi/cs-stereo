from pathlib import Path
import numpy as np
import torch.utils.data as data
import cv2


class StereoDataset(data.Dataset):

    def __init__(self, data_path, list_path, splits):
        super(StereoDataset, self).__init__()
        self.data_path = data_path
        self.height = 429
        self.width = 582
        self.im_suff = 'Resize'
        self.mat_suff = 'Material'
        self.records = []
        for split in splits:
            f = open(Path(list_path) / (split + '.txt'), 'r')
            lines = f.readlines()
            f.close()
            for i, line in enumerate(lines):
                splited = line.split()
                collection = splited[0]
                key = splited[1]
                rgb_exp = float(splited[2])
                nir_exp = float(splited[3])
                record = (collection, key, rgb_exp, nir_exp)
                self.records.append(record)
        self.records = sorted(self.records)
        self.n_records = len(self.records)

    def __getitem__(self, index):
        collection, key, rgb_exp, nir_exp = self.records[index]
        rgb = cv2.imread(self.fname(collection, key, self.im_suff, 'RGB', 'png'))
        assert(rgb.shape == (self.height, self.width, 3))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.transpose([2, 0, 1])
        rgb = rgb.astype(np.float32)
        nir = cv2.imread(self.fname(collection, key, self.im_suff, 'NIR', 'png'), cv2.IMREAD_GRAYSCALE)
        assert(nir.shape == (self.height, self.width))
        nir = nir[np.newaxis][:]
        nir = nir.astype(np.float32)
        rgb_exp = np.array([rgb_exp]).astype(np.float32)
        nir_exp = np.array([nir_exp]).astype(np.float32)
        material = np.load(self.fname(collection, key, self.mat_suff, '', 'npz'))
        rgb_mat = material['rgb']
        nir_mat = material['nir']
        return collection, key, rgb, nir, rgb_exp, nir_exp, rgb_mat, nir_mat

    def __len__(self):
        return self.n_records

    def fname(self, collection, key, suffix, camera, ftype):
        return str(Path(self.data_path) / collection / (camera + suffix) / (key + '_' + camera + suffix + '.' + ftype))
