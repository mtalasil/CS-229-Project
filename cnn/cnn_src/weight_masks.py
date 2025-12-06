from train import load_sharded
from generate_data_val_fn import save_sharded_w
import numpy as np
from scipy import ndimage
import copy
import matplotlib.pyplot as plt
from src.occ_grid import Occ_grid


dil_filter = np.array([[True, True, True],
                      [True, True, True], 
                      [True, True, True]], dtype=bool)

def gen_weight_maps(f_names, save_name):
    X, y, masks = load_sharded(f_names)
    dummy_occ = Occ_grid(y[0].shape[0], y[0].shape[1])
    weights_arr = []
    for i in range(masks.shape[0]):
        dil_mask = ndimage.binary_dilation(~masks[i], dil_filter, 3).astype(masks[i].dtype)
        diff = dil_mask & masks[i]
        weight_map = np.ones_like(masks[i], dtype=np.float32)
        weight_map[diff] = 2.0
        goal = np.argmin(y[i])
        goal_p = dummy_occ.idx2point(goal)
        goal_map = np.zeros_like(y[i], dtype=bool)
        goal_map[goal_p[0], goal_p[1]] = True
        goal_map = ndimage.binary_dilation(goal_map, dil_filter, 3).astype(goal_map.dtype)
        weight_map[goal_map] = 4.0
        weights_arr.append(weight_map)
    weights = np.stack(weights_arr, axis=0)

    save_sharded_w(f"{save_name}_{X.shape[0]}_exs", X, y, masks, weights)

def main():
    train_fnames = "train_data/train_dataset_*_exs_shard_*.npz"
    val_fnames = "val_data/val_dataset_*_exs_shard_*.npz"
    save_name_train = "train_data_w/train_dataset"
    save_name_val = "val_data_w/val_dataset"
    gen_weight_maps(val_fnames, save_name_val)
    gen_weight_maps(train_fnames, save_name_train)






if __name__ == "__main__":
    main()