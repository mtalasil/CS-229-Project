import numpy as np
from src import valueCNN

import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import glob


# things to try:
# he init

def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    total_weights = 0.0
    for xb, yb, mb, wb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        wb = wb.to(device)

        opt.zero_grad()

        pred = model(xb)             
        if pred.dim() == 4 and pred.size(1) == 1: # squeeze model output shape from (B, 1, H, W) to (B, H, W)
            pred = pred.squeeze(1)

        diff = pred - yb
        sq_err = diff**2
        sq_err_w = sq_err * wb

        batch_se = sq_err_w[mb].sum()
        batch_n = wb[mb].sum()

        if batch_n == 0:
            continue
        loss = batch_se / batch_n

        loss.backward()
        opt.step()

        total_loss += batch_se.item()
        total_weights += batch_n.item()

    avg_loss = total_loss / total_weights
    return avg_loss

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_weights = 0.0

    for xb, yb, mb, wb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        mb = mb.to(device)
        wb = wb.to(device)

        pred = model(xb)             
        if pred.dim() == 4 and pred.size(1) == 1: # squeeze model output shape from (B, 1, H, W) to (B, H, W)
            pred = pred.squeeze(1)

        diff = pred - yb
        sq_err = diff**2
        sq_err_w = sq_err * wb

        batch_se = sq_err_w[mb].sum()
        batch_n = wb[mb].sum()

        if batch_n == 0:
             continue

        total_loss += batch_se.item()
        total_weights += batch_n.item()

    avg_loss = total_loss / total_weights
    return avg_loss

def train(model, train_loader, val_loader, opt, device, base_ch, num_epochs=10, plot_curves=True):
    epochs = np.arange(num_epochs)
    train_losses = np.zeros(num_epochs)
    val_losses = np.zeros(num_epochs)

    best_val_loss = np.inf
    best_state = model.state_dict().copy()
    best_epoch = 0
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, opt, device)
        val_loss = eval_model(model, val_loader, device)
        train_losses[epoch] = train_loss
        val_losses[epoch] = val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            best_epoch = epoch

        print(f"Epoch {epoch:02d}: "
          f"train_loss={train_loss:.4f}, "
          f"val_loss={val_loss:.4f}")
    torch.save(best_state, f"unet_base{base_ch}_cnn_model_epoch_{best_epoch}.pt")

    if plot_curves:
        plt.figure()
        plt.scatter(epochs, train_losses, color='blue', label='Training Loss')
        plt.scatter(epochs, val_losses, color='red', label='Val Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("losses.png")
        plt.close()

def label_counts(labels, num_labels):
    counts = []
    for i in range(num_labels):
        counts.append(np.sum(labels == i))
    return counts

def load_sharded(f_name):
    files = sorted(glob.glob(f_name))
    inputs_list = []
    labels_list = []
    masks_list = []
    for f in files:
        print(f"loading {f}")
        data = np.load(f)
        inputs_list.append(data['inputs'])
        labels_list.append(data['labels'])
        masks_list.append(data['masks'])
    inputs = np.concatenate(inputs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    return inputs, labels, masks

def load_sharded_w(f_name):
    files = sorted(glob.glob(f_name))
    inputs_list = []
    labels_list = []
    masks_list = []
    weights_list = []
    for f in files:
        print(f"loading {f}")
        data = np.load(f)
        inputs_list.append(data['inputs'])
        labels_list.append(data['labels'])
        masks_list.append(data['masks'])
        weights_list.append(data['weights'])

    inputs = np.concatenate(inputs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    weights = np.concatenate(weights_list, axis=0)
    return inputs, labels, masks, weights

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X_train, y_train, masks_train, weights_train = load_sharded_w("train_data_w/train_dataset_*_exs_shard_*.npz")
    X_val, y_val, masks_val, weights_val = load_sharded_w("val_data_w/val_dataset_*_exs_shard_*.npz")

    print("Train inputs:", X_train.shape)
    print("Train labels:", y_train.shape)
    print("Val inputs:", X_val.shape)
    print("Val labels:", y_val.shape)

    print(X_train.shape[0])

    X_train_t = torch.from_numpy(X_train).float()
    y_train_t = torch.from_numpy(y_train).float()
    masks_train_t = torch.from_numpy(masks_train).bool()
    weights_train_t = torch.from_numpy(weights_train).float()

    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float()
    masks_val_t = torch.from_numpy(masks_val).bool()
    weights_val_t = torch.from_numpy(weights_val).float()

    # X_train_t = X_train_t[:256]
    # y_train_t = y_train_t[:256]
    # masks_train_t = masks_train_t[:256]
    # weights_train_t = weights_train_t[:256]

    # X_val_t = X_val_t[:256]
    # y_val_t = y_val_t[:256]
    # masks_val_t = masks_val_t[:256]
    # weights_val_t = weights_val_t[:256]

    batch_size = 64

    train_ds = TensorDataset(X_train_t, y_train_t, masks_train_t, weights_train_t)
    val_ds  = TensorDataset(X_val_t,  y_val_t, masks_val_t, weights_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False)

    in_channels = X_train_t.shape[1]
    print(in_channels)

    base_channels = 32

    model = valueCNN.ValueCNN(in_channels, base_channels).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train(model, train_loader, val_loader, optimizer,device, base_channels, 10)

if __name__ == "__main__":
    main()

'''
Ok so here is training framework:
start given n (s_t_grid, path_seq) training pairs
s_t_grid is of shape K x h x w, where K is number of timesteps 
action space is size 9

Simulation structure
1. Run sim
-must model object transitions
-as you go, append current map frames to list
2. perform A*
-actions
    -ok so with no restrictions, we could move to any of the 26 cubes around us, but there are restrictions
    -action restrictions are 
        -have to move forward in time - so we can only move to the 9 cubes in the next timestep
            -technically A* should still be optimal with this restriction since it can just be modelled as a directed edge (i.e. there are directed edges from t to t-1 nodes that are 1 (or sqrt(2)) spatial units away, but not vice versa) (can only move to nodes forward in time, not back)
        -cant move into an occupied cell (everything is "static" since we are traversing through time as well)
    -euclidean heuristic to goal in 2D space (i.e. euclidean dist is dist to goal of curr timestep, since recall goal is a line in space time)

Generate training data:
0. make list of training examples
1. run n simulations (simulations may have different time dimensions)
2. for a given simulation with T timesteps, generate (k prev frames + curr frame + goal map, action_id) pairs and add to list of training examples (these will be inputted into model)
3. shuffle training example list
4. now can train

Model: CNN 

model inputs:
    past k frames - k x h x w - MAYBE K SHOULD BE 1 SINCE DYNAMICS ARE MARKOVIAN (ONLY DEPEND ON PREV ACTION)
        - To ensure constant input size, if we dont have k prev frames, (say only r frames), then append frame 0 k-r times to start of tensor (equivalent of saying world is just static before first frame)
    curr frame - h x w
    FOR EACH FRAME ALSO INCLUDE MASK OF WHERE AGENT IS
    goal encoded as vector field point to goal (3 layer tensor with one layer being magnitude/potential (from gaussian or 1/(1+euclid_dist)), and other 2 being x and y component of unit vec pointing towards goal) - 3 x h x w
    total input: these stacked in a (k+1+3) x h x w tensor 

model outputs:
    -logits over actions (9)
        -before passing logits into cross entropy, find any invalid actions (based on current state in current frame) and set their logits to -inf
        -invalid actions include any states around me that are currently occupied, or that are one action away from an object (i.e. an object could move into that state with a single action - THIS INCLUDES DOING NOTHING ACTION - bc if an object is beside me, I must move)
        OR any actions that go off the map ofc

model loss:
    cross entropy between predicted action vector and true best action (scalar input indicating index of correct class label) we get from path_seq

Further modifications:
-only consider local window around you rather than entire grid to make more efficient as input to model
    -so rather than (k+1+3 x h x w) I have ()
-maybe ConvLSTM later to better account for temporal component
-add future predictions for some horizon h what occ map may look like as model input


'''

