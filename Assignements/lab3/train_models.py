# %%
import os
import gc

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from sklearn.preprocessing import StandardScaler

# %%
DATASET_DIR = os.path.relpath("data/")
os.makedirs(DATASET_DIR, exist_ok=True)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
# set the random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# train the network
num_epochs = 30


def load_data(feature_type="lmfcc"):
    """
    Load the data from the dataset directory.
    """
    print("Loading data...")
    train_x = torch.load(os.path.join(
        DATASET_DIR, "{}_train_x.pt".format(feature_type)))
    val_x = torch.load(os.path.join(
        DATASET_DIR, "{}_val_x.pt".format(feature_type)))
    test_x = torch.load(os.path.join(
        DATASET_DIR, "{}_test_x.pt".format(feature_type)))

    train_y = torch.load(os.path.join(DATASET_DIR, "train_y.pt"))
    val_y = torch.load(os.path.join(DATASET_DIR, "val_y.pt"))
    test_y = torch.load(os.path.join(DATASET_DIR, "test_y.pt"))

    return train_x, val_x, test_x, train_y, val_y, test_y


def train_model(model, name, train_loader, val_loader):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # setup logging so that you can follow training using TensorBoard (see https://pytorch.org/docs/stable/tensorboard.html)
    writer = SummaryWriter()

    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    # early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    

    bar = tqdm(range(num_epochs))
    for epoch in range(num_epochs):
        train_acc = 0
        train_loss = 0.0
        
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                labels = torch.argmax(labels, dim=1)
                train_acc += (predicted == labels).sum().item() / len(labels)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                labels = torch.argmax(labels, dim=1)
                val_acc += (predicted == labels).sum().item() / len(labels)

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc /= len(train_loader)
            val_acc /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            patience_counter = 0
            best_val_loss = val_loss
            # save the model with the best validation loss
            torch.save(model.state_dict(),
                       'models/{}_best_model.pt'.format(name))
        else:
            patience_counter += 1
            if patience_counter > 10:
                break

        # print(f'Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}')
        writer.add_scalars(
            'loss', {'train': train_loss, 'val': val_loss}, epoch)
        bar.set_description(
            f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}')
        bar.update(1)

    np.savez(
        os.path.join("models", "{}_run.npz".format(name)),
        train_losses=train_losses, val_losses=val_losses,
        train_accs=train_accs, val_accs=val_accs
    )
    writer.flush()


# %%
class MLPRelu(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden):
        super(MLPRelu, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Linear(input_dim, 256)
        self.layers = nn.ModuleList([nn.Linear(256, 256)
                                    for _ in range(n_hidden)])
        self.output_layer = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        return x


class MLPSigmoid(nn.Module):
    def __init__(self, input_dim, output_dim, n_hidden):
        super(MLPSigmoid, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Linear(input_dim, 256)
        self.layers = nn.ModuleList([nn.Linear(256, 256)
                                    for _ in range(n_hidden)])
        self.output_layer = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = torch.sigmoid(x)
        for layer in self.layers:
            x = layer(x)
            x = torch.sigmoid(x)
        x = self.output_layer(x)
        return x

# %%

def train_MLPs(Model, feature, n_hidden):
    X_train, X_val, _, y_train, y_val, _ = load_data(feature)
    X_train = X_train.to(device)
    X_val = X_val.to(device)
    y_train = y_train.to(device)
    y_val = y_val.to(device)

    # create the data loaders for training and validation sets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        # , num_workers=4, persistent_workers=True)
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(
        # , num_workers=4, persistent_workers=True)
        val_dataset, batch_size=batch_size, shuffle=False)

    for n in n_hidden:
        model = Model(X_train.shape[1], y_train.shape[1], n)
        train_model(model, "{}_{}_{}h".format(
            Model.__name__,
            feature, n), train_loader, val_loader)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    # for feature in ["single_lmfcc", "single_mspec", "lmfcc", "mspec"]:
    #     print("Training MLPs with {} features...".format(feature))
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     train_MLPs(feature, [1, 4])
    print("Training MLP (sigmoid) with lmfcc features...")
    train_MLPs(MLPSigmoid, "lmfcc", [4])
