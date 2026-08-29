from dataset.dataset import TableGraphDataset
from src.utils.smiles2graph import create_graph_data_from_smiles
from torch_geometric.data import Batch
import argparse
import time
from models.weaklearner import MLP_2HL, MLP_Maccs, MLP_Transformer  # MLP_ResNet MLP_Transformer
from models.ensemblemodel import DynamicNet, DynamicNetForMLPImage
from torch.optim import SGD, Adam
from dataset.dataset import TabularImageDataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import random

"""
Training pipeline for Tabular Descriptors + Molecular Image Data， Transformer 
"""
parser = argparse.ArgumentParser()

# Integer parameters with no default value and required flag
parser.add_argument('--feat_d', type=int, help='Feature dimension', default=19 + 167)  # Input feature dimension
parser.add_argument('--hidden_d', type=int, help='Hidden layer dimension', default=256)  # Hidden layer dimension of weak learner

parser.add_argument('--table_dim_in', type=int, default=19)  # Tabular feature input dimension
parser.add_argument('--table_dim_hidden', type=int, default=128)  # Note: processed parameter
parser.add_argument('--out_dim', type=int, default=128)  # Note: processed parameter
parser.add_argument('--combined_dim', type=int, default=128)  # Note: processed parameter
parser.add_argument('--dim_hidden1', type=int, default=128)  # Note: processed parameter
parser.add_argument('--dim_hidden2', type=int, default=128)  # Note: processed parameter

# Transformer
parser.add_argument('--transformer_embed_dim', type=int, default=512)  # Transformer
parser.add_argument('--transformer_heads', type=int, default=4)  # Transformer
parser.add_argument('--transformer_layers', type=int, default=4)  # Transformer

# Float parameters with no default value and required flag
parser.add_argument('--boost_rate', type=float, help='Boosting rate', default=1.0)
parser.add_argument('--lr', type=float, help='Learning rate', default=0.001)
parser.add_argument('--L2', type=float, help='L2 regularization coefficient', default=1.0e-2)  # 1.0e-2

# Integer parameters with default values
parser.add_argument('--num_nets', type=int, help='Number of networks', default=5)
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)   # 32
parser.add_argument('--epochs_per_stage', type=int, help='Epochs per stage', default=100)
parser.add_argument('--correct_epoch', type=int, help='Epoch to correct model', default=100)

# String parameters with no default value and required flag
parser.add_argument('--data', type=str, help='Path to data')
parser.add_argument('--tr', type=str, help='Path to training data')
parser.add_argument('--te', type=str, help='Path to testing data')
parser.add_argument('--out_f', type=str, help='Output file path',
                    default='../checkpoint/best_GrowTableImageTransformer.pth')

# Float parameter with default value


# Boolean flags
parser.add_argument('--sparse', action='store_true', help='Use sparse representation')
parser.add_argument('--normalization', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='Enable normalization (true/false)')
parser.add_argument('--cv', type=lambda x: (str(x).lower() == 'true'), default=True,
                    help='Enable cross-validation (true/false)')
parser.add_argument('--cuda', action='store_true', help='Use CUDA for GPU acceleration', default=True)

args = parser.parse_args()

if not args.cuda:
    torch.set_num_threads(4)


def get_optim(params, lr, weight_decay):
    optimizer = Adam(params, lr, weight_decay=weight_decay)
    # optimizer = SGD(params, lr, weight_decay=weight_decay)
    return optimizer


def root_mse(net_ensemble, loader):
    loss = 0
    total = 0

    for x, image_data, y in loader:
        if args.cuda:
            x = x.to(device)
            image_data = image_data.to(device) # 2025.3.7
        with torch.no_grad():
            _, out = net_ensemble.forward(x, image_data)
        y = y.cpu().numpy().reshape(len(y), 1)
        out = out.cpu().numpy().reshape(len(y), 1)
        loss += mean_squared_error(y, out) * len(y)
        total += len(y)
    return np.sqrt(loss / total)


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.
    Note: It assumes that y_true does not contain zeros to avoid division by zero.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0  # Avoid division by zero
    y_true = y_true[non_zero_indices]
    y_pred = y_pred[non_zero_indices]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# RESULTS
def get_predictions(net_ensemble, loader):
    net_ensemble.to_eval()  # Note: processed parameter
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, image_data, y in loader:
            # Note: processed parameter
            if args.cuda:
                x = x.to(device)
                image_data = image_data.to(device)
                y = y.to(device)

            # Note: processed parameter
            _, preds = net_ensemble.forward(x, image_data)

            # RESULTS
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    # RESULTS
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

def worker_init_fn(worker_id):
    np.random.seed(41 + worker_id)
    random.seed(41 + worker_id)


def set_seed(seed):
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)  # Note: processed parameter
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # CUDA RESULTS
    torch.backends.cudnn.benchmark = False  # Note: processed parameter
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


if __name__ == "__main__":
    set_seed(41)  # Set global random seed
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f" ：{device}")
    # Note: processed parameter
    file_path = "../data/processed/MemTrOC-Dataset.csv"
    data = pd.read_csv(file_path)
    # data = data.head(10)
    # Note: processed parameter
    X = data.iloc[:, 4:23].values  # 19
    y = data.iloc[:, 23].values  # Note: processed parameter

    smiles_list = data.iloc[:, 3].values  # 3 SMILES

    # Note: processed parameter
    X_train, X_test, y_train, y_test, smiles_train, smiles_test = train_test_split(X, y, smiles_list, test_size=0.1,
                                                                                   random_state=41)
    X_train, X_val, y_train, y_val, smiles_train, smiles_val = train_test_split(X_train, y_train, smiles_train,
                                                                                test_size=0.2 / 0.9, random_state=41)
    # Train Set
    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)  # Train Set scaler
    X_test = scaler_X.transform(X_test)  # Train Set scaler
    # Convert to PyTorch Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create dataset
    # TabularImageDataset smiles image
    train_dataset = TabularImageDataset(X_train_t, smiles_train, y_train_t)
    val_dataset = TabularImageDataset(X_val_t, smiles_val, y_val_t)
    test_dataset = TabularImageDataset(X_test_t, smiles_test, y_test_t)

    # Create DataLoader
    batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    N = len(X_train)
    print(type(args.lr))
    print(type(args.boost_rate))

    best_rmse = pow(10, 6)
    val_rmse = best_rmse
    best_stage = args.num_nets - 1
    c0 = y_train.mean()  # init_gbnn(train) # c0 Train Set Value Value
    net_ensemble = DynamicNetForMLPImage(c0, args.boost_rate)  # Note: processed parameter
    loss_f1 = nn.MSELoss()
    loss_models = torch.zeros((args.num_nets, 3))
    for stage in range(args.num_nets):  # Iterate across weak learner stages
        t0 = time.time()
        # Note: processed parameter
        model = MLP_Transformer.get_model(stage, args)  # MLP_ResNet MLP_Transformer
        if args.cuda:
            # model
            model = model.to(device)

        optimizer = get_optim(model.parameters(), args.lr, args.L2)  # Initialize optimizer
        net_ensemble.to_train()  # Set the models in ensemble net to train mode
        stage_mdlloss = []  # Note: processed parameter
        for epoch in range(args.epochs_per_stage):  # Note: processed parameter
            for i, (x, image_data, y) in enumerate(train_loader):

                if args.cuda:
                    # x = x
                    # y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
                    x = x.to(device)
                    image_data = image_data.to(device)
                    y = y.to(device).view(-1, 1)
                else:
                    y = y.view(-1, 1)

                middle_feat, out = net_ensemble.forward(x, image_data)  # Value
                out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)
                out = out.to(device)
                # print("out shape:", out.shape)
                # print("out:", out)
                # print("y: ", y)
                # out = out.view(-1, 1)
                # print("after out.view(-1, 1) out shape:", out.shape)
                # print("y shape:", y.shape)
                grad_direction = -(out - y)  # Value

                _, out = model(x, image_data, middle_feat)  # x
                # out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)
                out = out.view(-1, 1)
                loss = loss_f1(net_ensemble.boost_rate * out, grad_direction)  # T

                model.zero_grad()
                loss.backward()
                optimizer.step()  # Update weak learner parameters
                stage_mdlloss.append(loss.item() * len(y))

        net_ensemble.add(model)  # Add trained weak learner to ensemble
        sml = np.sqrt(np.sum(stage_mdlloss) / N)  # Average sample loss

        # Joint corrective step for ensemble refinement Reduced learning rate
        lr_scaler = 3
        # fully-corrective step
        stage_loss = []
        if stage > 0:
            # Adjusting corrective step learning rate
            if stage % 15 == 0:
                # lr_scaler *= 2
                args.lr /= 2
                args.L2 /= 2
            optimizer = get_optim(net_ensemble.parameters(), args.lr / lr_scaler, args.L2)
            for _ in range(args.correct_epoch):
                stage_loss = []
                for i, (x, image_data, y) in enumerate(train_loader):
                    x = x.to(device)
                    image_data = image_data.to(device)
                    y = y.to(device).view(-1, 1)
                    if args.cuda:
                        # x, y = x, y.view(-1, 1)
                        x, y = x.to(device), y.to(device)
                    _, out = net_ensemble.forward_grad(x, image_data)
                    out = torch.as_tensor(out, dtype=torch.float32).view(-1, 1)

                    loss = loss_f1(out, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    stage_loss.append(loss.item() * len(y))
        # print(net_ensemble.boost_rate)
        # store model
        elapsed_tr = time.time() - t0
        sl = 0
        if stage_loss != []:
            sl = np.sqrt(np.sum(stage_loss) / N)

        print(
            f'Stage - {stage}, training time: {elapsed_tr: .1f} sec, model RMSE loss: {sml: .5f}, Ensemble Net RMSE '
            f'Loss: {sl: .5f}')

        net_ensemble.to_file(args.out_f)
        if args.cuda:
            # net_ensemble = net_ensemble.to(device)
            net_ensemble = net_ensemble.to_cuda()
        net_ensemble = DynamicNetForMLPImage.from_file(args.out_f, lambda stage: MLP_Transformer.get_model(stage, args))  # MLP_ResNet MLP_Transformer

        if args.cuda:
            net_ensemble.to_cuda()
        net_ensemble.to_eval()  # Set the models in ensemble net to eval mode

        # Train
        tr_rmse = root_mse(net_ensemble, train_loader)
        if args.cv:
            val_rmse = root_mse(net_ensemble, val_loader)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_stage = stage

        te_rmse = root_mse(net_ensemble, test_loader)

        print(f'Stage: {stage}  RMSE@Train: {tr_rmse:.5f}, RMSE@Val: {val_rmse:.5f}, RMSE@Test: {te_rmse:.5f}')

        loss_models[stage, 0], loss_models[stage, 1] = tr_rmse, te_rmse

    tr_rmse, te_rmse = loss_models[best_stage, 0], loss_models[best_stage, 1]
    print(f'Best validation stage: {best_stage}  RMSE@Train: {tr_rmse:.5f}, final RMSE@Test: {te_rmse:.5f}')
    loss_models = loss_models.detach().cpu().numpy()
    # fname = './results/' + 'rmse'
    # np.savez(fname, rmse=loss_models, params=args)

    print("best_stage:", best_stage)
    net_ensemble = DynamicNetForMLPImage.from_file(args.out_f, lambda best_stage: MLP_Transformer.get_model(best_stage, args))  # MLP_ResNet MLP_Transformer

    # Note: processed parameter
    if args.cuda:
        net_ensemble.to_cuda()
    net_ensemble.to_eval()  # Note: processed parameter

    # RESULTS
    train_pred, train_true = get_predictions(net_ensemble, train_loader)
    val_pred, val_true = get_predictions(net_ensemble, val_loader)
    test_pred, test_true = get_predictions(net_ensemble, test_loader)

    prediction_train = train_pred
    prediction_val = val_pred
    prediction_test = test_pred
    y_train = train_true
    y_val = val_true
    y_test = test_true
    # _, prediction_train = net_ensemble.forward(X_train_t)
    # _, prediction_val = net_ensemble.forward(X_val_t)
    # _, prediction_test = net_ensemble.forward(X_test_t)

    from sklearn.metrics import r2_score

    R2_train = r2_score(y_train, prediction_train)
    R2_val = r2_score(y_val, prediction_val)
    R2_test = r2_score(y_test, prediction_test)

    # R2_train = 1 - torch.mean((y_train - prediction_train) ** 2) / torch.mean(
    #     (y_train - torch.mean(y_train)) ** 2)
    # R2_val = 1 - torch.mean((y_val - prediction_val) ** 2) / torch.mean(
    #     (y_val - torch.mean(y_val)) ** 2)
    # R2_test = 1 - torch.mean((y_test - prediction_test) ** 2) / torch.mean(
    #     (y_test - torch.mean(y_test)) ** 2)
    print("------------------------RESULTS------------------------")
    print(f'train: R2：{R2_train}\n')
    print(f'val: R2：{R2_val}\n')
    print(f'test: R2：{R2_test}\n')

    rmse_train = np.sqrt(mean_squared_error(y_train, prediction_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, prediction_val))
    rmse_test = np.sqrt(mean_squared_error(y_test, prediction_test))
    print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train, prediction_train))}\n')
    print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val, prediction_val))}\n')
    print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test, prediction_test))}\n')

    # Save the trained model (optional)
    # torch.save(trained_model.state_dict(), 'checkpoint/1DGBCNN_model.pth')

    # Train Set、Validation Set Test Set MAE
    mae_train = mean_absolute_error(y_train, prediction_train)
    mae_val = mean_absolute_error(y_val, prediction_val)
    mae_test = mean_absolute_error(y_test, prediction_test)

    # #  Train Set、Validation Set Test Set MAPE
    # mape_train = mean_absolute_percentage_error(y_train.numpy(), prediction_train.detach().numpy())
    # mape_val = mean_absolute_percentage_error(y_val.numpy(), prediction_val.detach().numpy())
    # mape_test = mean_absolute_percentage_error(y_test.numpy(), prediction_test.detach().numpy())

    print(f'train: MAE：{mae_train}\n')
    print(f'val: MAE：{mae_val}\n')
    print(f'test: MAE：{mae_test}\n')
    # print(f'train: MAPE：{mape_train}\n')
    # print(f'val: MAPE：{mape_val}\n')
    # print(f'test: MAPE：{mape_test}\n')

    # Save results to log file
    argsDict = args.__dict__
    # RESULTS Save results to log file
    log_path = '../checkpoint/tableImageTransformerTrain_log.txt'  # Transformer

    with open(log_path, 'a', encoding='utf-8') as f:  # utf-8
        # Add file header delimiter
        f.write("\n" + "=" * 60 + "\n")
        f.write("{" + "Training Results Record（Transformer ）".center(58) + "}\n")  # Transformer
        f.write("=" * 60 + "\n\n")

        # 、 、Test Set Metric
        f.write("## Model Evaluation Metrics\n")
        f.write("-" * 60 + "\n")
        f.write("|       Metric       |  Train Set  |  Validation Set  |  Test Set  |\n")
        f.write("-" * 60 + "\n")
        f.write(f"|     R² Value     | {R2_train:.4f}    | {R2_val:.4f}    | {R2_test:.4f}    |\n")
        f.write(f"|     RMSE     | {rmse_train:.4f}    | {rmse_val:.4f}    | {rmse_test:.4f}    |\n")
        f.write(f"|     MAE      | {mae_train:.4f}    | {mae_val:.4f}    | {mae_test:.4f}    |\n")
        f.write("-" * 60 + "\n\n")

        # Save hyperparameters
        f.write("## Hyperparameter Settings\n")
        f.write("-" * 60 + "\n")
        f.write("Name | Value\n")
        f.write("-" * 60 + "\n")
        for eachArg, value in argsDict.items():
            f.write(f"{eachArg.ljust(20)} | {str(value).ljust(40)}\n")
        f.write("-" * 60 + "\n\n")

        # Add file footer delimiter
        f.write("=" * 60 + "\n\n")