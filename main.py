import argparse
import os
import json
from tqdm import tqdm
from datetime import datetime

import numpy as np
from sklearn.metrics import mean_squared_error, classification_report, matthews_corrcoef, f1_score
from sklearn.model_selection import KFold

from dataloader import MultimodalDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
from transformers import AdamW, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from model import MultimodalModel

#os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-ac3C47b3-456e-56ff-aa3e-5731e429d659"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def pad_collate(batch):
    target = np.array([item[0] for item in batch], dtype=np.float32)
    video = [item[1] for item in batch]
    audio = [item[2] for item in batch]
    text = [item[3] for item in batch]
    subclip_masks = [item[4] for item in batch]
    lens = [len(x) for x in video]
    
    video = nn.utils.rnn.pad_sequence(video, batch_first=True, padding_value=0)
    audio = nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)
    text = nn.utils.rnn.pad_sequence(text, batch_first=True, padding_value=0)
    subclip_masks = nn.utils.rnn.pad_sequence(subclip_masks, batch_first=True, padding_value=0)
    
    lens = torch.LongTensor(lens)
    target = torch.tensor(target)
    mask = torch.arange(video.shape[1]).expand(len(lens), video.shape[1]) < lens.unsqueeze(1)
    mask = mask
    
    return [target, video, audio, text, mask, subclip_masks]

def train(fold, model, device, trainloader, optimizer, loss_function, epoch, grad_norms, activations, weight_distributions):
    current_loss = 0.0
    model = model.train()
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        targets, video, audio, text, mask, subclip_masks = data
        video = video.to(device)
        audio = audio.to(device)
        text = text.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        subclip_masks = subclip_masks.to(device)
        
        optimizer.zero_grad()
        outputs, attention_scores_cross_transformer = model(video, audio, text, mask, subclip_masks)
        loss = loss_function(outputs, targets)
        loss.backward()
        
        # Save gradients
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        optimizer.step()
        
        # Save activation histograms
        activation_values = []
        layer_names = []
        # Function to hook into each layer and collect activations
        def hook_fn(module, input, output):
            activation_values.append(output)
        # Register hooks for all layers
        hooks = []
        for name, layer in model.named_children():
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
            layer_names.append(name)
        # Perform forward pass to collect activations
        with torch.no_grad():
            model(video, audio, text, mask, subclip_masks)
        # Remove hooks
        for hook in hooks:
            hook.remove()
        activation_values_dict = {}
        for j, name in enumerate(layer_names):
            activation_values_dict[name] = activation_values[j]
        activations.append(activation_values_dict)
            
        # Plot attention maps
        for key in attention_scores_cross_transformer:
            fig, axes = plt.subplots(len(attention_scores_cross_transformer[key]), 1, figsize=(16,4*len(attention_scores_cross_transformer[key])))
            count = 0
            for ax, heatmap in zip(axes, attention_scores_cross_transformer[key]):
                heatmap = heatmap.detach().cpu().numpy()
                im = ax.imshow(heatmap, cmap='viridis', interpolation='nearest')
                ax.set_title(f'Example {count+1}')
                ax.axis('off')
                count += 1
        plt.tight_layout()
        cbar = fig.colorbar(im, ax=axes.tolist(), pad=0.02)
        cbar.set_label('Color Scale')
        cbar.ax.set_ylabel('Label on Color Bar', rotation=270, labelpad=20)
        cbar.ax.tick_params(axis='y', length=0, width=0, pad=10)
        cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])
        #plt.savefig(f'/home/jovyan/multimodal_mpc_call/plots/attention_map_{key}_{fold}_epoch_{epoch}_batch_{i}.png')
        
        weight_distribution_info = []
        for name, param in model.named_parameters():
            if param.requires_grad and 'bias' not in name:
                weight_distribution_info.append((name, param.data.clone().cpu().numpy()))
        weight_distributions.append(weight_distribution_info)
        
        current_loss += loss.item()
        outputs.detach()
        del outputs
        
    epoch_loss = current_loss / len(trainloader)
    
    return epoch_loss

def test(fold, model, device, testloader, results, movement=False, val=False):
    preds = []
    true = []
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader, 0), total=len(testloader)):
            targets, video, audio, text, mask, subclip_masks = data
            video = video.to(device)
            audio = audio.to(device)
            text = text.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            subclip_masks = subclip_masks.to(device)
            outputs, _ = model(video, audio, text, mask, subclip_masks)
            if movement:
                outputs = F.sigmoid(outputs) > 0.5
            preds.extend(outputs.detach().cpu().numpy())
            true.extend(targets.detach().cpu().numpy())
            
    res = {}
    preds = np.array(preds)
    true = np.array(true)
    if not movement:
        res_list = [mean_squared_error(true[:, i], preds[:, i], squared=False) for i in range(true[0].shape[-1])]
        res['rmse'] = res_list
    else:
        res_list = [f1_score(true[:, i], preds[:, i]) for i in range(true[0].shape[-1])]
        res['f1-score'] = res_list
        res_list = [matthews_corrcoef(true[:, i], preds[:, i]) for i in range(true[0].shape[-1])]
        res['mcc'] = res_list
        
    results[fold] = res

    return results

def main(config):
    EPOCHS = config.epochs
    BATCH_SIZE = config.batch_size
    HIDDEN_DIM = config.hidden_dim
    EMBEDDING_DIM = [int(e) for e in config.embedding_dim]
    DROPOUT = config.dropout
    DATA_DIR = config.data_dir
    NUM_LAYERS = config.num_layers
    NUM_HEADS = config.num_heads
    N_FOLDS = config.n_folds
    MAX_LEN = config.max_len
    LEARNING_RATE = config.learning_rate
    SAVE_DIR = config.save_dir
    OPTIMIZER = config.optimizer
    SCHEDULER = config.use_scheduler
    PATIENCE = config.patience
    MIN_EPOCHS = config.min_epochs
    MOVEMENT = config.movement
    OFFSET = config.offset
    VOLATILITY_WINDOW = config.volatility_window
    SUBCLIP_MAXLEN = config.subclip_maxlen
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: ", DEVICE)
    dataset = MultimodalDataset(DATA_DIR, SUBCLIP_MAXLEN)
    dataset.load_data(DATA_DIR, OFFSET, MOVEMENT, VOLATILITY_WINDOW)
    train_idx, val_idx, test_idx = dataset.make_splits()
    
    model = MultimodalModel(
        hidden_dim=HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        movement=MOVEMENT
    )
    
    #model = DataParallel(model)
    #model = model.to('cuda') 
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True)
    results_val = {}
    results = {}
    
    if MOVEMENT:
        loss_function = nn.BCEWithLogitsLoss()
    else:
        loss_function = nn.MSELoss()
        
    #for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    for fold in range(N_FOLDS):
        print('------------fold no---------{}----------------------'.format(fold))
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler, collate_fn=pad_collate)
        valloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_subsampler, collate_fn=pad_collate)
        testloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_subsampler, collate_fn=pad_collate)
        
        model.train()
        model.to(DEVICE)
        
        if OPTIMIZER=="adamw":
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        if SCHEDULER:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=5, num_training_steps=EPOCHS
            )
        else:
            scheduler = None
            
        best_metric = np.inf
        counter = 0
        training_losses = []
        sis_losses = []
        sil_losses = []
        cur_losses = []
        gold_losses = []
        Y10_bond_losses = []
        M3_bond_losses = []
        learning_rates = []
        grad_norms = []
        activations = []
        weight_distributions = []
        for epoch in range(1, EPOCHS +1):
            current_loss = train(fold, model, DEVICE, trainloader, optimizer, loss_function, epoch, grad_norms, activations, weight_distributions)
            training_losses.append(current_loss)
            results_val = test(fold, model, DEVICE, valloader, results_val, val=True, movement=MOVEMENT)
      
            if MOVEMENT:
                metrik = "f1-score"
            else:
                metrik = "rmse"
                
            sis_losses.append(results_val[fold][metrik][0])
            sil_losses.append(results_val[fold][metrik][1])
            cur_losses.append(results_val[fold][metrik][2])
            gold_losses.append(results_val[fold][metrik][3])
            Y10_bond_losses.append(results_val[fold][metrik][4])
            M3_bond_losses.append(results_val[fold][metrik][5])
            
            # Visualize Losses
            plt.figure(figsize=(10,6))
            plt.plot(training_losses, label='Training Loss')
            plt.plot(sis_losses, label='SI(s)')
            plt.plot(sil_losses, label='SI(l)')
            plt.plot(cur_losses, label='Cur')
            plt.plot(gold_losses, label='Gold')
            plt.plot(Y10_bond_losses, label='10Y Bond')
            plt.plot(M3_bond_losses, label='3M Bond')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            plt.savefig(f'/home/jovyan/multimodal_mpc_call/plots/loss_fold_{fold}_epoch_{epoch}.png')
            
            if MOVEMENT:
                print('Epoch %5d | Avg Train Loss %.3f | Val F1s %s'% (epoch, current_loss, str(results_val[fold]['f1-score'])))
            else:
                print('Epoch %5d | Avg Train RMSE %.3f | Val RMSEs %s'% (epoch, np.sqrt(current_loss), str(results_val[fold]['rmse'])))
                
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                learning_rates.append(current_lr)
                # Visualize learning rate
                plt.figure(figsize=(10, 6))
                plt.plot(learning_rates, marker='o', linestyle='-', color='b')
                plt.title('Learning Rate Schedule')
                plt.xlabel('Epoch')
                plt.ylabel('Learning Rate')
                plt.grid(True)
                plt.legend()
                plt.savefig(f'/home/jovyan/multimodal_mpc_call/plots/learning_rate_fold_{fold}_epoch_{epoch}.png')
            if current_loss < best_metric:
                counter = 0
            else:
                counter += 1
                if counter > PATIENCE and epoch > MIN_EPOCHS:
                    print("Early stopping")
                    break
                    
            # Visualize Gradients
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(grad_norms) + 1), grad_norms, marker='o', linestyle='-')
            plt.xlabel('Epochs')
            plt.ylabel('Gradient Norm')
            plt.title('Gradient Norms Over Epochs')
            plt.grid(True)
            plt.savefig(f'/home/jovyan/multimodal_mpc_call/plots/gradients_fold_{fold}_epoch_{epoch}.png')

            # Visualize weight distributions and their corresponding names
            colors = plt.cm.viridis(np.linspace(0, 1, 768))
            for j, weight_distribution_info in enumerate(weight_distributions):
                for name, weight_distribution in weight_distribution_info:
                    plt.figure(figsize=(6, 4))
                    if len(list(weight_distribution.shape)) > 2:
                        for i in range(weight_distribution.shape[2]):
                            values = weight_distribution[:, :, i].flatten()
                            offset = i * 0.01  # Small offset to separate histograms
                            plt.hist(values+offset, bins=50, alpha=0.2, label=f'{name} Channel {i}', color=colors[i], edgecolor='k', lw=0.5)
                    else:
                        plt.hist(weight_distribution.flatten(), bins=50, alpha=0.2, label=f'{name} Channel _', color=colors[0], edgecolor='k', lw=0.5)
                    plt.xlabel('Weight Value')
                    plt.ylabel('Frequency')
                    plt.title(f'Weight Distribution - {name}')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f'/home/jovyan/multimodal_mpc_call/plots/weight_distributions_fold_{fold}_epoch_{j}__layer_{name}.png')
            
            # Plot activation histograms
            activations_plot_dict = {}
            for layer_dict in activations:
                for sub_model in layer_dict:
                    if not torch.is_tensor(layer_dict[sub_model]):
                        if type(layer_dict[sub_model][0]) is list:
                            for i, tensor in enumerate(layer_dict[sub_model][0]):
                                 activations_plot_dict[sub_model + f"_{i}"] = tensor.cpu().numpy().flatten()
                        else:
                            activations_plot_dict[sub_model] = layer_dict[sub_model][0].cpu().numpy().flatten()
                        for layer in layer_dict[sub_model][1]:
                            activations_plot_dict[layer] = layer_dict[sub_model][1][layer].cpu().numpy().flatten()
                    else:
                        activations_plot_dict[sub_model] = layer_dict[sub_model].cpu().numpy().flatten()
            
            # Create a single figure to contain all histograms
            fig, axes = plt.subplots(nrows=len(activations_plot_dict), figsize=(8, 6 * len(activations_plot_dict)))

            # Iterate through the key-value pairs in activations_plot_dict
            for i, (label, data) in enumerate(activations_plot_dict.items()):
                # Create a histogram for each data array
                ax = axes[i]
                ax.hist(data, bins=50)  # You can adjust the number of bins as needed
                ax.set_title(label)
                ax.set_xlabel('Activation Value')
                ax.set_ylabel('Frequency')

            # Adjust layout to prevent overlap
            plt.tight_layout()
            # Save the single figure with all subplots as an image file
            plt.savefig(f'/home/jovyan/multimodal_mpc_call/plots/activations_fold_{fold}_epoch_{epoch}.png')
            
        results = test(fold, model, DEVICE, testloader, results, MOVEMENT)
        
        
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {N_FOLDS} FOLDS')
    print('--------------------------------')
    sums = []
    mccs = []
    print(results)
    for key, value in results.items():
        if MOVEMENT:
            #sums.append(np.array(value['report']['macro avg']['f1-score']))
            sums.append(np.array(value['f1-score']))
            mccs.append(np.array(value['mcc']))
        else:
            sums.append(np.array(value['rmse']))
    avgs = np.mean(sums, axis=0)
    print(f'Average over N runs F1/RMSE: {avgs} %')
    if MOVEMENT:
        avgs = np.mean(mccs, axis=0)
        print(f'Average over N runs MCC: {avgs} %')
              
    save_dir = SAVE_DIR
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
              
    save_folder = os.path.join(save_dir, datetime.today().strftime('%Y-%m-%d'))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
              
    save_dict = vars(config)
    save_dict["results"] = results
              
    now = datetime.now()
    current_time = now.strftime("%H-%M-%S")
              
    keys_values = save_dict.items()
    save_dict = {str(key): str(value) for key, value in keys_values}
    filename = os.path.join(save_folder, current_time) + ".json"
    with open(filename, "w") as outfile:
        json.dump(save_dict, outfile, indent=2)
    
    return avgs

if __name__ == '__main__':
    optimizer_set = {"adam", "adamw"}

    parser = argparse.ArgumentParser(description="Multimodal Transformer")
    parser.add_argument("-lr", "--learning-rate", default=1e-5, type=float)
    parser.add_argument("-bs", "--batch-size", default=2, type=int) # 32
    parser.add_argument("-e", "--epochs", default=30, type=int) # 10
    parser.add_argument('--patience', type=int, default=10) # 10
    parser.add_argument('--min-epochs', type=int, default=10) # 10
    parser.add_argument("-hd", "--hidden-dim", default=2, type=int) #256
    parser.add_argument('-ed','--embedding-dim', nargs='+', default=['768', '768', '768'])
    parser.add_argument("-nl", "--num-layers", default=2, type=int)
    parser.add_argument("-nh", "--num-heads", default=2, type=int)
    parser.add_argument("-ml", "--max-len", default=2500, type=int)
    parser.add_argument("-d", "--dropout", default=0.1, type=float)
    parser.add_argument("-nf", "--n-folds", default=2, type=int) #3
    parser.add_argument("-sm", "--subclip-maxlen", default=-1, type=int)
    parser.add_argument("--optimizer", type=str, choices=optimizer_set, default="adamw")
    parser.add_argument("--use-scheduler", action="store_false")
    parser.add_argument("--movement", action="store_false") # store_true
    parser.add_argument("-o", "--offset", default=2, type=int)
    parser.add_argument("-vola", "--volatility_window", default=3, type=int)
    parser.add_argument("--data-dir", type=str, default="./")
    parser.add_argument("--save-dir", type=str, default="./results")
    config = parser.parse_args()

    main(config)
                