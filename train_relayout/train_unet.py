import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import yaml
from dataloader import RelayoutDataset
from loss import WeightedDiceLoss
from utils import avg_extra_mask_intersection, median_extra_object, EarlyStopping
from utils import plot_masks
from datetime import datetime


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs,
                early_stopping, is_wandb=None, checkpoint_dir=None, save_interval=10, experiment_name=None, date=None):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        total_dice_loss = 0
        total_intersection_loss = 0
        for inputs, targets, weight_vector in train_loader:
            # Move data to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss, dice_loss, intersection_loss = criterion(outputs, targets, weight_vector)
            total_loss += loss.item()
            total_dice_loss += dice_loss.item()
            total_intersection_loss += intersection_loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

        if is_wandb:
            wandb.log({"epoch": epoch + 1, "loss": total_loss / len(train_loader.dataset)})
            wandb.log({"epoch": epoch + 1, "dice_loss": total_dice_loss / len(train_loader.dataset)})
            wandb.log({"epoch": epoch + 1, "intersection_loss": total_intersection_loss / len(train_loader.dataset)})

        # Checkpoint saving
        if checkpoint_dir is not None and (epoch + 1) % save_interval == 0:
            torch.save(model.state_dict(),
                       f'{checkpoint_dir}/epoch_{epoch}_unet_model_{experiment_name}_{date}.pth')

        training_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {training_loss}")
        # Evaluation after each epoch
        val_loss = evaluate_model(model, test_loader, criterion, device, False, is_wandb, epoch, mode="eval")
        print(f"Test Loss: {val_loss}")

        early_stopping.check_stop(val_loss)  # Call early stopping

        if early_stopping.early_stop:
            print("Early stopping")
            if checkpoint_dir is not None:
                torch.save(model.state_dict(),
                           f'{checkpoint_dir}/epoch_{epoch}_unet_model_{experiment_name}_{date}_early_stopping.pth')
            break


def evaluate_model(model, test_loader, criterion, device, visualization,
                   is_wandb=None, epoch=0, mode="eval"):
    model.eval()
    total_loss = 0
    total_dice_loss = 0
    total_intersection_loss = 0
    extra_mask_intersection = 0
    extra_mask_median_score = 0

    with torch.no_grad():
        for sample_index, (inputs, targets, weight_vector) in enumerate(test_loader):
            # Move data to the same device as the model
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            outputs = model(inputs)
            loss, dice_loss, intersection_loss = criterion(outputs, targets, weight_vector)
            total_loss += loss.item()
            total_dice_loss += dice_loss.item()
            total_intersection_loss += intersection_loss.item()

            # outputs to logits
            binary_outputs = torch.where(outputs > 0.5, 1, 0)
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            binary_outputs = binary_outputs.cpu().numpy()

            # Calculate metrics : 1) Average iou for the extra k+1_th predicted mask with other previous k_th masks
            #                     2) Median size of extra k+1_th predicted object
            avg_iou_sample = avg_extra_mask_intersection(binary_outputs, targets, weight_vector)
            median_sample_score = median_extra_object(binary_outputs, targets, weight_vector)

            extra_mask_intersection += avg_iou_sample
            extra_mask_median_score += median_sample_score

            if visualization and mode == "test":
                plot_masks(inputs_np, targets_np, binary_outputs, is_wandb, weight_vector,
                           avg_iou_sample, median_sample_score, epoch, sample_index)

        if is_wandb and mode == "test":
            wandb.log({"Extra Mask Intersection Score Test": extra_mask_intersection / len(test_loader.dataset)})
            wandb.log({"Extra Mask Median Score Test": extra_mask_median_score / len(test_loader.dataset)})

        if is_wandb and mode == "eval":
            wandb.log({"eval_loss": total_loss / len(test_loader.dataset)})
            wandb.log({"eval_dice_loss": total_dice_loss / len(test_loader.dataset)})
            wandb.log({"eval_intersection_loss": total_intersection_loss / len(test_loader.dataset)})
            wandb.log({"Extra Mask Intersection Score Eval": extra_mask_intersection / len(test_loader.dataset)})
            wandb.log({"Extra Mask Median Score Eval": extra_mask_median_score / len(test_loader.dataset)})

    return total_loss / len(test_loader.dataset)


if __name__ == "__main__":
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create datasets and loaders
    with open('train_relayout/unet_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    if config["visualization"]["wandb"]:
        # Initialize W&B with a project name
        wandb.init(project="unet-segmentation", name="unet-run")
        wandb.config.update(config)  # Log configuration to W&B

    # define paths to train and eval data
    paths = config['paths']
    train_data_dir = paths['train_data_dir']
    test_data_dir = paths['test_data_dir']

    train_dataset = RelayoutDataset(train_data_dir, config)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=config['training']['shuffle'])

    test_dataset = RelayoutDataset(test_data_dir, config)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Define the U-Net model
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=config['model']['in_channels'], out_channels=config['model']['out_channels'],
                           init_features=config['model']['init_features'], pretrained=False)

    model = model.to(config['training']['device'])

    criterion = WeightedDiceLoss(smooth=config["loss"]["dice_smooth"],
                                 intersection_penalty=config["loss"]["intersection_penalty"],
                                 intersection_loss=config["loss"]["intersection_loss"],
                                 dice_penalty=config["loss"]["dice_penalty"],
                                 extra_mask_weight=config["loss"]["last_channel_weights"],
                                 existing_mask_weights=config["loss"]["mask_channel_weights"],)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['lr'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=config['training']['lr_decay'],
                                          step_size=config['training']['step_size'])

    device = config['training']['device']
    visualization = config['visualization']['visualize']

    early_stopping = EarlyStopping(patience=config['training']['patience'], min_delta=config['training']['min_delta'])

    # Train and evaluate
    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device,
                num_epochs=config['training']['num_epochs'], early_stopping=early_stopping,
                is_wandb=config["visualization"]["wandb"], checkpoint_dir=config['paths']['checkpoint_dir'],
                save_interval=config['training']['checkpoint_interval'], experiment_name=config["training"]["exp_name"],
                date=date_str)
    evaluate_model(model, test_loader, criterion, device, visualization, is_wandb=config["visualization"]["wandb"], mode="test")

    # Save the model
    if config['paths']['checkpoint_dir'] is not None:
        checkpoint_dir = config['paths']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(),
                   f'{checkpoint_dir}/final_unet_model_{config["training"]["exp_name"]}_{date_str}.pth')

