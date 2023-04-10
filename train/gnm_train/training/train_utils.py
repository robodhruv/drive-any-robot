import wandb
import os
import numpy as np
from typing import List, Optional, Dict

from gnm_train.visualizing.action_utils import visualize_traj_pred
from gnm_train.visualizing.distance_utils import visualize_dist_pred, visualize_dist_pairwise_pred
from gnm_train.visualizing.visualize_utils import to_numpy
from gnm_train.training.logger import Logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam


def train_eval_loop(
    model: nn.Module,
    optimizer: Adam,
    train_dist_loader: DataLoader,
    train_action_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    epochs: int,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    pairwise_test_freq: int = 5,
    current_epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    use_wandb: bool = True,
):
    """
    Train and evaluate the model for several epochs.

    Args:
        model: model to train
        optimizer: optimizer to use
        train_dist_loader: dataloader for training distance predictions
        train_action_loader: dataloader for training action predictions
        test_dataloaders: dict of dataloaders for testing
        epochs: number of epochs to train
        device: device to train on
        project_folder: folder to save checkpoints and logs
        log_freq: frequency of logging to wandb
        image_log_freq: frequency of logging images to wandb
        num_images_log: number of images to log to wandb
        pairwise_test_freq: frequency of testing pairwise distance accuracy
        current_epoch: epoch to start training from
        alpha: tradeoff between distance and action loss
        learn_angle: whether to learn the angle or not
        use_wandb: whether to log to wandb or not
        load_best: whether to load the best model or not
    """
    assert 0 <= alpha <= 1
    latest_path = os.path.join(project_folder, f"latest.pth")

    for epoch in range(current_epoch, current_epoch + epochs):
        print(
            f"Start GNM Training Epoch {epoch}/{current_epoch + epochs - 1}"
        )
        train(
            model,
            optimizer,
            train_dist_loader,
            train_action_loader,
            device,
            project_folder,
            normalized,
            epoch,
            alpha,
            learn_angle,
            print_log_freq,
            image_log_freq,
            num_images_log,
            use_wandb,
        )

        eval_total_losses = []
        for dataset_type in test_dataloaders:
            print(
                f"Start {dataset_type} GNM Testing Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            dist_loader = test_dataloaders[dataset_type]["distance"]
            action_loader = test_dataloaders[dataset_type]["action"]
            test_dist_loss, test_action_loss = evaluate(
                dataset_type,
                model,
                dist_loader,
                action_loader,
                device,
                project_folder,
                normalized,
                epoch,
                alpha,
                learn_angle,
                print_log_freq,
                image_log_freq,
                num_images_log,
                use_wandb,
            )

            total_eval_loss = get_total_loss(test_dist_loss, test_action_loss, alpha)
            eval_total_losses.append(total_eval_loss)
            wandb.log({f"{dataset_type}_total_loss": total_eval_loss})
            print(f"{dataset_type}_total_loss: {total_eval_loss}")
            wandb.log({f"{dataset_type}_dist_loss": test_dist_loss})
            print(f"{dataset_type}_dist_loss: {test_dist_loss}")
            wandb.log({f"{dataset_type}_action_loss": test_action_loss})
            print(f"{dataset_type}_action_loss: {test_action_loss}")

        checkpoint = {
            "epoch": epoch,
            "model": model,
            "optimizer": optimizer,
            "avg_eval_loss": np.mean(eval_total_losses),
        }

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, latest_path)
        torch.save(checkpoint, numbered_path)  # keep track of model at every epoch

        if (epoch - current_epoch) % pairwise_test_freq == 0:
            print(f"Start Pairwise Testing Epoch {epoch}/{current_epoch + epochs - 1}")
            for dataset_type in test_dataloaders:
                if "pairwise" in test_dataloaders[dataset_type]:
                    pairwise_dist_loader = test_dataloaders[dataset_type]["pairwise"]
                    pairwise_accuracy = pairwise_acc(
                        model,
                        pairwise_dist_loader,
                        device,
                        project_folder,
                        epoch,
                        dataset_type,
                        print_log_freq,
                        image_log_freq,
                        num_images_log,
                        use_wandb=use_wandb,
                    )
                    wandb.log({f"{dataset_type}_pairwise_acc": pairwise_accuracy})
                    print(f"{dataset_type}_pairwise_acc: {pairwise_accuracy}")
    print()


def train(
    model: nn.Module,
    optimizer: Adam,
    train_dist_loader: DataLoader,
    train_action_loader: DataLoader,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Train the model for one epoch.

    Args:
        model: model to train
        optimizer: optimizer to use
        train_dist_loader: dataloader for distance training
        train_action_loader: dataloader for action training
        device: device to use
        project_folder: folder to save images to
        epoch: current epoch
        alpha: weight of action loss
        learn_angle: whether to learn the angle of the action
        print_log_freq: how often to print loss
        image_log_freq: how often to log images
        num_images_log: number of images to log
        use_wandb: whether to use wandb
    """
    model.train()
    dist_loss_logger = Logger("dist_loss", "train", window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", "train", window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", "train", window_size=print_log_freq
    )
    total_loss_logger = Logger("total_loss", "train", window_size=print_log_freq)

    variables = [
        dist_loss_logger,
        action_loss_logger,
        action_waypts_cos_sim_logger,
        multi_action_waypts_cos_sim_logger,
        total_loss_logger,
    ]

    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", "train", window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", "train", window_size=print_log_freq
        )
        variables.extend(
            [action_orien_cos_sim_logger, multi_action_orien_cos_sim_logger]
        )

    num_batches = min(len(train_dist_loader), len(train_action_loader))
    for i, val in enumerate(zip(train_dist_loader, train_action_loader)):
        dist_vals, action_vals = val
        (
            dist_obs_image,
            dist_goal_image,
            dist_trans_obs_image,
            dist_trans_goal_image,
            dist_label,
            dist_dataset_index,
        ) = dist_vals
        (
            action_obs_image,
            action_goal_image,
            action_trans_obs_image,
            action_trans_goal_image,
            action_goal_pos,
            action_label,
            action_dataset_index,
        ) = action_vals
        dist_obs_data = dist_trans_obs_image.to(device)
        dist_goal_data = dist_trans_goal_image.to(device)
        dist_label = dist_label.to(device)

        optimizer.zero_grad()

        dist_pred, _ = model(dist_obs_data, dist_goal_data)
        dist_loss = F.mse_loss(dist_pred, dist_label)

        action_obs_data = action_trans_obs_image.to(device)
        action_goal_data = action_trans_goal_image.to(device)
        action_label = action_label.to(device)

        _, action_pred = model(action_obs_data, action_goal_data)
        action_loss = F.mse_loss(action_pred, action_label)
        action_waypts_cos_similairity = F.cosine_similarity(
            action_pred[:, :, :2], action_label[:, :, :2], dim=-1
        ).mean()
        multi_action_waypts_cos_sim = F.cosine_similarity(
            torch.flatten(action_pred[:, :, :2], start_dim=1),
            torch.flatten(action_label[:, :, :2], start_dim=1),
            dim=-1,
        ).mean()
        if learn_angle:
            action_orien_cos_sim = F.cosine_similarity(
                action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
            ).mean()
            multi_action_orien_cos_sim = F.cosine_similarity(
                torch.flatten(action_pred[:, :, 2:], start_dim=1),
                torch.flatten(action_label[:, :, 2:], start_dim=1),
                dim=-1,
            ).mean()
            action_orien_cos_sim_logger.log_data(action_orien_cos_sim.item())
            multi_action_orien_cos_sim_logger.log_data(
                multi_action_orien_cos_sim.item()
            )

        total_loss = get_total_loss(dist_loss, action_loss, alpha)
        total_loss.backward()
        optimizer.step()

        dist_loss_logger.log_data(dist_loss.item())
        action_loss_logger.log_data(action_loss.item())
        action_waypts_cos_sim_logger.log_data(action_waypts_cos_similairity.item())
        multi_action_waypts_cos_sim_logger.log_data(multi_action_waypts_cos_sim.item())
        total_loss_logger.log_data(total_loss.item())

        if use_wandb:
            data_log = {}
            for var in variables:
                data_log[var.full_name()] = var.latest()
            wandb.log(data_log)

        if i % print_log_freq == 0:
            log_display = f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
            for var in variables:
                print(log_display + var.display())
            print()

        if i % image_log_freq == 0:
            visualize_dist_pred(
                to_numpy(dist_obs_image),
                to_numpy(dist_goal_image),
                to_numpy(dist_pred),
                to_numpy(dist_label),
                "train",
                project_folder,
                epoch,
                num_images_log,
                use_wandb=use_wandb,
            )
            visualize_traj_pred(
                to_numpy(action_obs_image),
                to_numpy(action_goal_image),
                to_numpy(action_dataset_index),
                to_numpy(action_goal_pos),
                to_numpy(action_pred),
                to_numpy(action_label),
                "train",
                normalized,
                project_folder,
                epoch,
                num_images_log,
                use_wandb=use_wandb,
            )
    return


def evaluate(
    eval_type: str,
    model: nn.Module,
    eval_dist_loader: DataLoader,
    eval_action_loader: DataLoader,
    device: torch.device,
    project_folder: str,
    normalized: bool,
    epoch: int = 0,
    alpha: float = 0.5,
    learn_angle: bool = True,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        eval_type (string): f"{data_type}_{eval_type}" (e.g. "recon_train", "gs_test", etc.)
        model (nn.Module): model to evaluate
        eval_dist_loader (DataLoader): dataloader for distance prediction
        eval_action_loader (DataLoader): dataloader for action prediction
        device (torch.device): device to use for evaluation
        project_folder (string): path to project folder
        epoch (int): current epoch
        alpha (float): weight for action loss
        learn_angle (bool): whether to learn the angle of the action
        print_log_freq (int): frequency of printing loss
        image_log_freq (int): frequency of logging images
        num_images_log (int): number of images to log
        use_wandb (bool): whether to use wandb for logging
    """
    model.eval()
    dist_loss_logger = Logger("dist_loss", eval_type, window_size=print_log_freq)
    action_loss_logger = Logger("action_loss", eval_type, window_size=print_log_freq)
    action_waypts_cos_sim_logger = Logger(
        "action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    multi_action_waypts_cos_sim_logger = Logger(
        "multi_action_waypts_cos_sim", eval_type, window_size=print_log_freq
    )
    total_loss_logger = Logger(
        "total_loss_logger", eval_type, window_size=print_log_freq
    )

    variables = [
        dist_loss_logger,
        action_loss_logger,
        action_waypts_cos_sim_logger,
        multi_action_waypts_cos_sim_logger,
        total_loss_logger,
    ]
    if learn_angle:
        action_orien_cos_sim_logger = Logger(
            "action_orien_cos_sim", eval_type, window_size=print_log_freq
        )
        multi_action_orien_cos_sim_logger = Logger(
            "multi_action_orien_cos_sim", eval_type, window_size=print_log_freq
        )
        variables.extend(
            [action_orien_cos_sim_logger, multi_action_orien_cos_sim_logger]
        )

    num_batches = min(len(eval_dist_loader), len(eval_action_loader))

    with torch.no_grad():
        for i, val in enumerate(zip(eval_dist_loader, eval_action_loader)):
            dist_vals, action_vals = val
            (
                dist_obs_image,
                dist_goal_image,
                dist_trans_obs_image,
                dist_trans_goal_image,
                dist_label,
                dist_dataset_index,
            ) = dist_vals
            (
                action_obs_image,
                action_goal_image,
                action_trans_obs_image,
                action_trans_goal_image,
                action_goal_pos,
                action_label,
                action_dataset_index,
            ) = action_vals
            dist_obs_data = dist_trans_obs_image.to(device)
            dist_goal_data = dist_trans_goal_image.to(device)
            dist_label = dist_label.to(device)

            dist_pred, _ = model(dist_obs_data, dist_goal_data)
            dist_loss = F.mse_loss(dist_pred, dist_label)

            action_obs_data = action_trans_obs_image.to(device)
            action_goal_data = action_trans_goal_image.to(device)
            action_label = action_label.to(device)

            _, action_pred = model(action_obs_data, action_goal_data)
            action_loss = F.mse_loss(action_pred, action_label)
            action_waypts_cos_sim = F.cosine_similarity(
                action_pred[:, :, :2], action_label[:, :, :2], dim=-1
            ).mean()
            multi_action_waypts_cos_sim = F.cosine_similarity(
                torch.flatten(action_pred[:, :, :2], start_dim=1),
                torch.flatten(action_label[:, :, :2], start_dim=1),
                dim=-1,
            ).mean()
            if learn_angle:
                action_orien_cos_sim = F.cosine_similarity(
                    action_pred[:, :, 2:], action_label[:, :, 2:], dim=-1
                ).mean()
                multi_action_orien_cos_sim = F.cosine_similarity(
                    torch.flatten(action_pred[:, :, 2:], start_dim=1),
                    torch.flatten(action_label[:, :, 2:], start_dim=1),
                    dim=-1,
                ).mean()
                action_orien_cos_sim_logger.log_data(action_orien_cos_sim.item())
                multi_action_orien_cos_sim_logger.log_data(
                    multi_action_orien_cos_sim.item()
                )

            total_loss = alpha * (1e-3 * dist_loss) + (1 - alpha) * action_loss

            dist_loss_logger.log_data(dist_loss.item())
            action_loss_logger.log_data(action_loss.item())
            action_waypts_cos_sim_logger.log_data(action_waypts_cos_sim.item())
            multi_action_waypts_cos_sim_logger.log_data(
                multi_action_waypts_cos_sim.item()
            )
            total_loss_logger.log_data(total_loss.item())

            if i % print_log_freq == 0:
                log_display = f"(epoch {epoch}) (batch {i}/{num_batches - 1}) "
                for var in variables:
                    print(log_display + var.display())
                print()

            if i % image_log_freq == 0:
                visualize_dist_pred(
                    to_numpy(dist_obs_image),
                    to_numpy(dist_goal_image),
                    to_numpy(dist_pred),
                    to_numpy(dist_label),
                    eval_type,
                    project_folder,
                    epoch,
                    num_images_log,
                    use_wandb=use_wandb,
                )
                visualize_traj_pred(
                    to_numpy(action_obs_image),
                    to_numpy(action_goal_image),
                    to_numpy(action_dataset_index),
                    to_numpy(action_goal_pos),
                    to_numpy(action_pred),
                    to_numpy(action_label),
                    eval_type,
                    normalized,
                    project_folder,
                    epoch,
                    num_images_log,
                    use_wandb=use_wandb,
                )
    data_log = {}
    for var in variables:
        log_display = f"(epoch {epoch}) "
        data_log[var.full_name()] = var.average()
        print(log_display + var.display())
    print()
    if use_wandb:
        wandb.log(data_log)
    return dist_loss_logger.average(), action_loss_logger.average()


def pairwise_acc(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
    save_folder: str,
    epoch: int,
    eval_type: str,
    print_log_freq: int = 100,
    image_log_freq: int = 1000,
    num_images_log: int = 8,
    use_wandb: bool = True,
    display: bool = False,
):
    """
    Evaluate the model on the pairwise distance accuracy metric. Given 1 observation and 2 subgoals, the model should determine which goal is closer.

    Args:
        model (nn.Module): The model to evaluate.
        eval_loader (DataLoader): The dataloader for the evaluation dataset.
        device (torch.device): The device to use for evaluation.
        save_folder (str): The folder to save the evaluation results.
        epoch (int): The current epoch.
        eval_type (str): The type of evaluation. Can be "train" or "val".
        print_log_freq (int, optional): The frequency at which to print the evaluation results. Defaults to 100.
        image_log_freq (int, optional): The frequency at which to log the evaluation results. Defaults to 1000.
        num_images_log (int, optional): The number of images to log. Defaults to 32.
        use_wandb (bool, optional): Whether to use wandb for logging. Defaults to True.
        display (bool, optional): Whether to display the evaluation results. Defaults to False.
    """
    correct_list = []
    model.eval()
    num_batches = len(eval_loader)

    with torch.no_grad():
        for i, vals in enumerate(eval_loader):
            (
                obs_image,
                close_image,
                far_image,
                transf_obs_image,
                transf_close_image,
                transf_far_image,
                close_dist_label,
                far_dist_label,
            ) = vals
            transf_obs_image = transf_obs_image.to(device)
            transf_close_image = transf_close_image.to(device)
            transf_far_image = transf_far_image.to(device)

            close_pred, _ = model(transf_obs_image, transf_close_image)
            far_pred, _ = model(transf_obs_image, transf_far_image)

            close_pred_flat = close_pred.reshape(close_pred.shape[0])
            far_pred_flat = far_pred.reshape(far_pred.shape[0])

            close_pred_flat = to_numpy(close_pred_flat)
            far_pred_flat = to_numpy(far_pred_flat)

            correct = np.where(far_pred_flat > close_pred_flat, 1, 0)
            correct_list.append(correct)
            if i % print_log_freq == 0:
                print(f"({i}/{num_batches}) batch of points processed")

            if i % image_log_freq == 0:
                visualize_dist_pairwise_pred(
                    to_numpy(obs_image),
                    to_numpy(close_image),
                    to_numpy(far_image),
                    to_numpy(close_pred),
                    to_numpy(far_pred),
                    to_numpy(close_dist_label),
                    to_numpy(far_dist_label),
                    eval_type,
                    save_folder,
                    epoch,
                    num_images_log,
                    use_wandb,
                    display,
                )
        if len(correct_list) == 0:
            return 0
        return np.concatenate(correct_list).mean()


def get_total_loss(dist_loss, action_loss, alpha):
    """Get total loss from distance and action loss."""
    return alpha * (1e-2 * dist_loss) + (1 - alpha) * action_loss


def load_model(model, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    loaded_model = checkpoint["model"]
    try:  # for DataParallel
        state_dict = loaded_model.module.state_dict()
        model.load_state_dict(state_dict)
    except (RuntimeError, AttributeError) as e:
        state_dict = loaded_model.state_dict()
        model.load_state_dict(state_dict)


def get_saved_optimizer(
    checkpoint: dict, device: torch.device
) -> torch.optim.Optimizer:
    optimizer = checkpoint["optimizer"]
    optimizer_to(optimizer, device)
    return optimizer


def optimizer_to(optim, device):
    """Move optimizer state to device."""
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
