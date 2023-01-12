import os
import wandb
import argparse
import numpy as np
import yaml
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn

from gnm_train.models.gnm import GNM
from gnm_train.models.siamese import SiameseModel
from gnm_train.models.stacked import StackedModel
from gnm_train.data.gnm_dataset import GNM_Dataset
from gnm_train.data.pairwise_distance_dataset import PairwiseDistanceDataset
from gnm_train.training.train_utils import (
    train_eval_loop,
    load_model,
    get_saved_optimizer,
)


def main(config):
    assert config["distance"]["min_dist_cat"] < config["distance"]["max_dist_cat"]
    assert config["action"]["min_dist_cat"] < config["action"]["max_dist_cat"]

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True  # good if input sizes don't vary
    transform = [
        transforms.ToTensor(),
        transforms.Resize(
            (config["image_size"][1], config["image_size"][0])
        ),  # torch does (h, w)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transform = transforms.Compose(transform)
    aspect_ratio = config["image_size"][0] / config["image_size"][1]

    # Load the data
    train_dist_dataset = []
    train_action_dataset = []
    test_dataloaders = {}

    if "context_type" not in config:
        config["context_type"] = "temporal"

    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "negative_mining" not in data_config:
            data_config["negative_mining"] = True
        if "goals_per_obs" not in data_config:
            data_config["goals_per_obs"] = 1
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "waypoint_spacing" not in data_config:
            data_config["waypoint_spacing"] = 1
            
        for data_split_type in ["train", "test"]:
            if data_split_type in data_config:
                for output_type in ["action", "distance", "pairwise"]:
                    
                    if output_type == "pairwise":
                        dataset = PairwiseDistanceDataset(
                            data_folder=data_config["data_folder"],
                            data_split_folder=data_config[data_split_type],
                            dataset_name=dataset_name,
                            transform=transform,
                            aspect_ratio=aspect_ratio,
                            waypoint_spacing=data_config["waypoint_spacing"],
                            min_dist_cat=config["distance"]["min_dist_cat"],
                            max_dist_cat=config["distance"]["max_dist_cat"],
                            close_far_threshold=config["close_far_threshold"],
                            negative_mining=data_config["negative_mining"],
                            context_size=config["context_size"],
                            context_type=config["context_type"],
                            end_slack=data_config["end_slack"],
                        )
                    else:
                        dataset = GNM_Dataset(
                            data_folder=data_config["data_folder"],
                            data_split_folder=data_config[data_split_type],
                            dataset_name=dataset_name,
                            is_action=(output_type == "action"),
                            transform=transform,
                            aspect_ratio=aspect_ratio,
                            waypoint_spacing=data_config["waypoint_spacing"],
                            min_dist_cat=config[output_type]["min_dist_cat"],
                            max_dist_cat=config[output_type]["max_dist_cat"],
                            negative_mining=data_config["negative_mining"],
                            len_traj_pred=config["len_traj_pred"],
                            learn_angle=config["learn_angle"],
                            context_size=config["context_size"],
                            context_type=config["context_type"],
                            end_slack=data_config["end_slack"],
                            goals_per_obs=data_config["goals_per_obs"],
                            normalize=config["normalize"],
                        )
                    if data_split_type == "train":
                        if output_type == "distance":
                            train_dist_dataset.append(dataset)
                            print(
                                f"Loaded {len(dataset)} {dataset_name} training points"
                            )
                        elif output_type == "action":
                            train_action_dataset.append(dataset)
                    else:
                        dataset_type = f"{dataset_name}_{data_split_type}"
                        if dataset_type not in test_dataloaders:
                            test_dataloaders[dataset_type] = {}
                        test_dataloaders[dataset_type][output_type] = dataset

    # combine all the datasets from different robots
    train_dist_dataset = ConcatDataset(train_dist_dataset)
    train_action_dataset = ConcatDataset(train_action_dataset)

    train_dist_loader = DataLoader(
        train_dist_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    train_action_loader = DataLoader(
        train_action_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )

    if "eval_batch_size" not in config:
        config["eval_batch_size"] = config["batch_size"]

    for dataset_type in test_dataloaders:
        for loader_type in test_dataloaders[dataset_type]:
            test_dataloaders[dataset_type][loader_type] = DataLoader(
                test_dataloaders[dataset_type][loader_type],
                batch_size=config["eval_batch_size"],
                shuffle=True,
                num_workers=config["num_workers"],
                drop_last=True,
            )

    # Create the model
    if config["model_type"] == "gnm":
        model = GNM(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model"] == "siamese":
        model = SiameseModel(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obs_encoding_size"],
            config["goal_encoding_size"],
        )
    elif config["model"] == "stacked":
        model = StackedModel(
            config["context_size"],
            config["len_traj_pred"],
            config["learn_angle"],
            config["obsgoal_encoding_size"],
        )
    else:
        raise ValueError(f"Model {config['model']} not supported")

    if len(config["gpu_ids"]) > 1:
        model = nn.DataParallel(model, device_ids=config["gpu_ids"])
    model = model.to(device)
    lr = float(config["lr"])

    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adam":
        optimizer = Adam(model.parameters(), lr=lr)
    elif config["optimizer"] == "adamw":
        optimizer = AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Optimizer {config['optimizer']} not supported")

    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path, map_location=device)
        load_model(model, latest_checkpoint)
        optimizer = get_saved_optimizer(latest_checkpoint, device)
        current_epoch = latest_checkpoint["epoch"] + 1

    torch.autograd.set_detect_anomaly(True)
    if config["train"]:
        train_eval_loop(
            model=model,
            optimizer=optimizer,
            train_dist_loader=train_dist_loader,
            train_action_loader=train_action_loader,
            test_dataloaders=test_dataloaders,
            epochs=config["epochs"],
            device=device,
            project_folder=config["project_folder"],
            normalized=config["normalize"],
            print_log_freq=config["print_log_freq"],
            image_log_freq=config["image_log_freq"],
            num_images_log=config["num_images_log"],
            pairwise_test_freq=config["pairwise_test_freq"],
            current_epoch=current_epoch,
            learn_angle=config["learn_angle"],
            alpha=config["alpha"],
            use_wandb=config["use_wandb"],
        )
    print("FINISHED TRAINING")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mobile Robot Agnostic Learning")

    # project setup
    parser.add_argument(
        "--config",
        "-c",
        default="config/gnm/gnm_public.yaml",
        type=str,
        help="Path to the config file in train_config folder",
    )
    args = parser.parse_args()

    with open("config/defaults.yaml", "r") as f:
        default_config = yaml.safe_load(f)

    config = default_config

    with open(args.config, "r") as f:
        user_config = yaml.safe_load(f)

    config.update(user_config)

    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(
        config[
            "project_folder"
        ],  # should error if dir already exists to avoid overwriting and old project
    )

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"], settings=wandb.Settings(start_method="fork")
        )
        wandb.run.name = config["run_name"]
        # update the wandb args with the training configurations
        if wandb.run:
            wandb.config.update(config)

    print(config)
    main(config)
