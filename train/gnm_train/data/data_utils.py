import numpy as np
import os
from PIL import Image
from typing import Any, Iterable

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F

VISUALIZATION_IMAGE_SIZE = (120, 160)
IMAGE_ASPECT_RATIO = (
    4 / 3
)  # all images are centered cropped to a 4:3 aspect ratio in training


def get_image_path(data_folder: str, f: str, time: int):
    return os.path.join(data_folder, f, f"{str(time)}.jpg")


def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=object,  # get rid of warning
    )


def to_ego_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)


def calculate_deltas(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate deltas between waypoints

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: deltas
    """
    num_params = waypoints.shape[1]
    origin = torch.zeros(1, num_params)
    prev_waypoints = torch.concat((origin, waypoints[:-1]), axis=0)
    deltas = waypoints - prev_waypoints
    if num_params > 2:
        return calculate_sin_cos(deltas)
    return deltas


def calculate_sin_cos(waypoints: torch.Tensor) -> torch.Tensor:
    """
    Calculate sin and cos of the angle

    Args:
        waypoints (torch.Tensor): waypoints
    Returns:
        torch.Tensor: waypoints with sin and cos of the angle
    """
    assert waypoints.shape[1] == 3
    angle_repr = torch.zeros_like(waypoints[:, :2])
    angle_repr[:, 0] = torch.cos(waypoints[:, 2])
    angle_repr[:, 1] = torch.sin(waypoints[:, 2])
    return torch.concat((waypoints[:, :2], angle_repr), axis=1)


def img_path_to_data(path: str, transform: transforms, aspect_ratio: float = IMAGE_ASPECT_RATIO) -> torch.Tensor:
    """
    Load an image from a path and transform it
    Args:
        path (str): path to the image
        transform (transforms): transform to apply to the image
        aspect_ratio (float): aspect ratio to crop the image to
    Returns:
        torch.Tensor: transformed image
    """
    img = Image.open(path)
    w, h = img.size
    img = TF.center_crop(
        img, (h, int(h * aspect_ratio))
    )  # crop to the right ratio
    viz_img = TF.resize(img, VISUALIZATION_IMAGE_SIZE)
    viz_img = TF.to_tensor(viz_img)
    transf_img = transform(img)
    return viz_img, transf_img


class RandomizedClassBalancer:
    def __init__(self, classes: Iterable) -> None:
        """
        A class balancer that will sample classes randomly, but will prioritize classes that have been sampled less

        Args:
            classes (Iterable): The classes to balance
        """
        self.counts = {}
        for c in classes:
            self.counts[c] = 0

    def sample(self, class_filter_func=None) -> Any:
        """
        Sample the softmax of the negative logits to prioritize classes that have been sampled less

        Args:
            class_filter_func (Callable, optional): A function that takes in a class and returns a boolean. Defaults to None.
        """
        if class_filter_func is None:
            keys = list(self.counts.keys())
        else:
            keys = [k for k in self.counts.keys() if class_filter_func(k)]
        if len(keys) == 0:
            return None  # no valid classes to sample
        values = [-(self.counts[k] - min(self.counts.values())) for k in keys]
        p = F.softmax(torch.Tensor(values), dim=0).detach().cpu().numpy()
        class_index = np.random.choice(list(range(len(keys))), p=p)
        class_choice = keys[class_index]
        self.counts[class_choice] += 1
        return class_choice

    def __str__(self) -> str:
        string = ""
        for c in self.counts:
            string += f"{c}: {self.counts[c]}\n"
        return string
