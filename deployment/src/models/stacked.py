import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple
from .modified_mobilenetv2 import MobileNetEncoder
from .base_model import BaseModel


class StackedModel(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obsgoal_encoding_size: Optional[int] = 2048,
    ) -> None:
        """
        Stacked architecture main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            obsgoal_encoding_size (int): size of the encoding of the obs+goal images
            learn_angle (bool): whether to predict the yaw of the robot
        """
        super(StackedModel, self).__init__(context_size, len_traj_pred, learn_angle)
        mobilenet = MobileNetEncoder(num_images=2 + self.context_size)
        self.obsgoal_mobilenet = mobilenet.features
        self.obsgoal_encoding_size = obsgoal_encoding_size
        self.compress_obsgoal = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obsgoal_encoding_size),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(self.obsgoal_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor):
        obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        obs_goal_encoding = self.obsgoal_mobilenet(obs_goal_input)
        obs_goal_encoding = self.flatten(obs_goal_encoding)
        obs_goal_encoding = self.compress_obsgoal(obs_goal_encoding)

        z = self.linear_layers(obs_goal_encoding)
        dist_pred = self.dist_predictor(z)
        action_pred = self.action_predictor(z)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred
