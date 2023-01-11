import numpy as np
import os
import pickle
import tqdm

from torchvision import transforms
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from gnm_train.data.data_utils import (
    img_path_to_data,
    get_image_path,
    RandomizedClassBalancer,
)


class PairwiseDistanceDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        transform: transforms,
        aspect_ratio: float,
        waypoint_spacing: int,
        min_dist_cat: int,
        max_dist_cat: int,
        close_far_threshold: int,
        negative_mining: bool,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
    ):
        """
        A dataset that contains a single observation and two subgoals. The task is to predict which subgoal is closer to the observation. This dataset is only used for evaluation.

        Args:
            data_folder (string): Directory with all the image data
            data_split_folder (string): Directory with filepaths.txt, a list of all trajectory names seperated by a newline in the dataset split
            dataset_name (string): Name of the dataset [recon, go_stanford, scand, tartandrive, etc.]
            is_action (bool): Whether to use the action dataset or the distance dataset
            transform (transforms): Transforms to apply to the image data
            aspect_ratio (float): Aspect ratio of the images (w/h)
            waypoint_spacing (int): Spacing between waypoints
            min_dist_cat (int): Minimum distance category to use
            max_dist_cat (int): Maximum distance category to use
            close_far_threshold (int): Threshold for close and far
            negative_mining (bool): Whether to use negative mining from the RECON paper (https://arxiv.org/abs/2104.05859)
            context_size (int): Number of previous observations to use as context
            context_type (str): Whether to use temporal, randomized, or randomized temporal context
            end_slack (int): Number of timesteps to ignore at the end of the trajectory
        """
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        # filepath to the list to all the names of the trajectories in the dataset split
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.transform = transform
        self.aspect_ratio = aspect_ratio
        self.waypoint_spacing = waypoint_spacing
        self.distance_categories = list(
            range(min_dist_cat, max_dist_cat + 1, waypoint_spacing)
        )
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.negative_mining = negative_mining and len(self.distance_categories) > 1
        if self.negative_mining:
            self.distance_categories.append(-1)

        threshold_index = -1
        self.close_far_threshold = close_far_threshold
        while (
            threshold_index + 1 < len(self.distance_categories)
            and self.distance_categories[threshold_index + 1] <= close_far_threshold
        ):
            threshold_index += 1

        self.pairwise_categories = []
        for i in range(threshold_index, len(self.distance_categories)):
            for j in range(threshold_index):
                self.pairwise_categories.append(
                    (self.distance_categories[j], self.distance_categories[i])
                )

        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self._gen_index_to_data()

    def _gen_index_to_data(self):
        self.index_to_data = []
        label_balancer = RandomizedClassBalancer(self.pairwise_categories)
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"pairwise_waypoint_spacing_{self.waypoint_spacing}_{self.min_dist_cat}_{self.max_dist_cat}_close_far_threshold_{self.close_far_threshold}_negative_mining_{int(self.negative_mining)}_context_size_{self.context_size}_end_slack_{self.end_slack}.pkl",
        )
        try:
            with open(index_to_data_path, "rb") as f1:
                self.index_to_data = pickle.load(f1)
        except:
            print(
                f"Sampling subgoals for each observation in the {self.dataset_name} pairwise distance dataset..."
            )
            print(
                "This will take a while, but it will only be done once for each configuration per dataset."
            )
            for i in tqdm.tqdm(range(len(self.traj_names))):
                f_close = self.traj_names[i]
                with open(
                    os.path.join(self.data_folder, f_close, "traj_data.pkl"), "rb"
                ) as f3:
                    close_traj_data = pickle.load(f3)
                traj_len = len(close_traj_data["position"])
                for curr_time in range(
                    self.context_size * self.waypoint_spacing,
                    traj_len - self.end_slack,
                ):
                    max_len = min(
                        int(self.max_dist_cat * self.waypoint_spacing),
                        traj_len - curr_time - 1,
                    )
                    filter_func = (
                        lambda tup: max(tup) * self.waypoint_spacing <= max_len
                    )
                    choice = label_balancer.sample(filter_func)
                    if choice is None:
                        break
                    close_len_to_goal, far_len_to_goal = choice

                    if far_len_to_goal == -1:  # negative mining
                        new = np.random.randint(1, len(self.traj_names))
                        f_rand = self.traj_names[(i + new) % len(self.traj_names)]
                        with open(
                            os.path.join(
                                os.path.join(self.data_folder, f_rand), "traj_data.pkl"
                            ),
                            "rb",
                        ) as f4:
                            rand_traj_data = pickle.load(f4)
                        rand_traj_len = len(rand_traj_data["position"])
                        far_time = np.random.randint(rand_traj_len)
                        f_far = f_rand
                    else:
                        far_time = curr_time + int(
                            far_len_to_goal * self.waypoint_spacing
                        )
                        f_far = f_close

                    close_time = curr_time + int(
                        close_len_to_goal * self.waypoint_spacing
                    )
                    assert (
                        close_time < traj_len
                    ), f"{curr_time}, {close_len_to_goal}, {traj_len}"
                    if f_close != f_far:
                        assert far_time < rand_traj_len, f"{far_time}, {rand_traj_len}"
                    else:
                        assert (
                            far_time < traj_len
                        ), f"{curr_time}, {far_len_to_goal}, {traj_len}"
                    self.index_to_data += [
                        (f_close, f_far, curr_time, close_time, far_time)
                    ]
            with open(index_to_data_path, "wb") as f2:
                pickle.dump(self.index_to_data, f2)

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> tuple:
        """
        Args:
            i (int): index to ith datapoint
        Returns:
            obs_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the observation for visualization
            close_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the closer subgoal out of the 2 sampled for visualization
            far_image (torch.Tensor): tensor of shape [3, H, W] containing the image of the farther subgoal out of the 2 sampled for visualization
            transf_obs_images (torch.Tensor): tensor of shape [(context_size) * 3, H, W] containing the images of the context and the observation after transformation for training
            transf_close_goal_images (torch.Tensor): tensor of shape [3, H, W] containing the images of the closer subgoal after transformation for training
            transf_far_goal_images (torch.Tensor): tensor of shape [3, H, W] containing the images of the farther subgoal after transformation for training
            close_dist_label (torch.Tensor): tensor of shape [1] containing the distance label of the closer subgoal
            far_dist_label (torch.Tensor): tensor of shape [1] containing the label for the farther subgoal
        """
        f_close, f_far, curr_time, close_time, far_time = self.index_to_data[i]
        assert curr_time <= close_time
        assert f_close != f_far or close_time < far_time

        transf_obs_images = []
        context = []
        if self.context_type == "randomized":
            # sample self.context_size random times from interval [0, curr_time) with no replacement
            context_times = np.random.choice(
                list(range(curr_time)), self.context_size, replace=False
            )
            context_times.append(curr_time)
            context = [(f_close, t) for t in context_times]
        elif self.context_type == "randomized_temporal":
            f_rand_close, _, rand_curr_time, _, _ = self.index_to_data[
                np.random.randint(0, len(self))
            ]
            context_times = list(
                range(
                    rand_curr_time + -self.context_size * self.waypoint_spacing,
                    rand_curr_time,
                    self.waypoint_spacing,
                )
            )
            context = [(f_rand_close, t) for t in context_times]
            context.append((f_close, curr_time))
        elif self.context_type == "temporal":
            context_times = list(
                range(
                    curr_time + -self.context_size * self.waypoint_spacing,
                    curr_time + 1,
                    self.waypoint_spacing,
                )
            )
            context = [(f_close, t) for t in context_times]
        else:
            raise ValueError(f"Invalid type {self.context_type}")
        for f, t in context:
            obs_image_path = get_image_path(self.data_folder, f, t)
            obs_image, transf_obs_image = img_path_to_data(
                obs_image_path,
                self.transform,
                self.aspect_ratio,
            )
            transf_obs_images.append(transf_obs_image)
        transf_obs_image = torch.cat(transf_obs_images, dim=0)

        close_image_path = get_image_path(self.data_folder, f_close, close_time)
        close_image, transf_close_image = img_path_to_data(
            close_image_path,
            self.transform,
            self.aspect_ratio,
        )

        far_image_path = get_image_path(self.data_folder, f_far, far_time)
        far_image, transf_far_image = img_path_to_data(
            far_image_path,
            self.transform,
            self.aspect_ratio,
        )

        close_dist_label = torch.FloatTensor(
            [(close_time - curr_time) / self.waypoint_spacing]
        )
        if f_close == f_far:
            far_dist_label = torch.FloatTensor([far_time - curr_time])
        else:
            far_dist_label = torch.FloatTensor([self.max_dist_cat])

        return (
            obs_image,
            close_image,
            far_image,
            transf_obs_image,
            transf_close_image,
            transf_far_image,
            close_dist_label,
            far_dist_label,
        )
