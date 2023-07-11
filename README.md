# GNM: A General Navigation Model to Drive Any Robot
#### Dhruv Shah*, Ajay Sridhar*, Arjun Bhorkar, Noriaki Hirose, Sergey Levine

_Berkeley AI Research_


[Project Page](https://sites.google.com/view/drive-any-robot) | [arXiV](https://arxiv.org/abs/2210.03370) | [Summary Video](https://www.youtube.com/watch?v=ICeD6iOglKc)

---
## Overview
This repository contains code for training a GNM with your own data, pre-trained model checkpoints, as well as example code to deploy it on a TurtleBot2/LoCoBot robot.

- `./train/train.py`: training script to train or fine-tune the GNM model on your custom data.
- `./train/process_*.py`: scripts to process rosbags or other formats of robot trajectories into training data.
- `./deployment/src/record_bag.sh`: script to collect a demo trajectory in the target environment on the robot. This trajectory is subsampled to generate a topological graph of the environment.
- `./deployment/src/navigate.sh`: script that deploys a trained GNM model on the robot to navigate to a desired goal in the generated topological graph. Please see relevant sections below for configuration settings.

## Train

This subfolder contains code for processing datasets and training a GNM from your own data.

### Pre-requisites

The codebase assumes access to a workstation running Ubuntu (tested on 18.04 and 20.04), Python 3.7+, and a GPU with CUDA 10+. It also assumes access to conda, but you can modify it to work with other virtual environment packages, or a native setup.
### Setup
Run the commands below inside the `drive-any-robot/` (topmost) directory:
1. Set up the conda environment:
    ```bash
    conda env create -f train/environment.yml
    ```
2. Source the conda environment:
    ```
    conda activate gnm_train
    ```
3. Install the gnm_train packages:
    ```bash
    pip install -e train/
    ```

### Data-Wrangling
In the [GNM paper](https://sites.google.com/view/drive-any-robot), we train on a combination of publicly available and unreleased datasets. Below is a list of publicly available datasets used for training; please contact the respective authors for access to the unreleased data.
- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [TartanDrive](https://github.com/castacks/tartan_drive)
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links)
- [GoStanford2 (Modified)](https://drive.google.com/drive/folders/1xrNvMl5q92oWed99noOt_UhqQnceJYV0?usp=share_link)

We recommend you to download these (and any other datasets you may want to train on) and run the processing steps below.

#### Data Processing 

We provide some sample scripts to process these datasets, either directly from a rosbag or from a custom format like HDF5s:
1. Run `process_bags.py` with the relevant args, or `process_recon.py` for processing RECON HDF5s. You can also manually add your own dataset by following our structure below (if you are adding a custom dataset, please checkout the [Custom Datasets](#custom-datasets) section).
2. Run `data_split.py` on your dataset folder with the relevant args.

After step 1 of data processing, the processed dataset should have the following structure:

```
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_2.jpg
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
    	├── 0.jpg
    	├── 1.jpg
    	├── ...
        ├── T_N.jpg
        └── traj_data.pkl
```  

Each `*.jpg` file contains an forward-facing RGB observation from the robot, and they are temporally labeled. The `traj_data.pkl` file is the odometry data for the trajectory. It’s a pickled dictionary with the keys:
- `"position"`: An np.ndarray [T, 2] of the xy-coordinates of the robot at each image observation.
- `"yaw"`: An np.ndarray [T,] of the yaws of the robot at each image observation.


After step 2 of data processing, the processed data-split should the following structure inside `gnm_release/train/gnm_train/data/data_splits/`:

```
├── <dataset_name>
│   ├── train
|   |   └── traj_names.txt
└── └── test
        └── traj_names.txt 
``` 

### Training your GNM
Run this inside the `gnm_release/train` directory:
```bash
python train.py -c <path_of_train_config_file>
```
The premade config yaml files are in the `train/config` directory. 

#### Custom Config Files
You can use one of the premade yaml files as a starting point and change the values as you need. `config/gnm/gnm_public.yaml` is good choice since it has commented arguments. `config/defaults.yaml` contains the default config values (don't directly train with this config file since it does not specify any datasets for training).

#### Custom Datasets
Make sure your dataset and data-split directory follows the structures provided in the [Data Processing](#data-processing) section. Locate `train/gnm_train/data/data_config.yaml` and append the following:

```
<dataset_name>:
    metric_waypoints_distance: <average_distance_in_meters_between_waypoints_in_the_dataset>
```

Locate your training config file and add the following text under the `datasets` argument (feel free to change the values of `end_slack`, `goals_per_obs`, and `negative_mining`):
```
<dataset_name>:
    data_folder: <path_to_the_dataset>
    train: data/data_splits/<dataset_name>/train/ 
    test: data/data_splits/<dataset_name>/test/ 
    end_slack: 0 # how many timesteps to cut off from the end of each trajectory  (in case many trajectories end in collisions)
    goals_per_obs: 1 # how many goals are sampled per observation
    negative_mining: True # negative mining from the ViNG paper (Shah et al.)
```

#### Training your GNM from a checkpoint
Instead of training from scratch, you can also load an existing checkpoint from the published results.
Add `load_run: <project_name>/<log_run_name>`to your .yaml config file in `gnm_release/train/config/`. The `*.pth` of the file you are loading to be saved in this file structure and renamed to “latest”: `gnm_release/train/logs/<project_name>/<log_run_name>/latest.pth`. This makes it easy to train from the checkpoint of a previous run since logs are saved this way by default. Note: if you are loading a checkpoint from a previous run, check for the name the run in the `gnm_release/train/logs/<project_name>/`, since the code appends a string of the date to each run_name specified in the config yaml file of the run to avoid duplicate run names. 


If you want to use our checkpoints, you can download the `*.pth` files from [this link](https://drive.google.com/drive/folders/1np7D0Ak7x10IoQn9h0qxn8eoxJQiw8Dr?usp=share_link).


## Deployment
This subfolder contains code to load a pre-trained GNM and deploy it on the open-source [LoCoBot indoor robot platform](http://www.locobot.org/). It can be easily adapted to be run on alternate robots, and researchers have been able to independently deploy it on the following robots – Clearpath Jackal, DJI Tello, Unitree A1, TurtleBot2, Vizbot – and in simulated environments like CARLA.

### LoCoBot Setup

This software was tested on a LoCoBot running Ubuntu 16.04 (now legacy, but should be forward compatible).


#### Software Installation (in this order)
1. ROS: [ros-kinetic](https://wiki.ros.org/kinetic/Installation/Ubuntu)
2. ROS packages: 
    ```bash
    sudo apt-get install ros-kinetic-usb-cam ros-kinetic-joy
    ```
3. [PyRobot](https://pyrobot.org/docs/software)
4. Conda 
    - Install anaconda/miniconda/etc. for managing environments
    - Make conda env with environment.yml (run this inside the `gnm_release/` directory)
        ```bash
        conda env create -f deployment/environment.yml
        ```
    - Source env 
        ```bash
        conda activate gnm_deployment
        ```
    - (Recommended) add to `~/.bashrc`: 
        ```bash
        echo “conda activate gnm_deployment” >> ~/.bashrc 
        ```
5. Install the `gnm_train` packages (run this inside the `gnm_release/` directory):
    ```bash
    pip install -e train/
    ```
6. (Recommended) Install [tmux](https://github.com/tmux/tmux/wiki/Installing) if not present.
    Many of the bash scripts rely on tmux to launch multiple screens with different commands. This will be useful for debugging because you can see the output of each screen.

#### Hardware Requirements
- LoCoBot: http://locobot.org
- A wide-angle RGB camera: [Example](https://www.amazon.com/ELP-170degree-Fisheye-640x480-Resolution/dp/B00VTHD17W). The `gnm_locobot.launch` file uses camera parameters that work with cameras like the ELP fisheye wide angle, feel free to modify to your own. Adjust the camera parameters in `gnm_release/deployment/config/camera.yaml` your camera accordingly (used for visualization).
- [Joystick](https://www.amazon.com/Logitech-Wireless-Nano-Receiver-Controller-Vibration/dp/B0041RR0TW)/[keyboard teleop](http://wiki.ros.org/teleop_twist_keyboard) that works with Linux. Add the index mapping for the _deadman_switch_ on the joystick to the `gnm_release/deployment/config/joystick.yaml`. You can find the mapping from buttons to indices for common joysticks in the [wiki](https://wiki.ros.org/joy). 


### Loading the model weights

Save the model weights *.pth file in `gnm_release/deployment/model_weights` folder. Our models weights are in [this link](https://drive.google.com/drive/folders/1np7D0Ak7x10IoQn9h0qxn8eoxJQiw8Dr?usp=share_link).

### Collecting a Topological Map

_Make sure to run these scripts inside the `gnm_release/deployment/src/` directory._


This section discusses a simple way to create a topological map of the target environment for deployment. For simplicity, we will use the robot in “path-following” mode, i.e. given a single trajectory in an environment, the task is to follow the same trajectory to the goal. The environment may have new/dynamic obstacles, lighting variations etc.

#### Record the rosbag: 
```bash
./record_bag.sh <bag_name>
```

Run this command to teleoperate the robot with the joystick and camera. This command opens up three windows 
1. `roslaunch gnm_locobot.launch`: This launch file opens the `usb_cam` node for the camera, the joy node for the joystick, and several nodes for the robot’s mobile base).
2. `python joy_teleop.py`: This python script starts a node that reads inputs from the joy topic and outputs them on topics that teleoperate the robot’s base.
3. `rosbag record /usb_cam/image_raw -o <bag_name>`: This command isn’t run immediately (you have to click Enter). It will be run in the gnm_release/deployment/topomaps/bags directory, where we recommend you store your rosbags.

Once you are ready to record the bag, run the `rosbag record` script and teleoperate the robot on the map you want the robot to follow. When you are finished with recording the path, kill the `rosbag record` command, and then kill the tmux session.

#### Make the topological map: 
```bash
./create_topomap.sh <topomap_name> <bag_filename>
```

This command opens up 3 windows:
1. `roscore`
2. `python create_topomap.py —dt 1 —dir <topomap_dir>`: This command creates a directory in `/gmn_release/deployment/topomaps/images` and saves an image as a node in the map every second the bag is played.
3. `rosbag play -r 5 <bag_filename>`: This command plays the rosbag at x5 speed, so the python script is actually recording nodes 5 seconds apart. The `<bag_filename>` should be the entire bag name with the .bag extension. You can change this value in the `make_topomap.sh` file. The command does not run until you hit Enter, which you should only do once the python script gives its waiting message. Once you play the bag, move to the screen where the python script is running so you can kill it when the rosbag stops playing.

When the bag stops playing, kill the tmux session.


### Running the model 
_Make sure to run this script inside the `gnm_release/deployment/src/` directory._

```bash
./navigate.sh “--model <model_name> --dir <topomap_dir>”
```

To deploy one of the models from the published results, we are releasing model checkpoints that you can download from [this link](https://drive.google.com/drive/folders/1np7D0Ak7x10IoQn9h0qxn8eoxJQiw8Dr?usp=share_link).


The `<model_name>` is the name of the model in the `gnm_release/deployment/config/models.yaml` file. In this file, you specify these parameters of the model for each model (defaults used):
- `path` (str, default: large_gnm.pth): path of the *.pth file in `gnm_release/deployment/model_weights/`
- `image_size` (List[int, int], default: [85, 64]): [width, height] of the input images
- `model_type` (str, default: gnm): one of these [gnm, stacked, siamese]
- `context` (int, default: 5): context length
- `len_traj_pred` (int, default: 5): number of future waypoints to predict 
- `normalize` (bool, default: True): whether or not to normalize the waypoints
- `learn_angle` (bool, default: True): whether or not to learn the yaw for each waypoint
- `obs_encoding_size` (int, default: 1024): observation encoding dimension (only for the GNM and the siamese model)
- `goal_encoding_size` (int, default: 1024): goal encoding dimension (only for the GNM and the siamese model)
- `obsgoal_encoding_size` (int, default: 2048): observation + goal encoding dimension (only for the stacked model)

Make sure these configurations match what you used to train the model. The configurations for the models we provided the weights for are provided in yaml file for your reference.

The `<topomap_dir>` is the name of the directory in `gmn_release/deployment/topomaps/images` that has the images corresponding to the nodes in the topological map. The images are ordered by name from 0 to N.

This command opens up 4 windows:

1. `roslaunch gnm_locobot.launch`: This launch file opens the usb_cam node for the camera, the joy node for the joystick, and several nodes for the robot’s mobile base).
2. `python navigate.py --model <model_name> —dir <topomap_dir>`: This python script starts a node that reads in image observations from the `/usb_cam/image_raw` topic, inputs the observations and the map into the model, and publishes actions to the `/waypoint` topic.
3. `python joy_teleop.py`: This python script starts a node that reads inputs from the joy topic and outputs them on topics that teleoperate the robot’s base.
4. `python pd_controller.py`: This python script starts a node that reads messages from the `/waypoint` topic (waypoints from the model) and outputs velocities to navigate the robot’s base.

When the robot is finishing navigating, kill the `pd_controller.py` script, and then kill the tmux session. If you want to take control of the robot while it is navigating, the `joy_teleop.py` script allows you to do so with the joystick.

### Adapting this code to different robots

We hope that this codebase is general enough to allow you to deploy it to your favorite ROS-based robots. You can change the robot configuration parameters in `gnm_release/deployment/config/robot.yaml`, like the max angular and linear velocities of the robot and the topics to publish to teleop and control the robot. Please feel free to create a Github Issue or reach out to the authors at shah@cs.berkeley.edu.


## Citing
```
@inproceedings{shah2022gnm,
   author    = {Dhruv Shah and Ajay Sridhar and Arjun Bhorkar and Noriaki Hirose and Sergey Levine},
   title     = {{GNM: A General Navigation Model to Drive Any Robot}},
   booktitle = {arXiV},
   year      = {2022},
   url      = {https://arxiv.org/abs/2210.03370}
}
```
