# Reading in the Wild - Columbus Dataset 

# Overview
To enable egocentric contextual AI in smart glasses, it's essential to record user interactions—including during reading. In this paper, we introduce **reading recognition**, a task to determine **when** a user is reading. We present the first large-scale multimodal **Reading in the Wild** dataset, featuring 100 hours of diverse reading and non-reading videos. Our approach leverages three modalities—egocentric RGB, eye gaze, and head pose—using a flexible transformer model that can combine these cues. We demonstrate that these modalities are both relevant and complementary, and we explore efficient encoding strategies. Moreover, our dataset supports classifying different types of reading, extending previous constrained studies to more realistic scenarios.

<p align="center">
<img src="media/ritw_columbus_teaser.gif" alt="comparison" width="70%">
</p>

## Dataset Statistics
The Columbus subset contains around 20 hours of data from 31 subjects containing reading and non-reading activities 
in indoor scenarios. It features 4 different languages and particularly focuses on evaluating edge cases.
<p align="center">
<img src="media/table1.png" alt="comparison" width="100%">
</p>

## Scenarios
The data covers a range of scenarios from paragraphs which have long continuous text to very short texts containing 
few words like on signage and posters. The reading materials also contain non-textual content which can be read like 
illustrative diagrams and charts as well as non-readable material like video and images. The experiments are 
conducted in 3 different mediums as well covering digital, print and objects.
<p align="center">
<img src="media/scenarios.png" alt="comparison" width="70%">
</p>

## Key Advantages over Existing Works
The primary advantage of `RITW` is its curation of natural reading data collected using the Aria Smart glasses. The 
reading data containing gaze at 60Hz which is significantly higher than other egocentric datasets and has 100 hrs of 
real data which is significantly higher than existing reading recognition datasets.
<p align="center">
<img src="media/table2.png" alt="comparison" width="60%">
</p>

## Base Model
A simple baseline model (`v1_default`) trained on the training data of the `Seattle` subset for benchmarking can be 
found  [here](https://github.com/facebookresearch/reading_in_the_wild). 
The model uses a 64x64 RGB crop from the RGB camera of the glasses centered on the wearer's eye gaze, 3D gaze 
velocities sampled at 60Hz spanning 2s from the eye tracking cameras and 3D head orientation and velocity sampled at 
60Hz spanning 2s from the IMU sensors. The model can selectively work with any combination of these modalities. It 
uses a 3-layer convolutional encoder model for each modality and a 1-layer transformer head to fuse the encoder 
outputs to output binary predictions.
<p align="center">
<img src="media/arch.png" alt="comparison" width="60%">
</p>

Additionally, the following alternate variants are available [here](https://github.com/facebookresearch/reading_in_the_wild):
+ `v1_1s`: uses a shorter 1s span for Gaze data
+ `v1_15Hz`: uses a lower 15Hz sampling frequency for Gaze data
+ `v1_large`: uses a larger RGB crop size of 128x128
+ `v1_medium`: outputs categorical predictions for medium (`no-read`, 'print`, 'digital` and `objects`).
+ `v1_mode`: outputs categorical predictions for reading modes(`no-read`, 'walk`, 'out-loud`, `engaged`, `scan`, 
`write/type` and `skim`).

Download and put the models in ```models/``` folder.
# Getting Started
## Setup
Use conda to create a new environment and install the required packages. The codebase has been tested with Python 3.12 and PyTorch 2.4.
```commandline
conda env create -f environment.yml
```
Once the environment is created, activate it:
```commandline
conda activate ritw-osu
```

## Download
```TODO```

## Prediction

### Modes
The codebase provides scripts for running predictions on the **Reading in The Wild** dataset using our flexible transformer model. The prediction pipeline supports two modes:

- **Single File Prediction:** Run inference on an individual VRS file to test and visualize the model’s performance on a specific sample.
- **Folder Prediction:** Evaluate the model on a folder containing multiple VRS files. This mode leverages parallel processing to efficiently run inference on all files and saves the results to CSV.

### Inference
The inference pipeline is configurable via a config file. An example config is shown below: 
```yaml
# Example: ../config/config.yaml
# conf/config.yaml
start_time: 0.0
snippet_gap: 0.01667  # roughly 1/60 seconds
mode: "folder"
modalities:
  - "gaze"
  - "imu"
  - "rgb"
  - "gaze+rgb"
  - "gaze+imu"
  - "imu+rgb"
  - "gaze+imu+rgb"
output_save_path: "output/"
root_dir: "/path/to/ritw/dataset/"
model_name:
  - "v1_default"
  - "v0"
num_workers: 4  # adjust based on available CPU cores
```
The config allows selecting modalities and models to infer on. Note that, to add new models, put the model in the `ritw-osu/models` directory and add the model name to the config file.

To run prediction, create a config.yaml file (also see `predict.yaml` for reference) and save to `ritw-osu/config`. Then use the following command:
```bash
python -m ritw.predict --config-name config.yaml
```
The command runs each file and model combinations in separate processes. The output is saved in the form of `csv` files in directory `<output_save_path>/<model_name>`. 

## Evaluation
The evaluation module allows you to assess the performance of the reading recognition system using prediction results stored from the inference step. This module leverages metadata from the recordings, applies configurable filters to focus on a specific subset of the dataset, computes various classification metrics for each modality, and outputs a summary table in Markdown format.

Below is an example configuration file (`conf/config.yaml`) for the evaluation module:

```yaml
# Example: ../config/config.yaml
metadata_file: "data/metadata.csv"
result_dir: "output/v1_default"
target_recall: 0.9
metrics:
  - "F1"
  - "Acc"
  - "P@R=0.9"
  - "T@R=0.9"
  - "Acc@R=0.9"
  - "F1@R=0.9"
  - "AUC"
modalities:
  - "gaze"
  - "rgb"
  - "imu"
  - "gaze+imu"
  - "imu+rgb"
  - "gaze+rgb"
  - "gaze+imu+rgb"
filters:
  ContainsNonText:
    - "images"
  ShortTextOrPara:
    - "paragraphs"
  Medium:
    - "digital"
  Platform:
    - "laptop"
```

After setting up your environment and ensuring that the metadata and prediction CSV files are available, run the evaluation module from the project’s root directory:
```bash
python -m ritw.evaluate --config-name config.yaml
```
