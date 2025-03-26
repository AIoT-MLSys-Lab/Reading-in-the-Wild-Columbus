# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from projectaria_tools.core import data_provider
from torch.utils.data import DataLoader


def create_sampled_array(df, num_samples, stride):
    data = df.values
    sampled_arrays = []
    max_start_index = int(data.shape[0] - (num_samples - 1) * stride)

    for start_index in range(max_start_index):
        indices = range(start_index, start_index + stride * num_samples, stride)

        if indices[-1] < data.shape[0]:
            sampled_arrays.append(data[indices])

    result_array = np.array(sampled_arrays)
    return result_array


def polar_to_xyz(polar: np.ndarray, z=None) -> np.ndarray:
    """Converts polar coordinate np arrays to xyz dimensions. Undoing xyz_to_polar
    to compute degree error.

    Turning [arctan(x/z), arctan(y/z)] -> [x, y, z], assuming z is large, since
    we only care about angle
    Args:
        polar: Nx2 polar coordinate vector or Nx3 for 3rd column as Z
        z: scaling factor (z coordinate) to resacle the output results
    returns:
        Nx3 xyz coordiante vector
    """
    # print("polar input", polar)
    # if isinstance(polar, np.ndarray):
    xyz = np.empty((polar.shape[0], 3), dtype=polar.dtype)
    tan = np.tan
    mul = np.multiply

    # Since we are only interested in angle, we can lower prediction error by
    # ignoring the 3rd dimension (which specifies distance)
    # We normalize the vector to fixed z. Assume z is 10 meters, if not given
    z = z if z else 10
    z = z if polar.shape[1] < 3 else polar[:, 2]
    xyz[:, 2] = z
    xyz[:, 0] = mul(tan(polar[:, 0]), z)
    xyz[:, 1] = mul(tan(polar[:, 1]), z)
    return xyz


def polar_to_unit_vector(p_polar):
    p_xyz = polar_to_xyz(np.array(p_polar).reshape(1, -1), 100)
    p_xyz_unitVec = p_xyz / np.linalg.norm(p_xyz)
    return p_xyz_unitVec


def get_eyegaze_point_at_depth(yaw, pitch, depth):
    """
    yaw: left yaw, right yaw, pitch
    depth: depth in meters
    """
    point3d = polar_to_unit_vector(np.array([yaw, pitch])) * depth
    return point3d


def get_calibs(provider):
    device_calib = provider.get_device_calibration()
    # print(device_calib.get_device_subtype())
    label = "camera-rgb"
    # transform_device_sensor = device_calib.get_transform_device_sensor(label)
    # transform_device_cpf = device_calib.get_transform_device_cpf()
    transform_cpf_rgb = device_calib.get_transform_cpf_sensor(label, get_cad_value=False)
    rt = transform_cpf_rgb.inverse()
    rt = rt.to_matrix()
    # returns None if the calibration label does not exist
    cam_calib = device_calib.get_camera_calib(label)
    return cam_calib, rt


def compute_depth_and3rdeye(preds):
    """
    preds: row x column. Columns: left yaw, right yaw, pitch
    """
    ipd = 0.063
    d = ipd / 2
    # if isinstance(preds, np.ndarray):
    tan = np.tan
    atan = np.arctan
    lan = np.linalg.norm
    cos = np.cos
    thirdeye = np.zeros((preds.shape[0], 2))
    intersection_xyz = np.zeros((preds.shape[0], 3))
    intersection_xyz[:, 0] = (
            d
            * (tan(preds[:, 0]) + tan(preds[:, 1]))
            / (tan(preds[:, 1]) - tan(preds[:, 0]))
    )  # x
    intersection_xyz[:, 2] = 2 * d / (tan(preds[:, 1]) - tan(preds[:, 0]))  # z
    intersection_xyz[:, 1] = intersection_xyz[:, 2] * tan(preds[:, 2])  # y
    # print(intersection_xyz)
    r = lan(intersection_xyz, axis=1)

    thirdeye[:, 0] = atan(intersection_xyz[:, 0] / intersection_xyz[:, 2])
    thirdeye[:, 1] = preds[:, 2]
    return r, thirdeye


def get_proj(cpf_3d, cam_calib, RT):
    cpf_3d = np.append(cpf_3d, 1)
    cpf_sensor = np.dot(RT, cpf_3d)
    proj = cam_calib.project(cpf_sensor[:3])
    return proj


def get_projections_et(
        latest_et_df: pd.DataFrame,
        cam_calib,
        cpf_to_rgb_T,
):
    cpf_origin = np.array([0, 0, 0])
    projections = []
    directions = []

    for row in latest_et_df.iterrows():
        V_cpf = get_eyegaze_point_at_depth(
            row[1]["yaw_rads_cpf"],
            row[1]["pitch_rads_cpf"],
            row[1]["depth_m"],
        )
        direction = V_cpf.flatten() - cpf_origin
        V_proj = get_proj(V_cpf, cam_calib, cpf_to_rgb_T)

        if V_proj is None:
            if len(projections) > 0:
                V_proj = projections[-1]
                direction = directions[-1]
            else:
                V_proj = np.array([704, 704], dtype=np.float64)

        projections.append((V_proj[0], V_proj[1]))
        directions.append(direction.flatten())

    return projections, directions


def project_gaze_vrs(gaze_path, vrs_path=None):
    """
    Input:
    gaze_path: path to either general or personalized eye gaze .csv file
    vrs_path: path to .vrs file

    Output:
    A dataframe with
    cols = ["tracking_timestamp_us", "projected_point_2d_x", "projected_point_2d_y", "transformed_gaze_x", "transformed_gaze_y", "transformed_gaze_z", "depth_m"]
    """
    gaze = pd.read_csv(gaze_path, engine='python')
    provider = data_provider.create_vrs_data_provider(vrs_path)
    cam_calib, cpf_to_rgb_T = get_calibs(provider)
    cols = ["left_yaw_rads_cpf", "right_yaw_rads_cpf", "pitch_rads_cpf"]
    _, third_eye = compute_depth_and3rdeye(gaze.loc[:, cols].to_numpy())
    gaze["yaw_rads_cpf"] = third_eye[:, 0]
    gaze["pitch_rads_cpf"] = third_eye[:, 1]
    projections, directions = get_projections_et(latest_et_df=gaze, cam_calib=cam_calib, cpf_to_rgb_T=cpf_to_rgb_T)
    projections = np.array(projections)
    directions = np.array(directions) / np.linalg.norm(directions, axis=-1, keepdims=True)
    data = {
        "tracking_timestamp_us": gaze["tracking_timestamp_us"],
        "projected_point_2d_x": projections[:, 1],
        "projected_point_2d_y": projections[:, 0],
        "transformed_gaze_x": directions[:, 0],
        "transformed_gaze_y": directions[:, 1],
        "transformed_gaze_z": directions[:, 2],
        "depth_m": gaze["depth_m"]
    }
    # Create a DataFrame from the dictionary
    out = pd.DataFrame(data)
    return out

def find_vrs_files(folder_path):
    """Find all .vrs files recursively in the given folder."""
    vrs_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.vrs'):
                vrs_files.append(os.path.join(root, file))
    return vrs_files

def get_vrs_file_list(cfg: DictConfig, orig_cwd: Path):
    if cfg.mode.lower() == "single":
        file_path = orig_cwd / cfg.root_dir / cfg.input_filename
        return [file_path] if file_path.exists() else []
    else:
        root_dir = orig_cwd / cfg.root_dir
        return list(root_dir.glob("*.vrs"))[:4]