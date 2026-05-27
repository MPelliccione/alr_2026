"""
Utilities for post-processing and exporting simulation results.

This module provides functions to export JAX and NumPy arrays representing
nodal solutions or arbitrary data fields into VTK files format for
visualization in tools like ParaView.
"""
import os
from pathlib import Path

import numpy as np
from jax import Array

from utils import generate_box_mesh

PROJ_ROOT = Path(__file__).parent.parent

def save_sol_to_vtk(sol: Array | np.ndarray, save_file_pth: Path | str, save_file_dir: Path | str = "paraview_out") -> None:
    """
    Converts a jax/numpy solution array to a vtk file viewable in paraview.

    Saves the vtk file to disk containing displacement magnitude and
    individual components.

    Parameters
    ----------
    sol : Array or np.ndarray
        Solution array, usually a function (displacement) evaluated at mesh nodes.
    save_file_pth : Path or str
        Filename or subpath for the VTK file.
    save_file_dir : Path or str, optional
        Base directory for storing ParaView output. Defaults to proj_root/paraview_out.
    """
    if save_file_dir == "paraview_out":
        save_file_dir = PROJ_ROOT / save_file_dir
    os.makedirs(save_file_dir, exist_ok=True)
    if isinstance(sol, Array):
        sol = np.array(sol, dtype=np.float64)
    save_file_path = Path(save_file_dir) / save_file_pth
    mesh = generate_box_mesh()
    assert mesh.points.shape[0] == sol.shape[0]
    u_magn = np.linalg.norm(sol, axis=1)
    point_infos = [
        ("U_magn", u_magn),
        ("Ux", sol[:, 0]),
        ("Uy", sol[:, 1]),
        ("Uz", sol[:, 2]),
    ]
    for name, point_info in point_infos:
        mesh.point_data[name] = point_info
    mesh.write(save_file_path)


def save_data_to_vtk(data: Array | np.ndarray, save_file_pth: Path | str, save_file_dir: Path | str = "paraview_out") -> None:
    """
    Converts a jax/numpy array to a vtk file viewable in paraview.

    Saves the vtk file to disk containing data magnitude and
    individual components.

    Parameters
    ----------
    data : Array or np.ndarray
        Data array, usually a function evaluated at mesh nodes.
    save_file_pth : Path or str
        Filename or subpath for the VTK file.
    save_file_dir : Path or str, optional
        Base directory for storing ParaView output. Defaults to proj_root/paraview_out.
    """
    if save_file_dir == "paraview_out":
        save_file_dir = PROJ_ROOT / save_file_dir
    os.makedirs(save_file_dir, exist_ok=True)
    if isinstance(data, Array):
        data = np.array(data, dtype=np.float64)
    save_file_path = Path(save_file_dir) / save_file_pth
    mesh = generate_box_mesh()
    assert mesh.points.shape[0] == data.shape[0]
    data_magn = np.linalg.norm(data, axis=1)
    point_infos = [
        ("Data_magn", data_magn),
        ("Data_x", data[:, 0]),
        ("Data_y", data[:, 1]),
        ("Data_z", data[:, 2]),
    ]
    for name, point_info in point_infos:
        mesh.point_data[name] = point_info
    mesh.write(save_file_path)


