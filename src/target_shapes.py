"""
Utilities for generating 3D target shape masks from characters.

This module provides functionality to render text characters into binary masks
using PIL, manipulate them using JAX, and extrude them into 3D point cloud
representations. It includes support for random letter sampling, custom fonts,
rotations, and batch generation based on YAML configurations.
"""
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import yaml
from jax import Array
from PIL import Image, ImageDraw, ImageFont

PROJ_ROOT = Path(__file__).parent.parent

def create_letter_mask(
    letter: str,
    grid_size: int,
    font_size: int,
    font_type: str = "GWMSansW1GUI-Light",
    angle: float = 0.0,
) -> Array:
    """
    Create a centered binary mask of a single character.

    The function draws a character using the specified font, optionally rotates it,
    and crops it to its bounding box before centering it within a square grid
    of the specified size.

    Parameters
    ----------
    letter : str
        The character to be rendered into the mask.
    grid_size : int
        The dimensions (width and height) of the output square grid.
    font_size : int
        Size of the font in points.
    font_type : str, optional
        The filename of the TrueType font (without .ttf extension) located
        in the assets folder. Defaults to "GWMSansW1GUI-Light".
    angle : float, optional
        Rotation angle in degrees, applied counter-clockwise. Defaults to 0.0.

    Returns
    -------
    Array
        A JAX boolean array of shape `(grid_size, grid_size)` where True
        indicates the presence of the character.
    """
    font = ImageFont.truetype(PROJ_ROOT / "assets" / f"{font_type}.ttf", font_size)
    temp_size = grid_size * 3
    temp_img = Image.new("L", (temp_size, temp_size), 0)
    draw = ImageDraw.Draw(temp_img)
    draw.text((temp_size / 2, temp_size / 2), letter, font=font, fill=255, anchor="mm")
    if angle != 0:
        temp_img = temp_img.rotate(angle, expand=False, resample=Image.BICUBIC)
    bbox = temp_img.getbbox()
    letter_img = temp_img.crop(bbox)
    w, h = letter_img.size
    x = (grid_size - w) // 2
    y = (grid_size - h) // 2
    final_img = Image.new("L", (grid_size, grid_size), 0)
    final_img.paste(letter_img, (x, y))
    return jnp.array(final_img) > 0


def generate_random_letters(
    key: Array, subset: list[str] | None = None, n: int = 1
) -> list[int]:
    """
    Generate random uppercase letters, optionally from a specific subset.

    Parameters
    ----------
    key : Array
        JAX random PRNG key used for sampling.
    subset : list of str, optional
        A list of characters (e.g., ["A", "B", "C"]) to restrict the
        selection. If None, samples from 'A' through 'Z'.
    n : int, optional
        The number of letters to generate. Defaults to 1.

    Returns
    -------
    list of str
        A list containing `n` randomly generated characters.
    """
    letter_dict = {chr(code): code for code in range(65, 91)}
    if subset is None:
        return [
            chr(int(letter_code))
            for letter_code in jax.random.randint(key, (n,), minval=65, maxval=91)
        ]
    else:
        code_subset = jnp.array([letter_dict[letter] for letter in subset])
        return [
            chr(int(letter_code))
            for letter_code in jax.random.choice(key, code_subset, (n,))
        ]


def generate_target_shape(
    key: Array,
    gen_limit_config: dict[str, Any] | None = None,
    rotate: bool = True,
    letter_subset: list[str] = None,
    mesh_size: int = 5,
    grid_size: int = 29,
    font_size: int = 36,
) -> tuple[Array, dict[str, Any]]:
    """
    Generate a 3D target shape mask from a random letter.

    This function samples a random letter and depth, creates a 2D mask,
    rotates it if required, and then extrudes it into a 3D point cloud
    representation based on the provided configuration.

    Parameters
    ----------
    key : Array
        JAX random PRNG key.
    gen_limit_config : dict[str, Any]
        Configuration dictionary containing limits for depth,
        rotation angles, and domain dimensions. Defaults to None,
        which means the default 'configs/letter_generation_config.yaml' will be read.
    rotate : bool, optional
        Whether to apply a random rotation from the config. Defaults to True.
    letter_subset : list of str, optional
        A subset of characters to sample from. Defaults to None.
    mesh_size : int, optional
        The resolution parameter for the mesh. Defaults to 5.
    grid_size : int, optional
        The width and height of the letter mask grid. Defaults to 29.
    font_size : int, optional
        The font size in points used for rendering. Defaults to 36.

    Returns
    -------
    tuple of (Array, dict[str, Any])
        A tuple containing:
        - target_shape: A JAX array of shape (N, 3) representing the coordinates.
        - target_config: A dictionary containing the parameters used to generate the shape.
    """
    if gen_limit_config is None:
        gen_limit_config_path = PROJ_ROOT / "configs/letter_generation_config.yaml"
        with open(gen_limit_config_path) as f:
            gen_limit_config = yaml.safe_load(f)
    target_config = {}
    depth = float(
        jax.random.uniform(
            key,
            (),
            minval=gen_limit_config["depth"]["min"],
            maxval=gen_limit_config["depth"]["max"],
        )
    )
    letter = generate_random_letters(key, subset=letter_subset)[0]  # returns list
    target_config.update(
        {
            "mesh_size": mesh_size,
            "font_size": font_size,
            "grid_size": grid_size,
            "depth": depth,
            "letter": letter,
        }
    )
    if rotate:
        num_angles = len(gen_limit_config["rotation_angles"])
        angle_ind = jax.random.choice(key, jnp.linspace(0, num_angles - 1, num_angles))
        angle = gen_limit_config["rotation_angles"][int(angle_ind)]
        target_config["rotation_angle"] = angle
    else:
        angle = 0.0
    # Create a letter mask
    letter_mask = create_letter_mask(
        letter=letter,
        grid_size=grid_size,
        font_size=font_size,
        angle=angle,
    )
    # make 3d by stacking in z direction
    letter_mask = jnp.stack([letter_mask] * gen_limit_config["domain_len_z"], axis=-1)
    depth_vec = jnp.array([0.0, 0.0, depth], dtype=jnp.float64)
    target_shape = jnp.where(letter_mask[..., None] == 1.0, depth_vec, 0.0)
    target_shape = target_shape.reshape(-1, 3)
    return target_shape, target_config


def generate_n_target_shapes(
    key: Array,
    n: int,
    gen_limit_config_path: str | Path = "letter_generation_config.yaml",
    rotate: bool = True,
    letter_subset: list[str] = None,
) -> list[tuple[Array, dict[str, Any]]]:
    """
    Generate multiple target shape masks in batch.

    Loads the generation configuration from a YAML file and maps
    `generate_target_shape` over a sequence of random subkeys.

    Parameters
    ----------
    key : Array
        JAX random PRNG key.
    n : int
        The number of target shapes to generate.
    gen_limit_config_path : str or Path, optional
        Name or path of the configuration file within the configs folder.
        Defaults to "letter_generation_config.yaml".
    rotate : bool, optional
        Whether to apply random rotations to the letters. Defaults to True.
    letter_subset : list of str, optional
        A subset of characters to sample from. Defaults to None which is all letters.

    Returns
    -------
    list of tuple of (Array, dict[str, Any])
        A list of length `n` containing pairs of target shape arrays
        and their respective configuration dictionaries.
    """
    gen_limit_config_path = PROJ_ROOT / "configs" / gen_limit_config_path
    with open(gen_limit_config_path) as f:
        gen_limit_config = yaml.safe_load(f)
    subkeys = jax.random.split(key, n)
    targets = list(
        map(
            lambda k: generate_target_shape(
                k, gen_limit_config, rotate, letter_subset
            ),
            subkeys,
        )
    )
    return targets
