import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
from torch.nn import Linear
from torch.nn.functional import relu


def trilinear_interpolation(volume, coordinates, device):
    D, H, W = volume.shape
    x = coordinates[..., 0]
    y = coordinates[..., 1]
    z = coordinates[..., 2]
    x0, x1 = torch.floor(x).long(), torch.ceil(x).long()
    y0, y1 = torch.floor(y).long(), torch.ceil(y).long()
    z0, z1 = torch.floor(z).long(), torch.ceil(z).long()
    x0, y0, z0 = torch.clamp(x0, 0, W-1), torch.clamp(y0, 0, H-1), torch.clamp(z0, 0, D-1)
    x1, y1, z1 = torch.clamp(x1, 0, W-1), torch.clamp(y1, 0, H-1), torch.clamp(z1, 0, D-1)

    c000 = volume[z0, y0, x0]
    c001 = volume[z0, y0, x1]
    c010 = volume[z0, y1, x0]
    c011 = volume[z0, y1, x1]
    c100 = volume[z1, y0, x0]
    c101 = volume[z1, y0, x1]
    c110 = volume[z1, y1, x0]
    c111 = volume[z1, y1, x1]

    xd, yd, zd = x - x0.float(), y - y0.float(), z - z0.float()
    c00 = c000 * (1 - xd) + c001 * xd
    c01 = c010 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd) + c101 * xd
    c11 = c110 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd
    return c.view(D, H, W)


def generate_rotation_matrix(axis, angle, device, mode='torch'):
    """
    Generates a 3x3 rotation matrix

    :param axis: A 3D list representing the axis of rotation.
    :param angle: The angle of rotation in degrees.
    :return: A 3x3 tensor representing the rotation matrix.
    """
    if mode == 'torch':
        axis = torch.tensor(axis, dtype=torch.float)

        degrees = torch.tensor(angle, dtype=torch.float)
        angle = torch.deg2rad(degrees)

        axis = axis / torch.norm(axis) # normalize to unit vector
        
        a = torch.cos(angle / 2)
        b, c, d = -axis * torch.sin(angle / 2)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return torch.tensor([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])#.to(device, dtype=torch.float)
    elif mode == 'scipy':
        angle = angle * np.pi / 180
        
        rotation = R.from_rotvec(angle * np.array(axis))
        return torch.tensor(rotation.as_matrix(), dtype=torch.float64)#.to(device, dtype=torch.float)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))


def rotate_volume(volume, rotation_matrix, device, mode='trilinear'):
    # Get the shape of the voxel grid
    D, H, W = volume.shape

    # Create a grid of 3D coordinates
    z, y, x = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W))
    z, y, x = z.float(), y.float(), x.float()
    #z, y, x = z.to(torch.float64), y.to(torch.float64), x.to(torch.float64)
    coordinates = torch.stack([x, y, z], dim=-1) # Shape: [D, H, W, 3]

    # Center the coordinates around 0
    coordinates -= torch.tensor([W // 2, H // 2, D // 2], dtype=torch.float64, requires_grad=True)

    # Apply the rotation matrix
    #rotated_coordinates = coordinates @ rotation_matrix
    rotated_coordinates = torch.einsum('ij,dhwj->dhwi', rotation_matrix, coordinates)

    # Translate the coordinates back to the original range
    rotated_coordinates += torch.tensor([W // 2, H // 2, D // 2], dtype=torch.float64, requires_grad=True)

    # Clamp the coordinates to be within the valid range
    rotated_coordinates = torch.clamp(rotated_coordinates, 0, min(D-1, H-1, W-1))

    if mode == 'nearest':
        # Create coordinates for gathering
        coordinates = rotated_coordinates.long().unbind(-1)
        # Perform gather operation to get the rotated volume
        rotated_volume = volume[coordinates[2], coordinates[1], coordinates[0]].view(D, H, W)
    elif mode == 'trilinear':
        volume = volume.to('cpu', dtype=torch.float)
        rotated_coordinates = rotated_coordinates.to('cpu', dtype=torch.float)
        rotated_volume = trilinear_interpolation(volume, rotated_coordinates, device)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))

    return rotated_volume

def normalize_volume(initial_volume, rotated_valume):
    initial_min = initial_volume.min()
    initial_max = initial_volume.max()

    normalized_volume = (rotated_valume - rotated_valume.min()) / (rotated_valume.max() - rotated_valume.min())
    normalized_volume = normalized_volume * (initial_max - initial_min) + initial_min

    return normalized_volume

def rotate_volume_cuda(volume, rotation_matrix, device, mode='trilinear'):
    # Get the shape of the voxel grid
    D, H, W = volume.shape

    # Create a grid of 3D coordinates
    z, y, x = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W))
    z, y, x = z.float(), y.float(), x.float()
    #z, y, x = z.to(torch.float64), y.to(torch.float64), x.to(torch.float64)
    coordinates = torch.stack([x, y, z], dim=-1).to(device, dtype=torch.float) # Shape: [D, H, W, 3]

    # Center the coordinates around 0
    coordinates -= torch.tensor([W // 2, H // 2, D // 2], dtype=torch.float64, requires_grad=True).to(device, dtype=torch.float)

    # Apply the rotation matrix
    #rotated_coordinates = coordinates @ rotation_matrix
    rotated_coordinates = torch.einsum('ij,dhwj->dhwi', rotation_matrix, coordinates)

    # Translate the coordinates back to the original range
    rotated_coordinates += torch.tensor([W // 2, H // 2, D // 2], dtype=torch.float64, requires_grad=True).to(device, dtype=torch.float)

    # Clamp the coordinates to be within the valid range
    rotated_coordinates = torch.clamp(rotated_coordinates, 0, min(D-1, H-1, W-1))

    # Gather the voxel values at the rotated coordinates
    #rotated_volume = torch.zeros_like(volume, requires_grad=True)
    #for d in range(D):
    #    for h in range(H):
    #        for w in range(W):
    #            x_, y_, z_ = rotated_coordinates[d, h, w].long()
    #            rotated_volume[d, h, w] = volume[z_, y_, x_]

    if mode == 'nearest':
        # Create coordinates for gathering
        coordinates = rotated_coordinates.long().unbind(-1)
        # Perform gather operation to get the rotated volume
        rotated_volume = volume[coordinates[2], coordinates[1], coordinates[0]].view(D, H, W)
    elif mode == 'trilinear':
        rotated_volume = trilinear_interpolation(volume, rotated_coordinates, device)
    else:
        raise ValueError('Invalid mode: {}'.format(mode))

    return rotated_volume


def apply_matmul(vector_neuron, rotation_matrix):
    if vector_neuron.shape[-1] != 3:
        raise ValueError('vector_neuron must have shape (..., 3)')
    #return vector_neuron @ rotation_matrix
    return torch.matmul(vector_neuron, rotation_matrix)