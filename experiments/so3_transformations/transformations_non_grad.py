import numpy as np

from scipy import special
from scipy.ndimage import affine_transform
import itertools
import torch

from scipy.ndimage import _ni_support


"""
Closely implemented from implementation of rotate from scipy.ndimage
"""

def generate_rotation_matrix(angle):
    c, s = special.cosdg(angle), special.sindg(angle)
    return np.array([[c, s], [-s, c]])

# issue: gradients don't flow through this function
def rotate_custom(input, rot_matrix, axes=(1, 0), reshape=True, output=None, order=3,
           mode='constant', cval=0.0, prefilter=True, verbose=False):
    input_arr = np.asarray(input)
    ndim = input_arr.ndim
    axes = list(axes)

    # some safe checks
    if ndim < 2:
        print('Danger! input array should be at least 2D')
        input_arr = np.expand_dims(input_arr, axis=0)
    if len(axes) != 2:
        raise ValueError('axes should contain exactly two values')
    if not all([float(ax).is_integer() for ax in axes]):
        raise ValueError('axes should contain only integer values')
    
    # safe guard for negative axes for reverse indexing
    if axes[0] < 0:
        axes[0] += ndim
    if axes[1] < 0:
        axes[1] += ndim
    if axes[0] < 0 or axes[1] < 0 or axes[0] >= ndim or axes[1] >= ndim:
        raise ValueError('invalid rotation plane specified')
    axes.sort()


    img_shape = np.asarray(input_arr.shape)
    in_plane_shape = img_shape[axes]


    if reshape:
        # Compute transformed input bounds
        iy, ix = in_plane_shape
        out_bounds = rot_matrix @ [[0, 0, iy, iy],
                                   [0, ix, 0, ix]]
        # Compute the shape of the transformed input plane
        out_plane_shape = (out_bounds.ptp(axis=1) + 0.5).astype(int)
    else:
        out_plane_shape = img_shape[axes]


    out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
    in_center = (in_plane_shape - 1) / 2
    offset = in_center - out_center

    output_shape = img_shape
    output_shape[axes] = out_plane_shape
    output_shape = tuple(output_shape)
    
    if verbose:
        print(f'Output Shape: {output_shape}')

    output = np.zeros(output_shape, dtype=input_arr.dtype.name)
    #complex_output = np.iscomplexobj(input_arr)
    #output = _ni_support._get_output(output, input_arr, shape=output_shape,
    #                                 complex_output=complex_output)

    if ndim == 2:
        affine_transform(input_arr, rot_matrix, offset, output_shape, output,
                         order, mode, cval, prefilter)
    else:
        # If ndim > 2, the rotation is applied over all the planes
        # parallel to axes
        planes_coord = itertools.product(
            *[[slice(None)] if ax in axes else range(img_shape[ax])
              for ax in range(ndim)])
        

        out_plane_shape = tuple(out_plane_shape)

        for coordinates in planes_coord:            
            ia = input_arr[coordinates]
            oa = output[coordinates]

            affine_transform(ia, rot_matrix, offset, out_plane_shape,
                             oa, order, mode, cval, prefilter)
    return output

def rotate_custom_angle(input, angle, axes=(1, 0), reshape=True, output=None, order=3,
           mode='constant', cval=0.0, prefilter=True, verbose=False):
    input_arr = np.asarray(input)
    ndim = input_arr.ndim

    if ndim < 2:
        raise ValueError('input array should be at least 2D')

    axes = list(axes)

    if len(axes) != 2:
        raise ValueError('axes should contain exactly two values')

    if not all([float(ax).is_integer() for ax in axes]):
        raise ValueError('axes should contain only integer values')

    if axes[0] < 0:
        axes[0] += ndim
    if axes[1] < 0:
        axes[1] += ndim
    if axes[0] < 0 or axes[1] < 0 or axes[0] >= ndim or axes[1] >= ndim:
        raise ValueError('invalid rotation plane specified')

    axes.sort()

    c, s = special.cosdg(angle), special.sindg(angle)

    rot_matrix = np.array([[c, s],
                              [-s, c]])

    img_shape = np.asarray(input_arr.shape)
    in_plane_shape = img_shape[axes]
    if reshape:
        # Compute transformed input bounds
        iy, ix = in_plane_shape
        out_bounds = rot_matrix @ [[0, 0, iy, iy],
                                   [0, ix, 0, ix]]
        # Compute the shape of the transformed input plane
        out_plane_shape = (out_bounds.ptp(axis=1) + 0.5).astype(int)
    else:
        out_plane_shape = img_shape[axes]

    out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
    in_center = (in_plane_shape - 1) / 2
    offset = in_center - out_center

    output_shape = img_shape
    output_shape[axes] = out_plane_shape
    output_shape = tuple(output_shape)

    complex_output = np.iscomplexobj(input_arr)
    output = _ni_support._get_output(output, input_arr, shape=output_shape,
                                     complex_output=complex_output)

    if ndim <= 2:
        affine_transform(input_arr, rot_matrix, offset, output_shape, output,
                         order, mode, cval, prefilter)
    else:
        # If ndim > 2, the rotation is applied over all the planes
        # parallel to axes
        planes_coord = itertools.product(
            *[[slice(None)] if ax in axes else range(img_shape[ax])
              for ax in range(ndim)])

        out_plane_shape = tuple(out_plane_shape)

        for coordinates in planes_coord:
            ia = input_arr[coordinates]
            oa = output[coordinates]
            affine_transform(ia, rot_matrix, offset, out_plane_shape,
                             oa, order, mode, cval, prefilter)

    return output

def rotate_custom_torch(input, rot_matrix, axes=(1, 0), reshape=True, order=3,
                        mode='constant', cval=0.0, prefilter=True, verbose=False):
    input_tensor = torch.tensor(input)
    ndim = input_tensor.ndim
    axes = list(axes)

    # some safe checks
    if ndim < 2:
        print('Danger! input array should be at least 2D')
        input_tensor = input_tensor.unsqueeze(0)
    if len(axes) != 2:
        raise ValueError('axes should contain exactly two values')
    if not all([float(ax).is_integer() for ax in axes]):
        raise ValueError('axes should contain only integer values')

    # safe guard for negative axes for reverse indexing
    if axes[0] < 0:
        axes[0] += ndim
    if axes[1] < 0:
        axes[1] += ndim
    if axes[0] < 0 or axes[1] < 0 or axes[0] >= ndim or axes[1] >= ndim:
        raise ValueError('invalid rotation plane specified')
    axes.sort()

    img_shape = torch.tensor(input_tensor.shape)
    in_plane_shape = img_shape[axes]

    if reshape:
        # Compute transformed input bounds
        iy, ix = in_plane_shape
        out_bounds = rot_matrix @ torch.tensor([[0, 0, iy, iy], [0, ix, 0, ix]])
        # Compute the shape of the transformed input plane
        out_plane_shape = (out_bounds.ptp(axis=1) + 0.5).int()
    else:
        out_plane_shape = img_shape[axes]

    out_center = rot_matrix @ ((out_plane_shape - 1) / 2).float()
    in_center = (in_plane_shape - 1) / 2
    offset = in_center - out_center

    output_shape = img_shape.tolist()
    output_shape[axes[0]], output_shape[axes[1]] = out_plane_shape.tolist()

    if verbose:
        print(f'Output Shape: {output_shape}')

    output = torch.zeros(output_shape, dtype=input_tensor.dtype)

    if ndim == 2:
        raise NotImplementedError("2D rotation is not supported in PyTorch")
    else:
        # If ndim > 2, the rotation is applied over all the planes
        # parallel to axes
        planes_coord = itertools.product(
            *[[slice(None)] if ax in axes else range(img_shape[ax])
              for ax in range(ndim)])

        for coordinates in planes_coord:
            ia = input_tensor[coordinates]
            oa = output[coordinates]

            # Manually implement the affine transformation
            grid = torch.tensor(list(itertools.product(
                torch.linspace(-0.5, out_plane_shape[0] - 0.5, out_plane_shape[0]),
                torch.linspace(-0.5, out_plane_shape[1] - 0.5, out_plane_shape[1])
            )))
            grid = grid.t().unsqueeze(0).repeat(ia.shape[0], 1, 1)

            print(f'Grid Shape: {grid.shape}')

            grid = (rot_matrix @ (grid - offset).unsqueeze(2)).squeeze(2)
            oa = torch.nn.functional.grid_sample(ia.unsqueeze(0).unsqueeze(0), grid.unsqueeze(0), mode=mode, padding_mode='zeros')
            output[coordinates] = oa.squeeze()

    return output 