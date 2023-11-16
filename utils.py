import numpy as np

def compute_3d_points(mkpts0, K0_depth, frame_depth):
    # Convert the 2D keypoints to homogeneous coordinates
    homogeneous_pts = np.concatenate([mkpts0, np.ones((mkpts0.shape[0], 1))], axis=1).T

    # Invert the camera matrix
    K0_inv = np.linalg.inv(K0_depth)

    # Compute the 3D points in the camera frame
    points_3d = []
    for i in range(homogeneous_pts.shape[1]):
        u, v, _ = homogeneous_pts[:, i]
        Z = frame_depth[int(v), int(u)]
        X, Y, W = Z * K0_inv @ homogeneous_pts[:, i]
        points_3d.append([X/W, Y/W, Z])

    return np.array(points_3d)



def convert_2d_to_3d(depth, K, coords=None, norm_scale=1000.0):
    
    if coords is None:
        coords_nonzero = np.transpose(np.vstack(np.where(depth != 0)), [1,0])  # x,y -> depth[x, y]
        # coords_nonzero = coords_nonzero[:, ::-1]
        coords_ = coords_nonzero
    else:
        coords_ = coords[:, ::-1]

    depth_masked = get_depth_at_coords(coords_, depth).reshape(-1, 1)
    
    if coords is not None:
        # 如果预先提供了coords, 其读取的深度可能存在0深度的点, 需要去除
        depth_nonzero_idx = (depth_masked!=0).squeeze(-1)
        coords_ = coords_[depth_nonzero_idx]  # 同时将对应的coords也去除
        depth_masked = depth_masked[depth_nonzero_idx]  # 将depth_masked中0深度的点去除
    else:
        depth_nonzero_idx = None
        

    # Compute the x and y coordinates in the camera frame
    xmap_masked = coords_[:, 1][:, np.newaxis]
    ymap_masked = coords_[:, 0][:, np.newaxis]

    # Compute the 3D points in the camera frame
    pt2 = depth_masked / (K[2, 2] * norm_scale)
    pt0 = (xmap_masked - K[0, 2]) * pt2 / K[0, 0]
    pt1 = (ymap_masked - K[1, 2]) * pt2 / K[1, 1]

    # Concatenate the x, y, and z coordinates to form the 3D points
    points = np.concatenate((pt0, pt1, pt2), axis=1)

    return points, depth_nonzero_idx


def get_depth_at_coords(coords, depth):
    """
    Get the depth values at the given 2D coordinates by bilinear interpolation
    :param coords: 2D coordinates of shape (N, 2)
    :param depth: depth image of shape (H, W)
    :return: depth values at the given coordinates of shape (N,)
    """
    H, W = depth.shape
    x, y = coords[:, 1], coords[:, 0]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.ceil(x).astype(int), np.ceil(y).astype(int)
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)
    Ia = depth[y0, x0]
    Ib = depth[y1, x0]
    Ic = depth[y0, x1]
    Id = depth[y1, x1]
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    # Set the weight to 0 for points where the depth is 0
    wa = np.where(depth[y0, x0] == 0, 0, wa).astype('float64')
    wb = np.where(depth[y1, x0] == 0, 0, wb).astype('float64')
    wc = np.where(depth[y0, x1] == 0, 0, wc).astype('float64')
    wd = np.where(depth[y1, x1] == 0, 0, wd).astype('float64')
    
    # Normalize the weights
    total_weight = wa + wb + wc + wd
    
    idx = np.where(total_weight == 0)
    wa[idx] = 0.25
    wb[idx] = 0.25
    wc[idx] = 0.25
    wd[idx] = 0.25
    total_weight[idx] = 1.0

    wa /= total_weight
    wb /= total_weight
    wc /= total_weight
    wd /= total_weight
    
    # Check if the coordinates are exactly on a pixel
    on_pixel = (x == x0) & (y == y0)
    # If the coordinates are exactly on a pixel, return the depth value at that pixel
    I = np.where(on_pixel, Ia, wa * Ia + wb * Ib + wc * Ic + wd * Id)
    return I


import torch
def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """
    _EPS = 1e-5
    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)  # 计算两组点云加权的中心点
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]  # 将两组点云中心化
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)  # SVD中的协方差

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)  # SVD分解
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
def create_colorbar():
    plt.ioff()  # 关闭交互模式
    # 创建一个从0到1的线性间隔的数组，表示置信度的可能范围
    confidence_values = np.linspace(0, 1, 256).reshape(256, 1)[::-1, :]
    confidence_values = np.repeat(confidence_values, 5, axis=-1)
    # 创建一个新的图像
    fig, ax = plt.subplots(figsize=(1, 6))

    # 在图像中显示置信度值
    cax = ax.imshow(confidence_values, cmap='jet', origin='lower')
    ax.set_axis_off()

    # 添加一个颜色条的标签
    # fig.colorbar(cax, ax=ax, orientation='vertical', label='Confidence')

    # 将Figure对象转换为RGB图像
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # 关闭图像，释放资源
    plt.close(fig)
    plt.ion()  # 打开交互模式
    return img


def add_colorbar_to_image(image, colorbar):
    # 确保两个图像的高度相同
    if image.shape[0] != colorbar.shape[0]:
        scale_factor = image.shape[0] / colorbar.shape[0]
        colorbar = cv2.resize(colorbar, None, fx=scale_factor, fy=scale_factor)
    # 将两个图像拼接在一起
    combined_image = np.hstack((image, colorbar))
    return combined_image


def generate_bounding_box(image):
    """
    Generate 2D bounding box using points with pixel value of 1 in the image.

    Args:
        image: numpy array of shape (H, W) representing the image.

    Returns:
        A tuple of (x_min, y_min, x_max, y_max) representing the bounding box coordinates.
    """
    # Find all 2d points with pixel value of 1
    points = np.argwhere(image == 1)
    # Get the minimum and maximum x and y coordinates
    x_min = np.min(points[:, 1])
    y_min = np.min(points[:, 0])
    x_max = np.max(points[:, 1])
    y_max = np.max(points[:, 0])
    # Make the bounding box a square by taking the larger of the width and height
    width = x_max - x_min
    height = y_max - y_min
    if width > height:
        y_center = (y_min + y_max) // 2
        y_max = y_min + width
        y_min = y_center - width // 2
    else:
        x_center = (x_min + x_max) // 2
        x_max = x_min + height
        x_min = x_center - height // 2
    # Ensure the bounding box does not go out of bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)
    # Return the modified bounding box
    return x_min, y_min, x_max, y_max


def crop_image_by_bounding_box(bounding_box, image):
    """
    Crop an image based on a given bounding box.

    Args:
        bounding_box: A tuple of (x_min, y_min, x_max, y_max) representing the bounding box coordinates.
        image: A numpy array of shape (H, W) representing the image.

    Returns:
        A numpy array of shape (y_max - y_min, x_max - x_min) representing the cropped image.
    """
    x_min, y_min, x_max, y_max = bounding_box
    # Ensure the bounding box does not go out of bounds
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image.shape[1], x_max)
    y_max = min(image.shape[0], y_max)
    return image[y_min:y_max, x_min:x_max]



def pad_image(image, target_shape):
    """
    Pad an image with zeros to match a target shape.

    Args:
        image: A numpy array of shape (H, W) representing the image.
        target_shape: A tuple of (H, W) representing the target shape.

    Returns:
        A numpy array of shape (target_shape[0], target_shape[1]) representing the padded image.
    """
    # Get the dimensions of the input image
    height, width = image.shape
    # Get the dimensions of the target shape
    target_height, target_width = target_shape
    # If the input image is already larger than the target shape, return the input image
    if height >= target_height and width >= target_width:
        return image
    # Compute the amount of padding needed in each dimension
    pad_height = max(0, target_height - height)
    pad_width = max(0, target_width - width)
    # Pad the image with zeros
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant')
    return padded_image
