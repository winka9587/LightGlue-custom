
from typing import Any
from superglue_utils import estimate_pose
import cv2
import numpy as np

# 垂直拼接两图像 (w0, h0, 3) (w1, h1, 3)
def concat_image_v(image0, image1):
    image0_rows, image0_cols, _ = image0.shape
    image1_rows, image1_cols, _ = image1.shape

    if image0_cols > image1_cols:
        zeros = np.zeros((image0_rows, image0_cols - image1_cols, 3), dtype=np.uint8) + 255
        image1 = np.concatenate((image1, zeros), axis=1)
    elif image1_cols > image0_cols:
        zeros = np.zeros((image1_rows, image1_cols - image0_cols, 3), dtype=np.uint8) + 255
        out = np.concatenate((out, zeros), axis=1)

    ret_image = np.concatenate((image0, image1), axis=0)
    return ret_image

"""
    Homo_Projector
        能够根据matcher得到的点对估计相对位姿, 
        能够单独指定相对位姿计算投影后的图像
"""
class Homo_Projector:
    def __init__(self):
        pass
    
    # quick start
    # 输入kpts,内参K, 图像image, 以及其他图像
    def __call__(self, last_frame, frame, mkpts0, mkpts1, K0, K1, concat_img, thresh=1., conf=0.99999):
        projected_image = self.project_image_by_kpts(last_frame, frame, mkpts0, mkpts1, K0, K1, thresh, conf)
        if projected_image is None:
            print("Error in calculate image project by kpts")
        return concat_image_v(concat_img, projected_image)
         
    # 输入图像和关键点mkpts, 使用对应的关键点估计相对位姿, 使用估计得到的旋转和位移对图像进行变换
    def project_image_by_kpts(self, last_frame, frame, mkpts0, mkpts1, K0, K1, thresh=1., conf=0.99999):
        # 计算相对位姿
        thresh = 1.
        # ret_pose = estimate_pose(mkpts0, mkpts1, K0, K1, thresh, conf)
        ret_pose = self.estimate_relative_pose(mkpts0, mkpts1, K0, K1, thresh, conf)
        if ret_pose is None:
            print('cannot calculate relative pose')
            projected_image = frame
        else:
            rotate_matrix, translation, kpts_inlier_mask = ret_pose
            projected_image, ret_status = self.project_image(last_frame, frame, rotate_matrix, translation)
        if ret_status is False:
            margin = 10
            H0, W0 = last_frame.shape
            H1, W1 = frame.shape
            H, W = max(H0, H1), W0 + W1 + margin
            sc = min(H / 640., 2.0)
            Ht = int(30 * sc)  # text height
            txt_color_fg = (255, 255, 255)
            txt_color_bg = (0, 0, 0)
            print('cannot project image')
            text = [
                'No Project'
            ]
            for i, t in enumerate(text):
                cv2.putText(projected_image, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                            1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        # (w, h) -> (w, h, 3)
        projected_image = np.repeat(projected_image[:, :, np.newaxis], 3, axis=2)
        return projected_image
            
    def estimate_relative_pose(self, mkpts0, mkpts1, K0, K1, thresh, conf):
        MIN_MATCH_COUNT = 10
        sift = cv2.xfeatures2d.SIFT_create()
        flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

        # 计算关键点和描述符
        kpts0, desc0 = sift.detectAndCompute(mkpts0, None)
        kpts1, desc1 = sift.detectAndCompute(mkpts1, None)

        # 匹配关键点
        matches = flann.knnMatch(desc0, desc1, k=2)

        # 选择好的匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > MIN_MATCH_COUNT:
            # 获取匹配点的坐标
            src_pts = np.float32([kpts0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpts1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 计算Homography变换
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # 计算旋转矩阵和平移向量
            if M is not None:
                H = np.linalg.inv(K1) @ M @ K0
                rotate_matrix = H[:, :2]
                translation = H[:, 2] / np.linalg.norm(H[:, :2], axis=1, keepdims=True)
                return rotate_matrix, translation, mask.ravel().tolist()
        return None

            
    # 输入(3,3)旋转矩阵RotateMatrix,和(3,1)大小translation对两个图像imge1和imge2进行变换，
    # 将图像1投影到图像2，同时图像1半透明化
    def project_image2(self, image1, image2, RotateMatrix, translation):
        MIN_MATCH_COUNT = 10
        sift = cv2.xfeatures2d.SIFT_create()
        flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

        # 计算关键点和描述符
        kpts0, desc0 = sift.detectAndCompute(image1, None)
        kpts1, desc1 = sift.detectAndCompute(image2, None)

        # 匹配关键点
        matches = flann.knnMatch(desc0, desc1, k=2)

        # 选择好的匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > MIN_MATCH_COUNT:
            # 获取匹配点的坐标
            src_pts = np.float32([kpts0[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpts1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 计算Homography变换
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # 投影image1到image2上
            h, w = image1.shape[:2]
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            image2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            # 将image1半透明化
            overlay = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))
            alpha = 0.5
            image2 = cv2.addWeighted(image2, 1 - alpha, overlay, alpha, 0)
            ret_status = True
            log_text = "Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT)
        else:
            ret_status = False
            log_text = "Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT)
        # 同时在图像绘制文本, 提示match不足以支持位姿估计
        H, W = image2.shape
        sc = min(H / 640., 2.0)
        Ht = int(30 * sc)  # text height
        txt_color_fg = (255, 0, 0)
        txt_color_bg = (0, 0, 0)
        print(log_text)
        text = [
            log_text
        ]
        for i, t in enumerate(text):
            cv2.putText(image2, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
            
        image2 = np.repeat(image2[:, :, np.newaxis], 3, axis=2)
        return image2, ret_status
    


    def project_image(self, image1, image2, K1, K2, R, T, match_num):
        H, W = image2.shape
        if match_num != 0:
            # 获取图像的高度和宽度
            height, width = image1.shape

            # 创建一个网格坐标矩阵
            x1, y1 = np.meshgrid(np.arange(width), np.arange(height))
            x1 = x1.flatten()
            y1 = y1.flatten()
            z1 = np.ones_like(x1)

            # 将像素坐标转换为相机1的归一化平面坐标
            # p1 = np.vstack((x1, y1, z1))
            # p1 = np.vstack((x1.flatten(), y1.flatten(), z1.flatten())).T
            p1 = np.vstack((x1.flatten(), y1.flatten(), np.ones_like(x1.flatten())))

            p1_normalized = np.dot(np.linalg.inv(K1), p1)

            # 使用外参数将点从相机1坐标系转换到相机2坐标系
            p2 = np.dot(R, p1_normalized) + T.reshape(-1, 1)


            # 将点映射回相机2的归一化平面坐标
            p2_normalized = p2 / p2[2]

            # 使用相机2的内参数将点映射回像素坐标
            p2_pixel = np.dot(K2, p2_normalized)

            # 获取相应像素的颜色值
            x2 = p2_pixel[0].astype(int)
            y2 = p2_pixel[1].astype(int)

            # 确保映射后的像素坐标在图像2的范围内
            valid_mask = (x2 >= 0) & (x2 < width) & (y2 >= 0) & (y2 < height)

            # 创建投影后的单通道图像2
            projected_image = np.zeros_like(image2)
            projected_image[y2[valid_mask], x2[valid_mask]] = image1[y1[valid_mask], x1[valid_mask]]
        else:
            projected_image = image1.copy()

        # 将projected_image半透明化与image2拼接
        alpha = 0.5
        image2 = cv2.addWeighted(image2, 1 - alpha, projected_image, alpha, 0)
        
        # 单通道->3通道
        image2 = np.repeat(image2[:, :, np.newaxis], 3, axis=2)
        
        # 同时在图像绘制文本, 提示match不足以支持位姿估计
        log_text = "matches num: {}".format(match_num)
        
        sc = min(W / 640., 2.0)
        Ht = int(30 * sc)  # text height
        txt_color_fg = (0, 0, 0)
        txt_color_bg = (0, 0, 0)
        print(log_text)
        text = [
            log_text
        ]
        for i, t in enumerate(text):
            cv2.putText(image2, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                        1.0*sc, txt_color_bg, 2, cv2.LINE_AA)

        return image2, True
