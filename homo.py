
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
    def project_image(self, image1, image2, RotateMatrix, translation):
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
            return image2, ret_status
        else:
            print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
            ret_status = False
            return image2, ret_status
