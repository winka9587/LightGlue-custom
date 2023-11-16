# from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED
# from lightglue.utils import load_image, rbd

# # SuperPoint+LightGlue
# extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()  # load the extractor
# matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher

# # or DISK+LightGlue, ALIKED+LightGlue or SIFT+LightGlue
# extractor = DISK(max_num_keypoints=2048).eval().cuda()  # load the extractor
# matcher = LightGlue(features='disk').eval().cuda()  # load the matcher

# # load each image as a torch.Tensor on GPU with shape (3,H,W), normalized in [0,1]
# image0 = load_image('/data1/jl/awsl-JL/object-deformnet-master/object-deformnet-master/data/Real/test/scene_1//0000_color.png').cuda()
# image1 = load_image('/data1/jl/awsl-JL/object-deformnet-master/object-deformnet-master/data/Real/test/scene_1//0277_color.png').cuda()

# print(image0.shape)
# import cv2, torch
# cv2.imshow("img0", torch.permute(image0, (1,2,0)).cpu().numpy())
# cv2.waitKey(0)

# # extract local features
# feats0 = extractor.extract(image0)  # auto-resize the image, disable with resize=None
# feats1 = extractor.extract(image1)

# # match the features
# matches01 = matcher({'image0': feats0, 'image1': feats1})
# feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
# matches = matches01['matches']  # indices with shape (K,2)
# points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
# points1 = feats1['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)




from pathlib import Path
import argparse
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from homo import Homo_Projector, concat_image_v
import numpy as np

from utils import create_colorbar, add_colorbar_to_image, generate_bounding_box, crop_image_by_bounding_box, pad_image

# 引入LightGlue相关的类和函数
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED

from superglue_utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)


# 读取不定长的输入input，用于初始化与输入同样数量的VideoStreamer
# BEGIN: 7f0d3m5x8z9a
class VideoStreamerCollect:
    def __init__(self, inputs, labels, vs_opt):
        assert len(inputs) == len(labels)
        vs_dict = {labels[i]: VideoStreamer(input_src, vs_opt.resize, vs_opt.skip,
                            vs_opt.image_glob, vs_opt.max_length, labels[i]) for i, input_src in enumerate(inputs) if input_src is not None}
        self.vs_collection = vs_dict
        
    def next_frame(self):
        frame_collect = {}
        ret_collect = {}
        for label, vs in self.vs_collection.items():
            frame, ret = vs.next_frame()
            frame_collect[label] = frame
            ret_collect[label] = ret
        return frame_collect, ret_collect
    
    def get_i(self):
        i_list = {label: vs.i for label, vs in self.vs_collection.items()}
        assert all(x == list(i_list.values())[0] for x in i_list.values())
        return i_list['rgb']
    
    
    def cleanup(self):
        for label, vs in self.vs_collection.items():
            vs.cleanup()
            print("{} VideoStreamer clean up".format(label))

# END: 7f0d3m5x8z9a
# ...

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default='/data4/cxx/dataset/desk.mp4',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file. default=\'0\' use camera')
    parser.add_argument(
        '--input_depth', type=str, default=None,
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file. default=\'0\' use camera')
    parser.add_argument(
        '--input_mask', type=str, default=None,
        help='Only match keypoints in the mask area')
    
    
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    # SuperPoint Setting
    parser.add_argument(
        '--max_num_keypoints', type=int, default=2048,
        help='Maximum number of keypoints detected by Superpoint'
             ' (SuperGlue: \'-1\' keeps all keypoints, LightGlue default 2048)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    
    # Run setting
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    
    parser.add_argument(
        '--homo', action='store_true',
        help='project image0 to image1')
    parser.add_argument(
        '--depth_match', action='store_true',
        help='match 3D pointcloud')

    opt = parser.parse_args()
    print(opt)

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    # ...

    # 可视化用颜色条
            
    colorbar = create_colorbar()

    # 初始化LightGlue
    extractor = SuperPoint(max_num_keypoints=opt.max_num_keypoints).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    keys = ['keypoints', 'keypoint_scores', 'descriptors']
    # ...
    
    input_labels = ['rgb', 'depth', 'mask']
    input_paths = [opt.input, opt.input_depth, opt.input_mask]
    vsc = VideoStreamerCollect(input_paths, input_labels, opt)
    frames, rets = vsc.next_frame()
    assert all(rets), 'Error when reading the first frame (try different --input?)'
    
    frame = frames['rgb']
    frame_tensor = frame2tensor(frame, device)
    # 使用LightGlue提取特征
    last_data = extractor.extract(frame_tensor)
    # last_data = {'keypoints0': last_data['keypoints'], 'descriptors0': last_data['descriptors']}
    # last_data['image0'] = frame_tensor
    last_frame = frame
    last_frame_depth = frames['depth']
    if opt.input_mask is not None:
        last_frame_mask = frames['mask']

    frames, rets = vsc.next_frame()
    last_image_id = 0

    """
    color intrinsics is:  [ 640x480  p[326.371 243.675]  f[384.918 383.909]  Inverse Brown Conrady [-0.0535972 0.062711 -0.00110631 6.81885e-05 -0.0201631] ]
    depth intrinsics is:  [ 640x480  p[321.701 239.973]  f[389.702 389.702]  Brown Conrady [0 0 0 0 0] ]
    """
    fx = 384.918
    fy = 383.909
    cx = 326.371
    cy = 243.675
    K0 = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    K1 = K0.copy()
    K0_depth = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    K1_depth = K0_depth.copy()
    # ...
    timer = AverageTimer()
    if opt.homo:
        hpr = Homo_Projector()
    while True:
        frames, rets = vsc.next_frame()
        if not any(rets):
            print('Finished demo_superglue.py')
            break
        frame = frames['rgb']
        frame_depth = frames['depth']
        if opt.input_mask is not None:
            frame_mask = frames['mask']
        timer.update('data')
        stem0, stem1 = last_image_id, vsc.get_i() - 1
        # ...

        frame_tensor = frame2tensor(frame, device)
        # 提取当前帧的特征
        curr_data = extractor.extract(frame_tensor)
        # curr_data = {'keypoints1': curr_data['keypoints'], 'descriptors1': curr_data['descriptors']}
        # 使用LightGlue匹配特征
        # pred = matcher.match({'image0': last_data['descriptors0'], 'image1': curr_data['descriptors1']})
        # match the features
        # pred = matcher({'image0': last_data['descriptors0'], 'image1': curr_data['descriptors1']})
        pred = matcher({'image0': last_data, 'image1': curr_data})

        # 需要编写逻辑来解析matches的结果
        # 注意：以下代码块是伪代码，您需要根据LightGlue返回的匹配格式进行调整
        kpts0 = last_data['keypoints'][0].cpu().numpy()
        kpts1 = curr_data['keypoints'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()  # 这个可能需要根据LightGlue的输出调整
        # 其余的显示和处理逻辑与原先的SuperGlue代码类似
        # 拼接图像并在图像上可视化匹配结果kpts0和kpts1
        timer.update('forward')

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        scores = confidence[valid]  # 用于match
        color = cm.jet(confidence[valid])

        text = [
            'LightGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = extractor.conf.detection_threshold
        m_thresh = matcher.conf.filter_threshold
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        
        if opt.input_mask is not None:
            last_bbox = generate_bounding_box(last_frame_mask)
            bbox = generate_bounding_box(frame_mask)
    
            # 检查mkpts0与mkpts1的2D坐标，分别保留其中在last_bbox和bbox中的点。记录保留点的下标choose_bbox0和choose_bbox1
            from utils import generate_bounding_box
            last_bbox = generate_bounding_box(last_frame_mask)
            bbox = generate_bounding_box(frame_mask)
            choose_bbox0 = np.logical_and(mkpts0[:, 0] > last_bbox[0], mkpts0[:, 0] < last_bbox[2]) & np.logical_and(mkpts0[:, 1] > last_bbox[1], mkpts0[:, 1] < last_bbox[3])
            choose_bbox1 = np.logical_and(mkpts1[:, 0] > bbox[0], mkpts1[:, 0] < bbox[2]) & np.logical_and(mkpts1[:, 1] > bbox[1], mkpts1[:, 1] < bbox[3])
            choose_bbox_idx = np.logical_and(choose_bbox0, choose_bbox1)

            # Keep the remaining keypoint matches
            mkpts0 = mkpts0[choose_bbox_idx]
            mkpts1 = mkpts1[choose_bbox_idx]
            scores = scores[choose_bbox_idx]
            # 将last_frame中last_bbox以外的部分设置为白色
            last_frame_mask_tmp = np.zeros_like(last_frame_mask)
            frame_mask_tmp = np.zeros_like(frame_mask)
            last_frame_mask_tmp[last_bbox[1]:last_bbox[3], last_bbox[0]:last_bbox[2]] = 255
            last_frame[last_frame_mask_tmp == 0] = 255
            
            # 让bbox以外的部分变为白色, bbox以内的部分半透明，mask==1的部分保持原来的样子。
            import numpy as np
            import cv2

            # 创建一个frame的副本
            frame_copy = frame.copy()
            # 将bbox以外的部分设置为白色
            frame_copy[:bbox[1], :] = 255
            frame_copy[bbox[3]:, :] = 255
            frame_copy[:, :bbox[0]] = 255
            frame_copy[:, bbox[2]:] = 255

            # bbox以内的部分半透明
            alpha = 0.5
            overlay = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
            cv2.addWeighted(overlay, alpha, frame_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]], 1 - alpha, 0, frame_copy[bbox[1]:bbox[3], bbox[0]:bbox[2]])
            # mask==1的部分保持原来的颜色
            frame_copy[frame_mask == 1] = frame[frame_mask == 1]
            frame = frame_copy
        
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
        

        out = add_colorbar_to_image(out, colorbar)
        
        if opt.homo:
            # last_frame, frame, mkpts0, mkpts1, K0, K1, concat_img, thresh=1., conf=0.99999
            out = hpr(last_frame, frame, mkpts0, mkpts1, K0, K1, out)

        
        if opt.depth_match:
            # 将匹配的深度点反投影得到3D点
            from utils import convert_2d_to_3d
            pts0_3d, _ = convert_2d_to_3d(last_frame_depth, K0)
            pts1_3d, _ = convert_2d_to_3d(frame_depth, K1)       
            # mkpts0与mkpts1的shape是相同的
            mkpts0_3d, choose0 = convert_2d_to_3d(last_frame_depth, K0, coords=mkpts0)  
            mkpts1_3d, choose1 = convert_2d_to_3d(frame_depth, K1, coords=mkpts1) 
            choose_ = np.logical_and(choose0, choose1)
            choose_idx = np.where(choose_)
            # 下面输入的coords=参数已经必定是有深度值的, 因此第二个返回值不再需要了
            mkpts0_3d, _ = convert_2d_to_3d(last_frame_depth, K0, coords=mkpts0[choose_idx])  
            mkpts1_3d, _ = convert_2d_to_3d(frame_depth, K1, coords=mkpts1[choose_idx]) 
            
            mkpts0_3d_choosed = mkpts0_3d
            mkpts1_3d_choosed = mkpts1_3d
            scores = scores[choose_]
            from vision_tool.PointCloudRender import PointCloudRender
            # pcr = PointCloudRender()
            # pcr.render_multi_pts("pts0 and kpts0", [pts0_3d, mkpts0_3d], [np.array([0, 0, 255]), np.array([255, 0, 0])])
            # pcr.render_multi_pts("pts1 and kpts1", [pts1_3d, mkpts0_3d], [np.array([0, 0, 255]), np.array([255, 0, 0])])
            
            # choose0和choose1取交集choose_
            # mkpts0_3d = mkpts0_3d[choose_]
            # mkpts1_3d = mkpts1_3d[choose_]
            from utils import compute_rigid_transform
            import torch
            transform = compute_rigid_transform(torch.from_numpy(mkpts0_3d_choosed).unsqueeze(0), 
                                                torch.from_numpy(mkpts1_3d_choosed).unsqueeze(0), 
                                                torch.from_numpy(scores).unsqueeze(0))
            transform = transform.squeeze(0).numpy()
            
            
            hpr2 = Homo_Projector()
            rotate_matrix = transform[:3, :3]
            translation = transform[:3, 3]
            projected_image, ret_status = hpr2.project_image(last_frame, frame, rotate_matrix, translation)
            # 将last_frame的关键点mkpts0[choose_idx]也投影到frame上, 并与frame的mkpts1[choose_idx]连线,
            # 线的颜色使用cm.jet(scores)
            
            def project_points(points_3d, rotate_matrix, translation, camera_matrix):
                """
                将3D点投影到2D平面上
                :param points_3d: 3D点, shape=(N, 3)
                :param rotate_matrix: 旋转矩阵, shape=(3, 3)
                :param translation: 平移向量, shape=(3,)
                :param camera_matrix: 相机内参矩阵, shape=(3, 3)
                :return: 投影后的2D点, shape=(N, 2)
                """
                points_3d = np.dot(points_3d, rotate_matrix.T) + translation
                points_2d = np.dot(points_3d, camera_matrix.T)
                points_2d = points_2d[:, :2] / points_2d[:, 2:]
                return points_2d

            projected_mkpts0 = project_points(mkpts0_3d_choosed, rotate_matrix, translation, K0)
            projected_mkpts0 = projected_mkpts0[:, :2].astype(int)
            projected_image = np.repeat(projected_image[:, :, np.newaxis], 3, axis=2)
            
            line_color = cm.jet(scores)[:, :3][:, ::-1]  # matplotlib RGB, opencv BGR, 同时注意, jet生成的是4位的颜色
            for i in range(len(projected_mkpts0)):
                start_point = tuple(map(int, projected_mkpts0[i]))
                end_point = tuple(map(int, mkpts1[choose_idx][i]))
                cv2.line(projected_image, start_point, end_point, tuple(line_color[i]*255.), 2)
                        
            
            out = concat_image_v(out, projected_image)

        if not opt.no_display:            
            cv2.imshow('LightGlue matches', out)
            
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vsc.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k: curr_data[k] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_frame_depth = frame_depth
                if opt.input_mask is not None:
                    last_frame_mask = frame_mask
                last_image_id = (vsc.get_i() - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                extractor.conf.detection_threshold = min(max(
                    0.0001, extractor.conf.detection_threshold*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    extractor.conf.detection_threshold))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matcher.conf.filter_threshold = min(max(
                    0.05, matcher.conf.filter_threshold+d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matcher.conf.filter_threshold))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints

        timer.update('viz')
        timer.print()

        if opt.output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

    cv2.destroyAllWindows()
    vsc.cleanup()