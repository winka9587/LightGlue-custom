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
import matplotlib.cm as cm
import torch

# 引入LightGlue相关的类和函数
from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED

from superglue_utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

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

    opt = parser.parse_args()
    print(opt)

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    # ...

    # 初始化LightGlue
    extractor = SuperPoint(max_num_keypoints=opt.max_num_keypoints).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    keys = ['keypoints', 'keypoint_scores', 'descriptors']
    # ...

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    # 使用LightGlue提取特征
    last_data = extractor.extract(frame_tensor)
    # last_data = {'keypoints0': last_data['keypoints'], 'descriptors0': last_data['descriptors']}
    # last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0

    # ...
    timer = AverageTimer()
    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished demo_superglue.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1
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
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
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
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)

        if not opt.no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k: curr_data[k] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
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
    vs.cleanup()