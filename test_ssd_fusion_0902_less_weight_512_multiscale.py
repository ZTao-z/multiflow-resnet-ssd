from __future__ import print_function
import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import pickle
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
from torch.autograd import Variable
from data import VOCroot, COCOroot, COCO_300, COCO_512, COCODetection
from data import AnnotationTransform, VOCDetection, BaseTransform, VOC_300, VOC_512

from ssd_fusion_0311_v2_less_weight_512_VOC_ms import build_ssd
import torch.utils.data as data
from layers.functions import Detect, PriorBox
from utils.nms_wrapper import nms
from utils.timer import Timer
import cv2
import math

parser = argparse.ArgumentParser(description='Receptive Field Block Net')

parser.add_argument('-v', '--version', default='RFB_vgg',
                    help='RFB_vgg ,RFB_E_vgg or RFB_mobile version.')
parser.add_argument('-s', '--size', default='512',
                    help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default='weights/RFB300_80_5.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--cpu', default=False, type=bool,
                    help='Use cpu nms')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--device', type=int, help='cuda device')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    cfg = (COCO_300, COCO_512)[args.size == '512']

# torch.cuda.set_device(args.device)

priorbox = PriorBox(cfg)
priors = Variable(priorbox.forward(), volatile=True)
priors = priors.cuda()


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005, obj_thresh=0.01):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = (21, 81)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')
    total_detect_time = 0
    total_nms_time = 0

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        print(all_boxes[0:10])
        all_boxes = np.array(all_boxes)
        print(all_boxes.shape)
        testset.evaluate_detections(all_boxes, save_folder)
        return

    for i in range(num_images):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]])

        x = Variable(transform(img).unsqueeze(0), volatile=True)
        if cuda:
            x = x.cuda()
            scale = scale.cuda()

        _t['im_detect'].tic()
        out = net(x)  # forward pass
        boxes, scores = detector.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        total_detect_time += detect_time
        if i == 0:
            total_detect_time -= detect_time

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=args.cpu)
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()
        total_nms_time += nms_time
        if i == 0:
            total_nms_time -= nms_time

        if i % 20 == 0 and i != 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                  .format(i + 1, num_images, detect_time, nms_time))
            print('detect:%4f nms: %4f fps:%4f  %4f' % (total_detect_time, total_nms_time,
                                                        (i - 1) / (total_nms_time + total_detect_time),
                                                        (i - 1) / total_detect_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)


def im_detect(net, im_org, target_size, transform, cuda, means):
    # im = cv2.resize(im_org,target_size,target_size,3)
    im = cv2.resize(np.array(im_org), (target_size, target_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    im -= means
    im = im.transpose((2, 0, 1))
    scale = torch.Tensor([im_org.shape[1], im_org.shape[0],
                          im_org.shape[1], im_org.shape[0]])

    x = Variable((torch.from_numpy(im)).unsqueeze(0), volatile=True)
    if cuda:
        x = x.cuda()
        scale = scale.cuda()

    out = net(x)

    cfg_temp = VOC_512
    cfg['min_dim'] = target_size
    size = math.ceil(target_size / 4)
    multi = target_size / 300
    for i in range(0, len(cfg['feature_maps'])):
        size = net.sizes[i]
        cfg['feature_maps'][i] = size
    # for i in range(0,len(cfg['min_sizes'])):
    #     cfg['min_sizes'][i] *= multi
    #     cfg['max_sizes'][i] *= multi
    priorbox_temp = PriorBox(cfg_temp)
    priors_temp = priorbox_temp.forward().cuda()
    priors_temp = Variable(priors_temp, volatile=True)

    boxes, scores = detector.forward(out, priors_temp)
    boxes = boxes[0]
    scores = scores[0]

    # scale = target_size
    boxes *= scale
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    return (boxes, scores)


def flip_im_detect(net, im_org, target_size, transform, cuda, means):
    im_f = cv2.flip(im_org, 1)
    det_f, scores = im_detect(net, im_f, target_size, transform, cuda, means)

    det_t = np.zeros(det_f.shape)
    det_t[:, 0] = im_org.shape[1] - det_f[:, 2]
    det_t[:, 1] = det_f[:, 1]
    det_t[:, 2] = im_org.shape[1] - det_f[:, 0]
    det_t[:, 3] = det_f[:, 3]

    return det_t, scores
    # det_t[:, 4] = det_f[:, 4]
    # det_t[:, 5] = det_f[:, 5]


def bbox_vote(det):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    # det = det[np.where(det[:, 4] > 0.2)[0], :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these  det
        merge_index = np.where(o >= 0.45)[0]
        det_accu = det[merge_index, :]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(det_accu[:, -1:], (1, 4))
            max_score = np.max(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score
            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    return dets


def multi_scale_test(save_folder, net, detector, cuda, testest, transform, means, max_per_image=300, thresh=0.005,
                     obj_thresh=0.01):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = (21, 81)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    target_size = 512

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')
    total_detect_time = 1
    total_nms_time = 0

    if args.retest:
        f = open(det_file, 'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return

    for i in range(num_images):
        img = testset.pull_image(i)

        # org
        boxes, scores = im_detect(net, img, target_size, transform, cuda, means)
        boxes_f, scores_f = flip_im_detect(net, img, target_size, transform, cuda, means)
        boxes0 = np.row_stack((boxes, boxes_f))
        scores0 = np.row_stack((scores, scores_f))

        # shrink only detect small objects
        # boxes1, scores1 = im_detect(net,img,int(0.9*target_size),transform,cuda,means)
        # total_boxes = np.row_stack((total_boxes,boxes1))
        # total_scores = np.row_stack((total_scores,scores1))

        #shrink
        boxes_s1, scores_s1 = im_detect(net, img, int( 0.53* target_size), transform, cuda, means)
        boxes_s1_f, scores_s1_f = flip_im_detect(net, img, int(0.53 * target_size), transform, cuda, means)
        boxes_s1 = np.row_stack((boxes_s1, boxes_s1_f))
        scores_s1 = np.row_stack((scores_s1, scores_s1_f))

        boxes_s2, scores_s2 = im_detect(net, img, int( 0.75* target_size), transform, cuda, means)
        boxes_s2_f, scores_s2_f = flip_im_detect(net, img, int(0.75 * target_size), transform, cuda, means)
        boxes_s2 = np.row_stack((boxes_s2, boxes_s2_f))
        scores_s2 = np.row_stack((scores_s2, scores_s2_f))


        # enlarge: only detect small objects
        boxes3, scores3 = im_detect(net, img, int(1.75 * target_size), transform, cuda, means)
        boxes3_f, scores3_f = flip_im_detect(net, img, int(1.75 * target_size), transform, cuda, means)
        boxes3 = np.row_stack((boxes3, boxes3_f))
        scores3 = np.row_stack((scores3, scores3_f))
        index3 = np.where(np.maximum(boxes3[:, 2] - boxes3[:, 0] + 1, boxes3[:, 3] - boxes3[:, 1] + 1) < 128)[0]
        boxes3 = boxes3[index3]
        scores3 = scores3[index3]

        boxes4, scores4 = im_detect(net, img, int(1.5 * target_size), transform, cuda, means)
        boxes4_f, scores4_f = flip_im_detect(net, img, int(1.5 * target_size), transform, cuda, means)
        boxes4 = np.row_stack((boxes4, boxes4_f))
        scores4 = np.row_stack((scores4, scores4_f))
        index4 = np.where(np.maximum(boxes4[:, 2] - boxes4[:, 0] + 1, boxes4[:, 3] - boxes4[:, 1] + 1) < 192)[0]
        boxes4 = boxes4[index4]
        scores4 = scores4[index4]

        # boxes5, scores5 = im_detect(net, img, int(2.0 * target_size), transform, cuda, means)
        # boxes5_f, scores5_f = flip_im_detect(net, img, int(2.0 * target_size), transform, cuda, means)
        # boxes5 = np.row_stack((boxes5, boxes5_f))
        # scores5 = np.row_stack((scores5, scores5_f))
        # index5 = np.where(np.maximum(boxes5[:, 2] - boxes5[:, 0] + 1, boxes5[:, 3] - boxes5[:, 1] + 1) < 256)[0]
        # boxes5 = boxes5[index5]
        # scores5 = scores5[index5]

        if args.dataset == 'VOC':
            boxes = np.row_stack((boxes0,boxes_s1,boxes_s2,boxes3,boxes4))
            scores = np.row_stack((scores0,scores_s1,scores_s2,scores3,scores4))

        # scale each detection back up to the image
        # total_detect_time += detect_time
        # if i == 0:
        #     total_detect_time -= detect_time

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            # keep = nms(c_dets, 0.45, force_cpu=args.cpu)
            # c_dets = c_dets[keep, :]
            c_dets = bbox_vote(c_dets)
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()
        total_nms_time += nms_time
        if i == 0:
            total_nms_time -= nms_time

        if i % 20 == 0 and i != 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                  .format(i + 1, num_images, 0, nms_time))
            print('detect:%4f nms: %4f fps:%4f  %4f' % (total_detect_time, total_nms_time,
                                                        (i - 1) / (total_nms_time + total_detect_time),
                                                        (i - 1) / total_detect_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)


if __name__ == '__main__':
    # load net
    img_dim = (300, 512)[args.size == '512']
    num_classes = (21, 81)[args.dataset == 'COCO']
    net = build_ssd('test', img_dim, num_classes)  # initialize detector
    state_dict = torch.load(args.trained_model, map_location=lambda storage, loc: storage)
    # create new OrderedDict that does not contain `module.`

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    print('Finished loading model!')
    print(net)
    # load data
    if args.dataset == 'VOC':
        testset = VOCDetection(
            VOCroot, [('2007', 'test')], None, AnnotationTransform())
    elif args.dataset == 'COCO':
        testset = COCODetection(
            COCOroot, [('2014', 'minival')], None)
        # COCOroot, [('2017', 'test-dev')], None)
    else:
        print('Only VOC and COCO dataset are supported now!')
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    # evaluation
    top_k = (300, 200)[args.dataset == 'COCO']
    detector = Detect(num_classes, 0, cfg)
    save_folder = os.path.join(args.save_folder, args.dataset)
    rgb_means = ((104, 117, 123), (103.94, 116.78, 123.68))[args.version == 'RFB_mobile']
    multi_scale_test(save_folder, net, detector, args.cuda, testset,
                     BaseTransform(net.size, rgb_means, (2, 0, 1)),
                     rgb_means, top_k, thresh=0.01)
    # test_net(save_folder, net, detector, args.cuda, testset,
    #          BaseTransform(net.size, rgb_means, (2, 0, 1)),
    #          top_k, thresh=0.01)
