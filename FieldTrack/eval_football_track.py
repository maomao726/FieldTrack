import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
import pandas as pd
from loguru import logger
import logging


from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info
from trackers.my_tracker.ocsort import OCSort
from trackers.tracking_utils.timer import Timer

import json
from tqdm import tqdm
import pickle

from utils.Predictor import Predictor
from football_field.FieldRegister import FieldRegister as FieldRegister

# finetuning
from football_field.GroundTruthParser import GroundTruthParser
import motmetrics as mm


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
VIDEO_EXT = [".mp4", ".avi", ".mov", ".flv", ".mkv"]
from config import get_args
#from utils.args import make_parser


def list_video_files(path):
    video_files = []
    for f in os.listdir(path):
        file_path = osp.join(path, f)
        if osp.isdir(file_path) or osp.splitext(file_path)[-1].lower() in VIDEO_EXT:
            video_files.append(f)
    return video_files

def plot_tracking_field(field, tracks):
    '''
        field: warpped image, np.array, shape (720, 1100, 3)
        tracks: track after project, np.array, shape (N, 3~), each row include (x, y, tid, ...)
    '''

    result = field.copy()
    for track in tracks:
        result = cv2.circle(result, (int(track[1]), int(track[0])), 5, (0, 255, 0), -1)
        result = cv2.putText(result, str(int(track[2])), (int(track[1])-10, int(track[0])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result

def imageflow_eval(predictor, vis_folder, current_time, args):
    
    video_list = list_video_files(args.path)
    if len(video_list) == 0:
        logger.error(f"No video found in {args.path}")
        return
    
    register = FieldRegister(args.field_pretrained, args.field_input_size, args.device)  
    gtparser = GroundTruthParser(args.raw_video_path, os.path.join(args.path))
    mot_acc = mm.MOTAccumulator(auto_id=False)
    mh = mm.metrics.create()
    summaries = None

    ## cache root
    cache_root = osp.join('football/cache', f"{osp.basename(args.path)}")
    if not osp.exists(cache_root):
        os.makedirs(cache_root, exist_ok=True)
    
    for video_basename in video_list:
        video = osp.join(args.path, video_basename)
        gtparser.set_tracking_video(osp.basename(video).split('.')[0])
        mot_acc.reset()
        
        tracker = OCSort(det_thresh=args.track_thresh, use_byte=args.use_byte,
                        dist_threshold=args.dist_thresh, mse_tolerence=args.mse_tolerance,
                        min_hits=args.min_hits, delta_t=args.deltat, 
                        asso_func=args.asso, inertia=args.inertia, 
                        use_app_embed=args.use_app_embed, weight_app_embed=args.weight_app_embed,
                        app_embed_alpha=args.app_embed_alpha)
        
        video_len = 0
        if args.demo_type == "video":
            cap = cv2.VideoCapture(video)
            video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif args.demo_type == "image":
            frame_list = sorted([i for i in os.listdir(video) if osp.splitext(i)[-1].lower() in IMAGE_EXT])
            video_len = len(frame_list)

    
        logger.info(f"Processing video: {video_basename}")
        timer = Timer()

        ## detection cache
        results = []
        
        for frame_id in range(video_len):
            frame_path = osp.join(video, frame_list[frame_id])
            frame = cv2.imread(frame_path)

            outputs, _ = predictor.inference(frame, timer, frame_path)
            
            outputs, reg_res = register.inference(frame, outputs)
            # for p in outputs[0]:
            #     cv2.circle(reg_res['warpped'], (int(p[1]), int(p[0])), 5, (0, 255, 0), -1)
            # cv2.imshow("frame", reg_res['warpped'])
            # print(outputs)
            # cv2.waitKey(0)
            #det_res.append([output.detach().cpu().numpy() for output in outputs])
            
                
            #outputs, _ = predictor.inference(frame, timer)
            
            
            # img_debug = frame.copy()
            # field_debug = reg_res['warpped'].copy()
            # for row in range(outputs[0].shape[0]):
            #     if outputs[0][row, 2] * outputs[0][row, 3] > 0.1:
            #         img_debug = cv2.rectangle(img_debug, (int(outputs[1][row, 0]), int(outputs[1][row, 1])), 
            #                                               (int(outputs[1][row, 2]), int(outputs[1][row, 3])), (0, 255, 0), 2)
            #         img_debug = cv2.putText(img_debug, f"{outputs[0][row, 2] * outputs[0][row, 3]:.2f}", (int(outputs[1][row, 0]) - 5, int(outputs[1][row, 1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #         field_debug = cv2.circle(field_debug, (int(outputs[0][row, 1]), int(outputs[0][row, 0])), 5, (0, 255, 0), -1)
            #         field_debug = cv2.putText(field_debug, f"{outputs[0][row, 2] * outputs[0][row, 3]:.2f}", (int(outputs[0][row, 1]) - 5, int(outputs[0][row, 0]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.imwrite("debug.jpg", img_debug)
            # cv2.imwrite("field_debug.jpg", field_debug)
            # breakpoint()
            
            online_targets = tracker.update(outputs, frame, register.field_size, reg_res['warpped'])
            online_pos = []
            online_ids = []
            
            gt_id, hyp_id, C = gtparser.compare_with_gt(frame_id, online_targets)
            mot_acc.update(gt_id, hyp_id, C, frameid=frame_id)
            
            for t in online_targets:
                p = [t[0], t[1]]
                tid = t[2]
                # vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                # if p[0] * p[1] > args.min_box_area:
                online_pos.append(p)
                online_ids.append(tid)
                results.append(
                    f"{frame_id},{tid},{p[0]:.2f},{p[1]:.2f}\n"
                )
            timer.toc()


            if args.save_frames:
                online_im = reg_res['warpped'].copy()
                #online_im = plot_tracking_field(reg_res['warpped'], online_targets)
                save_folder = osp.join(vis_folder, video_basename.split('.')[0])
                if not osp.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(osp.join(save_folder, f"{frame_id}.jpg"), online_im)

                if args.show_frame:
                    cv2.imshow("ori", frame)
                    cv2.imshow("online", online_im)
                    cv2.waitKey(10)

            frame_id += 1 
    
        if args.demo_type == "video":
            cap.release()
        
        ## save cache
        # with open(cache_path, 'wb') as f:
        #     pickle.dump(det_res, f)
        # f.close()

        summary = mh.compute(mot_acc, metrics=mm.metrics.motchallenge_metrics, name=osp.basename(video).split('.')[0])
        
        if summaries is None:
            summaries = summary
        else:
            summaries = pd.concat([summaries, summary], axis=0)
        
        if args.save_result:
            res_file = osp.join(vis_folder, f"{video_basename.split('.')[0]}.txt")
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")
    
    # return summaries_stats
    summaries_stats = {
        'idf1' : 2 * summaries['idtp'].sum() / (2 * summaries['idtp'].sum() + summaries['idfp'].sum() + summaries['idfn'].sum()),
        'mota' : 1 - (summaries['num_false_positives'].sum() + summaries['num_misses'].sum() + summaries['num_switches'].sum()) / summaries['num_objects'].sum(),
        'fp' : summaries['num_false_positives'].sum(),
        'fn' : summaries['num_misses'].sum(),
        'idsw' : summaries['num_switches'].sum(),
        'gt' : summaries['num_objects'].sum(),
        'idp' : summaries['idtp'].sum() / (summaries['idtp'].sum() + summaries['idfp'].sum()),
        'idr' : summaries['idtp'].sum() / (summaries['idtp'].sum() + summaries['idfn'].sum())
    }
    print(summaries_stats)
    return summaries_stats

def main(exp, args):
    if not args.expn:
        args.expn = exp.exp_name

    args.trt = False
    output_dir = osp.join(args.output_dir, os.path.basename(args.path))
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir)
    if args.save_result:
        os.makedirs(vis_folder, exist_ok=True)


    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    trt_file = None
    decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16, args.detection_prepared)
    current_time = time.localtime()
    
    return imageflow_eval(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    #print("----")
    #args = make_parser().parse_args()
    parser = argparse.ArgumentParser("OC-SORT parameters")
    parser.add_argument("--finetune", action="store_true", help="finetune the model")
    finetuning = parser.parse_args().finetune
    if not finetuning:
        args = get_args("eval_my")
    else:
        import nni
        tuner_params = nni.get_next_parameter()
        print(tuner_params)
        args = get_args("eval_my", tuner_params)
        
    exp = get_exp(args.exp_file, args.name)
    
    
    ## single run
    result = main(exp, args)
    if finetuning:
        #_logger.info(f"mota: {result['mota']}")
        nni.report_final_result(result['mota'])
    else:
        print("--result--")
        print("data: ", osp.basename(args.path))
        print("model: ", args.ckpt)
        for k, v in result.items():
            print(f"{k}: {v}")
