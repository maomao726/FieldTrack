import numpy as np
import os
from .data.LabelParsing import RawVideoLabelParser, CuttedVideoLabelParser

class GroundTruthParser:
    def __init__(self, raw_path, cutted_path):
        self.raw_parser = RawVideoLabelParser(raw_path)
        self.cutted_parser = CuttedVideoLabelParser(os.path.join(cutted_path, "annotations.json"))


    def set_tracking_video(self, video_id):
        self.video_id = video_id
        self.frame_table = self.cutted_parser.getFrameIdx(video_id)
    
    def get_ground_truth(self, frame_id):
        frame_idx = self.frame_table[frame_id]
        pos, is_in = self.raw_parser.getPlayerPosition(frame_idx[0], frame_idx[1])
        
        # ground truth format: (x, y, id)
        # since the field is 720x1100, we need to scale the position from (-36 ~ 36, -55 ~ 55) to (0 ~ 720, 0 ~ 1100)
        pos[:, 0] = (pos[:, 0] + 36) * 10
        pos[:, 1] = (pos[:, 1] + 55) * 10
        
        is_in = np.nonzero(is_in)[0]
        
        gt = np.concatenate([pos[is_in], is_in[:, None]], axis=1)
        return gt
    
    def compare_with_gt(self, frame_id, tracking_result, max_dist = 20):
        '''
            tracking_result : [x, y, id]

            ** return **
            
            gt_id, pred_id, costMatrix
        '''
        gt = self.get_ground_truth(frame_id)
        # print(gt, tracking_result)
        # obj : true object
        # hyp : hypothesis object (tracking result)
        obj = np.atleast_2d(gt[:, :2]).astype(float)
        hyp = np.atleast_2d(tracking_result[:, :2]).astype(float)
        if obj.size == 0 and hyp.size == 0:
            return np.array([]), np.array([]), np.empty((0, 0))
        
        assert hyp.shape[1] == obj.shape[1], "Dimension mismatch"
        delta = obj[:, np.newaxis] - hyp[np.newaxis, :]
        C = np.sum(delta ** 2, axis=-1)
        C = np.sqrt(C)

        C[C > max_dist] = np.inf
        return gt[:, 2], tracking_result[:, 2], C