import numpy as np
import os
from .data.LabelParsing import RawVideoLabelParser, CuttedVideoLabelParser
import cv2

class GroundTruthParser:
    def __init__(self, raw_path, cutted_path):
        self.raw_parser = RawVideoLabelParser(raw_path)
        self.cutted_parser = CuttedVideoLabelParser(os.path.join(cutted_path, "annotations.json"))

    def get_gt_video_num(self):
        return self.cutted_parser.video_count

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
        


        tracking_result_in_field = (-72 <= tracking_result[:, 0]) & (tracking_result[:, 0] < 720) & (-110 <= tracking_result[:, 1]) & (tracking_result[:, 1] < 1100)
        tracking_result_in_field = np.nonzero(tracking_result_in_field)[0]
        tracking_result = tracking_result[tracking_result_in_field]
        
        #print(tracking_result)
        gt_in_field = (-72 <= gt[:, 0]) & (gt[:, 0] < 720) & (-110 <= gt[:, 1]) & (gt[:, 1] < 1100)
        gt_in_field = np.nonzero(gt_in_field)[0]
        gt = gt[gt_in_field]

        ##debug
        # debug_img = np.zeros((1100, 720, 3), dtype=np.uint8)
        # for i in range(gt.shape[0]):
        #     cv2.circle(debug_img, (int(gt[i, 0]), int(gt[i, 1])), 5, (0, 255, 0), -1)
        #     cv2.putText(debug_img, str(int(gt[i, 2])), (int(gt[i, 0]), int(gt[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # for i in range(tracking_result.shape[0]):
        #     cv2.circle(debug_img, (int(tracking_result[i, 0]), int(tracking_result[i, 1])), 5, (0, 0, 255), -1)
        #     cv2.putText(debug_img, str(int(tracking_result[i, 2])), (int(tracking_result[i, 0]), int(tracking_result[i, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # cv2.imshow("debug", debug_img)
        # print(frame_id)
        # cv2.waitKey(0)
            

        #breakpoint()
        obj = np.atleast_2d(gt[:, :2]).astype(float)
        hyp = np.atleast_2d(tracking_result[:, :2]).astype(float)

        

        
        if obj.size == 0 and hyp.size == 0:
            return np.array([]), np.array([]), np.empty((0, 0))
        
        assert hyp.shape[1] == obj.shape[1], "Dimension mismatch"
        delta = obj[:, np.newaxis] - hyp[np.newaxis, :]
        C = np.sum(delta ** 2, axis=-1)
        C = np.sqrt(C)

        C[C > max_dist] = np.inf
        #breakpoint()
        return gt[:, 2], tracking_result[:, 2], C