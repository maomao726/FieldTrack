import numpy as np
import cv2
import torch
from football_field.usage import field_create_model, field_preprocessor, field_inference, field_postprocessing, field_get_homography


class FieldRegister:
    field_size = (720, 1100)
    template = np.zeros((field_size[0], field_size[1], 3), dtype=np.uint8)
    
    def __init__(
        self,
        pretrained_pth,
        input_size,
        device
    ):
        self.device = device
        self.model = field_create_model(pretrained_pth, self.device).eval()
        self.preprocessor = field_preprocessor(input_size)

        kp_x, kp_y = np.meshgrid(np.arange(0, self.field_size[1]+1, self.field_size[1] / 10), 
                                 np.arange(0, self.field_size[0]+1, self.field_size[0] / 6), indexing='ij')
        self.field_keypoints = np.stack([kp_y.flatten(), kp_x.flatten(), np.ones_like(kp_x.flatten())], axis=-1)
        img_grid = np.zeros((self.field_size[0], self.field_size[1], 3), dtype=np.uint8)
        for i in range(self.field_keypoints.shape[0]):
            x, y, _ = self.field_keypoints[i]
            img_grid = cv2.putText(img_grid, f"{i}", (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite("field_keypoints.jpg", img_grid)

    def inference(self, frame, pred_result, exp_size):
        '''
        args:
        frame: input frame, np.ndarray, shape (H, W, 3)
        pred_result: prediction result from the object detection model, list of np.ndarray, each with shape (N, 7)
        exp_size: expected size of the frame, tuple of int, (H, W)
        
        return:
        proj_result : list of np.ndarray
            [0] : projected detection result, np.ndarray, shape (N, 5)
            [1] : predicted bbox (rescaled), np.ndarray, shape (N, 4)
        register_result : dict
            "heatmaps" : heatmaps of keypoint detection model, np.ndarray, shape (77, 256, 256)
            "keypoints" : detected keypoints, np.ndarray, shape (77, 2)
            "conf_score" : confidence score of the keypoints, float
            "homography" : homography matrix, np.ndarray, shape (3, 3)
            "warpped" : warpped frame, np.ndarray, shape (field_height, field_width, 3)
        '''
        
        
        result = {}
        result["heatmaps"] = field_inference(self.model, self.preprocessor, frame, self.device)
        result["keypoints"], result["conf_score"] = field_postprocessing(result["heatmaps"])
        result["homography"] = field_get_homography(result["keypoints"], self.field_keypoints)

        warpped = cv2.transpose(frame)
        warpped = cv2.warpPerspective(warpped, result["homography"], (self.field_size[0], self.field_size[1]))
        warpped = cv2.transpose(warpped)
        result["warpped"] = warpped

        proj_result = [None for _ in pred_result] if pred_result is not None else None

        #Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        if pred_result is not None:
            # scale
            detections = pred_result[0][:, :4]
            img_h, img_w = frame.shape[:2]
            #scale = torch.tensor([exp_size[0] / float(img_h), exp_size[1] / float(img_w)], dtype=torch.float32).to(self.device)
            scale = min(exp_size[0] / float(img_h), exp_size[1] / float(img_w))
            detections /= scale
            bot_center = torch.stack([detections[:, 3], (detections[:, 0] + detections[:, 2]) / 2, torch.ones_like(detections[:, 0])], dim=1)
            # for d in bot_center:
            #     frame = cv2.circle(frame, (int(d[1]), int(d[0])), 5, (0, 255, 0), -1)
            # cv2.imwrite("debug_ori.jpg", frame)
            bot_center = bot_center @ torch.from_numpy(result["homography"]).float().to(self.device).t()
            bot_center = bot_center[:, :2] / bot_center[:, 2:]

            # for i, detection in enumerate(bot_center):
            #     warpped = cv2.circle(warpped, (int(detection[1]), int(detection[0])), 5, (0, 255, 0), -1)
            # cv2.imwrite("debug.jpg", warpped)
            bot_center = torch.cat([bot_center[:], pred_result[0][:, 4:]], dim=1)
            proj_result[0] = bot_center
            proj_result.append(detections)

        return proj_result, result
    

class FieldRegister_AfterTrack:
    """
    FieldRegister class for the post-tracking stage.
    This class is used to project the tracking result to the football field.
    """
    field_size = (720, 1100)
    template = np.zeros((field_size[0], field_size[1], 3), dtype=np.uint8)
    
    def __init__(
        self,
        pretrained_pth,
        input_size,
        device
    ):
        self.device = device
        self.model = field_create_model(pretrained_pth, self.device).eval()
        self.preprocessor = field_preprocessor(input_size)

        kp_x, kp_y = np.meshgrid(np.arange(0, self.field_size[1]+1, self.field_size[1] / 10), 
                                 np.arange(0, self.field_size[0]+1, self.field_size[0] / 6), indexing='ij')
        self.field_keypoints = np.stack([kp_y.flatten(), kp_x.flatten(), np.ones_like(kp_x.flatten())], axis=-1)
        img_grid = np.zeros((self.field_size[0], self.field_size[1], 3), dtype=np.uint8)
        for i in range(self.field_keypoints.shape[0]):
            x, y, _ = self.field_keypoints[i]
            img_grid = cv2.putText(img_grid, f"{i}", (int(y), int(x)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite("field_keypoints.jpg", img_grid)

    def inference(self, frame, pred_result, exp_size = None):
        '''
            * frame format : np.ndarray, (w, h, 3), color format : BGR
            * pred_result : list of np.ndarray, (N, 4~)

                **list length is only 1**
                array content : [x1, y1, x2, y2, (anything else)]
                
                    x for width, y for height
                ```
            ** return **
            proj_result, result
            * proj_result : same format as result, but shape as (N, 2~)
                array content : [x, y, (anything else)]
                
                    x for court_length, y for court_width

        '''
        
        result = {}
        result["heatmaps"] = field_inference(self.model, self.preprocessor, frame, self.device)
        result["keypoints"], result["conf_score"] = field_postprocessing(result["heatmaps"])
        result["homography"] = field_get_homography(result["keypoints"], self.field_keypoints)

        warpped = cv2.transpose(frame)
        warpped = cv2.warpPerspective(warpped, result["homography"], (self.field_size[0], self.field_size[1]))
        warpped = cv2.transpose(warpped)
        result["warpped"] = warpped

        #Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        pred_to_field = [None for _ in pred_result]
        if pred_result is not None:
            # scale
            detections = pred_result[0][:, :4]
            
            img_h, img_w = frame.shape[:2]
            #scale = torch.tensor([exp_size[0] / float(img_h), exp_size[1] / float(img_w)], dtype=torch.float32).to(self.device)
            # scale = min(exp_size[0] / float(img_h), exp_size[1] / float(img_w))
            # detections /= scale
            bot_center = np.stack([detections[:, 3], (detections[:, 0] + detections[:, 2]) / 2, np.ones((detections.shape[0]))], axis=1)

            # for d in bot_center:
            #     frame = cv2.circle(frame, (int(d[1]), int(d[0])), 5, (0, 255, 0), -1)
            # cv2.imwrite("debug_ori.jpg", frame)
            bot_center = bot_center @ result["homography"].T
            bot_center = bot_center[:, :2] / bot_center[:, 2:]

            for i, detection in enumerate(bot_center):
                warpped = cv2.circle(warpped, (int(detection[1]), int(detection[0])), 5, (0, 255, 0), -1)
            cv2.imwrite("debug_ori.jpg", warpped)
            cv2.waitKey(3)

            bot_center = np.concatenate([bot_center[:], pred_result[0][:, 4:]], axis=1)
            pred_to_field[0] = bot_center

        return pred_to_field, result