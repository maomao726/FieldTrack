import json
import os
import numpy as np
from loguru import logger


class RawVideoLabelParser():
    def __init__(self, raw_path):
        
        ## member variables
        self.frame_length = None    # total frame length
        self.frame_size = None      # frame size
        self.annotations = None     # labels {projection matrix(3D to 2D), player position, field gt, etc.}
        self.key_pt_dict = None     # keypoint labels
        self.num_cam = None         # number of cameras
        self.num_keypt = None       # number of keypoint
        self.homo_labels = None     # homography labels (2D to 2D)
        
        
        ## load labels {projection matrix(3D to 2D), player position, field gt, etc.}
        label_file_path = os.path.join(raw_path, "labels/label.json")
        label_fp = open(label_file_path, 'r')
        label_dict = json.load(label_fp)
        self.annotations = label_dict["annotations"]
        self.frame_length = len(self.annotations.keys())
        self.frame_size = tuple(label_dict["frame_size"])
        logger.info('[RawVideoLabelParser] : Parsing is done. Total Length: %d, frame size: '%(self.frame_length), self.frame_size, self.annotations['0'].keys())
        
        ## load keypoints labels
        
        

        try:
            key_pt_path = os.path.join(raw_path, "keypoints")
            self.key_pt_dict = {}
            for dirpth, _, _ in os.walk(key_pt_path):
                if dirpth == key_pt_path:
                    continue
                js = json.load(open(os.path.join(dirpth, "keypoints.json"), 'r'))
                self.key_pt_dict[dirpth.split('_')[-1]] = js
            self.num_cam = len(self.key_pt_dict.keys())
            self.num_keypt = len(self.key_pt_dict['0']['0'])
            logger.info('[RawVideoLabelParser] : Keypoint is loaded. Total Length: %d. Num of ref_points: %d'%(self.num_cam, self.num_keypt))
        except:
            logger.warning("keypoints directory is not exist. Are you running \"gen_keypoints.py\"?")
        
        ## load homography matrix label
        homo_path = os.path.join(raw_path, "labels/homography_matrix_2D_to_2D.json")
        if not os.path.exists(homo_path):
            logger.warning("homography_matrix_2D_to_2D is not exist.\n\
                            If you're training or evaluating, please make sure you've run \"to_2d_homography.py\".")
        else:
            homo_fp = open(homo_path, 'r')
            self.homo_labels = json.load(homo_fp)
            logger.info('[RawVideoLabelParser] : homography matrix are loaded, Len : %d, Cam : %d, each length : %d' \
                %(len(self.homo_labels), len(self.homo_labels['0']), len(self.homo_labels['0'])))

        if "visible" in self.annotations["0"]:
            logger.warning("Visible annotations are detected, will use these to select objects for metrics calculation")
        
        

    def getProjectionMatrix(self, cidx, fidx):
        '''
            args: 
                idx: target frame
            description: 
                projection matrix for each cam, global to local
            return:
                ndarray, shape: 3x4
        '''
        assert fidx < self.frame_length, "index %d is out of the bound, which is %d "%(fidx, self.frame_length)
        res = np.array(self.annotations[str(fidx)]['projection_matrix']["cam%d" % cidx], dtype=np.float32).reshape(3, 4)
        return res
    
    def getHomographyMatrix(self, cidx, fidx):
        '''
            args: 
                cidx: target camera
                fidx: target frame 
            description: 
                homography matrix , global to local
            return:
                ndarray, shape: 3x3
        '''
        return np.array(self.homo_labels[str(fidx)][str(cidx)]).reshape(3, 3)

    def getPlayerPosition(self, cidx, fidx):
        '''
            args: 
                idx: target frame
            description: 
                given camera index, frame index, return global pos and is visibility in frame or not
            return:
                (ndarray - shape: 22x3 , ndarray - shape 22x1)
        '''
        players_pos_global = []

        proj_mat = self.getProjectionMatrix(cidx, fidx)

        assert fidx < self.frame_length, "index %d is out of the bound, which is %d "%(fidx, self.frame_length)
        for _, player_pos in self.annotations[str(fidx)]['player_position'].items() :
            players_pos_global.append(np.array([player_pos[0], player_pos[1], 0, 1], dtype=np.float32))

        
            
        players_pos_global = np.stack(players_pos_global, axis = 0)

        players_pos_image = (proj_mat @ players_pos_global.T).T
        players_pos_image /= players_pos_image[:, 2:]


        
        ## debug
        # img = cv2.imread("/media/hcchen/data/data/dataset_.98/frames/cam_%d/cam%d_%05d.png"%(cidx, cidx, fidx))
        # for i in range(players_pos_image.shape[0]):
        #     img = cv2.circle(img, (int(players_pos_image[i, 0]), int(players_pos_image[i, 1])), 5, (0, 0, 255), -1)
        # cv2.imwrite("debug.jpg", img)
        
        if "visible" in self.annotations[str(fidx)]:
            is_in = np.array(self.annotations[str(fidx)]["visible"][str(cidx)])
        else:

            is_in = (0 <= players_pos_image[:, 0]) & \
                    (players_pos_image[:,0] < self.frame_size[0]) & \
                    (0 <= players_pos_image[:, 1]) & \
                    (players_pos_image[:, 1] < self.frame_size[1])
            
        

        return players_pos_global[:, [1, 0]], is_in
    
    
    def getKeypt(self, cam, idx):
        '''
            args:
                cam : target camera
                idx : target frame
            description:
                return the keypoint for given frame and camera
            return:
                ndarray, shape: NxHxW
        '''
        return np.array(self.key_pt_dict[str(cam)][str(idx)])


## Parser for synthetic video (may include multiple cameras) ##
class CuttedVideoLabelParser():
    def __init__(self, label_file_path):
        label_fp = open(label_file_path, 'r')
        label_dict = json.load(label_fp)
        self.annotations = label_dict["annotations"]
        self.video_count = len(self.annotations.keys())
        self.frame_size = tuple(label_dict["frame_size"])
        logger.info('[SytheticVideoLabelParser] : Parsing is done. Number of videos: %d, frame size: '%(self.video_count), self.frame_size, self.annotations.keys())
    
    def getFrameIdx(self, video_id):
        '''
            args:
                video_id : target video
            description:
                given video_index, return the camera index for each frames
            return:
                nparray len(video_length)
        '''
        information = self.annotations[video_id]
        result_cam = []
        for idx in range(len(information['cut_frames'])):
            result_cam = result_cam + [information['shot_list'][idx]] * ((information['video_length'] if (idx+1 == len(information['cut_frames'])) else information['cut_frames'][idx+1]) - information['cut_frames'][idx])
        assert len(result_cam) == information['video_length']
        
        res_frame = np.arange(information['start_frame_id'], information['start_frame_id']+information['video_length'])
        assert res_frame.shape[0] == information['video_length']
        
        
        return np.array([(result_cam[i], res_frame[i]) for i in range(information['video_length'])])
    def getVideoName(self, video_idx: int):
        '''
            args:
                video_idx : target video_index
            description:
                given video_index, return the video name
            return:
                string
        '''
        return self.annotations[str(video_idx)]['id']