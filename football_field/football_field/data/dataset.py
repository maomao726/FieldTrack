'''
    VideoDataset
        Dataset for input videoes.

'''

from torch.utils.data import Dataset
from .LabelParsing import RawVideoLabelParser
import torchvision.transforms.v2 as transforms
from utils import Instance
import numpy as np
import cv2
import torch
import os
from torchvision import tv_tensors
from torch.nn.functional import interpolate
import random
    
## FieldDataset is used for training and testing the keypoint model and the refine model
class FieldDataset(Dataset):
    def __init__(self, opt : Instance, data_idx : list):
        '''
    
        data_idx:
            "init_estimation" :
                list of index of frames
                [0] : camera index
                [1] : frame index
            "refinement" :
                list of index of frames
                [0] : camera index
                [1] : [frame index, frame index]
        
            
        datatype:
            "training" or "testing"
        
        loading_mode:
            "init_estimation" or "refinement"
        '''
        super(FieldDataset, self).__init__()
        self.data_idx = data_idx
        self.opt = opt
        self.raw_parser = RawVideoLabelParser(self.opt.raw_path)
        self.X_meshgrid = None
        self.Y_meshgrid = None
        
        self.transform = {
            "kp_model_training" : transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomResizedCrop(size=(self.opt.img_size, self.opt.img_size), scale=(0.8, 1.2), ratio=(0.8, 1.2), antialias=True),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5)
            ]),
            "else" : transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(size=(self.opt.img_size, self.opt.img_size), antialias=True)
            ]),
            "Normalize" : transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        }
        self.data = {}
        for i in range(self.raw_parser.num_cam):
            self.data[i] = {}
            for j in range(self.raw_parser.frame_length):
                self.data[i][j] = None
        
        ## meshgrid for keypoint heatmap generation
        ## generate X/Y meshgrid
        x = np.linspace(0, self.raw_parser.frame_size[0], self.raw_parser.frame_size[0])
        y = np.linspace(0, self.raw_parser.frame_size[1], self.raw_parser.frame_size[1])
        x, y = np.meshgrid(x, y)
        self.X_meshgrid = torch.from_numpy(x).to(self.opt.device)
        self.Y_meshgrid = torch.from_numpy(y).to(self.opt.device)
          
    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, index):
        cam = self.data_idx[index][0]
        idx = self.data_idx[index][1]

        # randomly select source frame for refinement
        if self.opt.loading_mode == "refinement":
            idx = [random.randint(max(idx - 32, 0), idx - 1), idx]
        
        img = []
        img_origin = None

        if self.opt.loading_mode == "init_estimation":
            for i in range(idx-self.opt.n_input+1, idx+1):
                img_tmp = cv2.imread(os.path.join(self.opt.raw_path, "frames", "cam_%d"%(cam), "cam%d_%05d.png"%(cam, i)))
                
                if self.opt.type == "testing" and i == idx:
                    img_origin = cv2.resize(img_tmp, (self.opt.img_size, self.opt.img_size))
                    
                img_tmp = torch.Tensor(img_tmp).type(torch.float32).permute(2, 0, 1)
                img_tmp = self.transform["Normalize"](img_tmp)
                img.append(img_tmp)
                
        elif self.opt.loading_mode == "refinement":
            for i in idx:
                img_tmp = cv2.imread(os.path.join(self.opt.raw_path, "frames", "cam_%d"%(cam), "cam%d_%05d.png"%(cam, i)))
                
                if self.opt.type == "testing" and i == idx[1]:
                    img_origin = cv2.resize(img_tmp, (self.opt.img_size, self.opt.img_size))
                
                img_tmp = torch.Tensor(img_tmp).type(torch.float32).permute(2, 0, 1)
                img_tmp = self.transform["Normalize"](img_tmp)
                img.append(img_tmp)

        img = torch.vstack(img)
        
        if self.opt.loading_mode == "init_estimation":           
            heatmaps = self.get_heatmaps_from_keypt(cam, idx)
            if self.opt.type == "testing":
                heatmaps = interpolate(heatmaps.unsqueeze(0), size=(self.opt.img_size, self.opt.img_size), mode='bilinear', align_corners=True).type(torch.float32).squeeze(0)
                img = self.transform["else"](img)
                return img.to(self.opt.device), heatmaps.to(self.opt.device), img_origin
            else:
                
                heatmaps = tv_tensors.Mask(heatmaps, dtype=torch.float32)


                img, heatmaps = self.transform["kp_model_training"](img, heatmaps)
                return img.to(self.opt.device), heatmaps.to(self.opt.device)
            
        elif self.opt.loading_mode == "refinement":
            heatmap_from = self.get_heatmaps_from_keypt(cam, idx[0]).unsqueeze(0)
            heatmap_from = interpolate(heatmap_from, size=(self.opt.img_size, self.opt.img_size), mode='bilinear', align_corners=True).type(torch.float32).squeeze(0)
            heatmap_to = self.get_heatmaps_from_keypt(cam, idx[1]).unsqueeze(0)
            heatmap_to = interpolate(heatmap_to, size=(self.opt.img_size, self.opt.img_size), mode='bilinear', align_corners=True).type(torch.float32).squeeze(0) 
            
            img = self.transform["else"](img)

            if self.opt.type == "training":

                return img.to(self.opt.device), heatmap_from.to(self.opt.device), heatmap_to.to(self.opt.device)
            else:
                return img.to(self.opt.device), heatmap_from.to(self.opt.device), heatmap_to.to(self.opt.device), img_origin

    def get_heatmaps_from_keypt(self, cam, idx):
        kpt = torch.from_numpy(self.raw_parser.getKeypt(cam, idx)).to(self.opt.device)
        
        std = torch.Tensor([2.5, 2.5]).to(self.opt.device)
        heatmaps = torch.exp(-((self.X_meshgrid - kpt[:, 0][:, None, None])**2 / (2 * std[0]**2) + 
                    (self.Y_meshgrid - kpt[:, 1][:, None, None])**2 / (2 * std[1]**2))).type(torch.float32)
        
        return heatmaps
        

## VideoDataset is used on videos, which may includes camera switching.
# class VideoDataset(Dataset):
#     def __init__(self, opt):
#         '''
#             arg:
#                 opt : {
#                     raw_path : path of annotations of raw video
#                     raw_label_name : name of annotations of raw video label
#                     video_path : path of annotations of sythetic video label
#                     video_label_name : name of annotations of sythetic video label
#                     is_training : is in the training mode
#                     courtMask_frames_path : path for masked court images
#                     playerDetected_frames_path : path for player detection images
#                 }
#         '''
#         self.opt = opt
#         print("[VideoDataset] : Dataset Creating. Config: ", vars(self.opt))
#         if self.opt.is_training:
#             self.raw_parser = RawVideoLabelParser(self.opt.raw_path)
#         self.video_parser = SytheticVideoLabelParser(self.opt.video_path)
        
#         self.tmp_path = "./tmp/"
#         os.makedirs("./tmp/")
#         print("[VideoDataset] : Dataset Created.")

    
#     def __getitem__(self, index):
#         '''
#             arg:
#                 index: video index
#             description:
#                 training : return (seg_mask+player_detection, isCut, proj_mat, pos, videoName)
#                 not training : return (seg_mask+player_detection, isCut, videoName)
#             return:
#                 self.opt.is_training -> Tx3xWxH, Tx1, Tx3xW_fieldxH_field, Tx22x2, Tx22x1
#                 else -> Tx3xWxH, Tx2, str
#         '''
        
#         frame_input = []

#         camera_list = self.video_parser.getCameraIdx(index)
#         frame_list = self.video_parser.getFrameIdx(index)
#         videoName = self.video_parser.getVideoName(index)

#         # scene feature, Tx1
#         cut_input = torch.Tensor([0] + [0 if camera_list[i] == camera_list[i-1] else 1 for i in range(1, len(camera_list))]).reshape([-1,1]).type(torch.bool)
        
#         # frames feature, Tx3xWxH
#         # check if the feature is already generated
#         if os.path.exists(self.tmp_path + "frames_%s.pt"%(videoName)):
#             frame_input = torch.load(self.tmp_path + "frames_%s.pt"%(videoName))
#         else:
#             for cidx, fidx in zip(camera_list, frame_list):
#                 # frames feature
#                 mask_path = os.path.join(self.opt.courtMask_frames_path , "cam_%d/cam%d_%05d.png"%(cidx, cidx, fidx))
#                 detection_path = os.path.join(self.opt.playerDetected_frames_path , "cam_%d/bbox_cam%d_%05d.png"%(cidx, cidx, fidx))
                
#                 # prework loading, should be part of pipeline and doesn't need to be loaded in final version.
#                 mask = cv2.imread(mask_path)
#                 detection = cv2.imread(detection_path)
                
#                 assert mask is not None, "mask image %s not found"%(mask_path)
#                 assert detection is not None, "detection image %s not found"%(detection_path)
                
#                 frame_feature = img_preprossing(mask, detection, self.opt.img_size)
#                 #print(frame_feature.shape)
#                 frame_input.append(frame_feature)

#             frame_input = torch.stack(frame_input, dim=0)
#             torch.save(frame_input, self.tmp_path + "frames_%s.pt"%(videoName))

#         if not self.opt.is_training:
#             return frame_input, cut_input, videoName
        
#         else: # training, load ground truth
#             proj_mask = []
#             player_position = []
#             is_in_image = []
            
#             for cidx, fidx in zip(camera_list, frame_list):
#                 # load ground truth
#                 proj_mask.append(self.raw_parser.getFieldGT(fidx=fidx, cidx=cidx).transpose(2, 0, 1))
#                 player_position.append(self.raw_parser.getPlayerPositionInGlobal(idx = fidx)[:,:2])
#                 is_in_image.append(self.raw_parser.getPlayerPositionInImage(idx = fidx, cam_idx = cidx)[1])

#             ## position normalization
#             player_position = torch.tensor(np.stack(player_position, axis = 0))
#             player_position = torch.div(player_position, config.field_size)


#             return frame_input, \
#                     cut_input, \
#                     torch.tensor(np.stack(proj_mask, axis = 0)), \
#                     torch.tensor(np.stack(player_position, axis = 0)),\
#                     torch.Tensor(np.stack(is_in_image, axis = 0)).type(torch.bool)


#     def __len__(self):
#         '''
#             return:
#                 num of videos in this dataset
#         '''
#         return self.video_parser.video_count
    
#     def __del__(self):
#         '''
#             delete the tmp folder
#         '''
#         if os.path.exists(self.tmp_path):
#             os.system("rm -r %s"%(self.tmp_path))
#             print("[VideoDataset] : tmp folder deleted.")
#         return
