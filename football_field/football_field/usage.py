from .model.model import ResNetUNet
import torchvision.transforms as transforms
import torch
from torch.nn.functional import interpolate
import numpy as np
import cv2
from .data.LabelParsing import RawVideoLabelParser

def field_create_model(pretrained_pth, device):
    return ResNetUNet(pretrained_pth).to(device)

def field_preprocessor(input_size):
    return transforms.Compose([
        transforms.Resize(input_size),
        #transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

@torch.no_grad()
def field_inference(model, preprocessor, image, device):
    img_shape = image.shape
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    image = preprocessor(image)
    image = image.unsqueeze(0).to(device)
    model.eval()

    output = model(image)
    output = interpolate(output, size=(img_shape[0], img_shape[1]), mode='bilinear', align_corners=False)
    return output.squeeze(0)

@torch.no_grad()
def field_postprocessing(output, conf_threshold = 0.01):
    '''
    filtering keypoints from pred
        pred : (C, H, W)
        threshold : float

    return : 
        np.ndarray : (n, 2), where n is the number of keypoints
        float : confidence score
    '''
    pt_filter = output > conf_threshold
    suc_count = 0
    pt_list = []

    confidence_score = 0
    for h in range(pt_filter.shape[0]):
        indices = torch.nonzero(pt_filter[h])
        
        if indices.nelement() != 0:
            
            # find center by weighted average
            pt_x = torch.sum(indices[:, 0].float() * output[h, indices[:, 0], indices[:, 1]]) / torch.sum(output[h, indices[:, 0], indices[:, 1]])
            pt_y = torch.sum(indices[:, 1].float() * output[h, indices[:, 0], indices[:, 1]]) / torch.sum(output[h, indices[:, 0], indices[:, 1]])
            #pt_max = torch.argmax(output[h])
            #pt_x, pt_y = pt_max // output.shape[2], pt_max % output.shape[2]
            pt_list.append(np.array([pt_x.item(), pt_y.item()]))
            #print(h, pt_x, pt_y, output[h, pt_x, pt_y])
            std = torch.sqrt(torch.sum((indices[:, 0] - pt_x)**2 + (indices[:, 1] - pt_y)**2) / indices.shape[0])
            confidence_score += 1 / (1 + std)
            suc_count += 1
        else:
            pt_list.append(np.array([-100, -100]))
    

    assert  suc_count >= 4, f"Less than 4 keypoints detected, suc_count = {suc_count}"
    return np.array(pt_list), confidence_score  * (suc_count >= 4) / suc_count


def field_get_homography(
        pt_on_img : np.ndarray,
        pt_on_field : np.ndarray,
):
    '''
    get homography matrix
        pt_on_img : np.ndarray, (n, 2), where n is the number of keypoints
        pt_on_field : np.ndarray, (n, 2), where n is the number of keypoints
        ori_size : tuple, (H, W), original image size
        field_size : tuple, (H, W), field size

    return : 
        np.ndarray : homography matrix
    '''
    
    valid_pt_mask = pt_on_img[:, 0] >= 0
    pt_on_img = pt_on_img[valid_pt_mask]
    pt_on_field = pt_on_field[valid_pt_mask]
    # print(pt_on_field, pt_on_img)
    H, _ = cv2.findHomography(pt_on_img, pt_on_field, cv2.RANSAC, 6.0)
    # breakpoint()
    return H

def football_data_labelParser(raw_datapath : str):
    return RawVideoLabelParser(raw_datapath)