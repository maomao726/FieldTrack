import torch
import os.path as osp
import cv2
from yolox.utils import postprocess
from yolox.data.data_augment import preproc
from loguru import logger

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False,
        result_prepared=False
    ):
        self.result_prepared = result_prepared
        self.device = device
        if result_prepared:
            logger.info("Using prepared results.")
            return
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.fp16 = fp16
        # if trt_file is not None:
        #     from torch2trt import TRTModule

        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))

        #     x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
        #     self.model(x)
        #     self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer, img_path=None):
        assert (not self.result_prepared) or img_path is not None, "img_path should not be None when using prepared results."

        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        if self.result_prepared:
            detection_path = img_path[:-4] + ".txt"
            bboxes = []
            with open(detection_path, "r") as f:
                for line in f:
                    line = line.strip().split(" ")
                    line = [float(x) for x in line]
                    line = line + [1.0, 1.0, 0.0]
                    bboxes.append(line)
                f.close()
            if len(bboxes) != 0:
                bboxes = torch.tensor(bboxes, device=self.device, dtype=torch.float32)
            else:
                bboxes = None
            return [bboxes], img_info

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16
        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )

        if outputs[0] is not None:
                scale = min(self.test_size[0] / float(height), self.test_size[1] / float(width))
                outputs[0][:, :4] /= scale
        return outputs, img_info
