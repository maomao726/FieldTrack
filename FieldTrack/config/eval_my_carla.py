
class Args:
    def __init__(self):
        self.name = None
        self.expn = None
        self.output_dir = "football"
        # distributed
        self.fp16 = True
        self.fuse = True
        self.trt = False
        self.exp_file = "exps/example/mot/yolox_dancetrack_test.py"
        
        # det args
        self.detection_prepared = True
        self.ckpt = "pretrained/ocsort_dance_model.pth.tar"
        self.conf = 0.005
        self.nms = 0.7
        self.tsize = None
        self.seed = None
        
        # tracking args
        self.track_thresh =  0.174
        self.dist_thresh = 100    # for mse, if dimstance > dist_thresh, then gain is set to -np.inf
        self.mse_tolerance = 20 # mse_tolerence, if distance > mse_tolerence, gain is set to 0
        self.min_hits = 1
        self.inertia = 0.9
        self.deltat = 3
        self.track_buffer = 30
        self.min_box_area = 10
        self.asso = "mse"
        self.use_byte = True
        self.use_app_embed = True
        self.weight_app_embed = 0.1
        self.app_embed_alpha = 0.1
        
        self.demo_type = "image"
        self.path = "/media/hcchen/data/data/dataset_carla_combined/cut"
        self.save_result = True
        self.save_frames = False
        self.show_frame = False
        self.aspect_ratio_thresh = 1.6
        self.device = "cpu"
        
        # for field detection
        self.field_pretrained = "/media/hcchen/data/football_extension/carla/training/combined_0/best.pth"
        self.field_input_size = (256, 256)
        self.raw_video_path = "/media/hcchen/data/data/dataset_carla_combined"

        

    
        