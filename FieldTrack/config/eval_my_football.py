
class Args:
    def __init__(self):
        self.name = None
        self.expn = None
        self.output_dir = "football"
        # distributed
        self.fp16 = True
        self.fuse = True
        self.trt = False
        self.exp_file = "exps/example/mot/yolox_x_mix_mot20_ch.py"
        
        # det args
        self.detection_prepared = False
        self.ckpt = "pretrained/ocsort_x_mot20.pth.tar"
        self.conf = 0.005
        self.nms = None
        self.tsize = None
        self.seed = None
        
        # tracking args
        self.track_thresh =  0.518
        self.dist_thresh = 50    # for mse, if dimstance > dist_thresh, then gain is set to -np.inf
        self.mse_tolerance = 20 # mse_tolerence, if mse > mse_tolerence, gain is set to 0
        self.min_hits = 1
        self.inertia = 0.128
        self.deltat = 1
        self.track_buffer = 30
        self.min_box_area = 10
        self.asso = "mse"
        self.use_byte = True
        self.use_app_embed = True
        self.weight_app_embed = 0.15
        self.app_embed_alpha = 0.33
        
        self.demo_type = "image"
        self.path = "/media/hcchen/307cc387-4594-4a7a-b426-1994a987f8c4/football_data/football_combined/cut"
        self.save_result = True
        self.save_frames = True
        self.show_frame = False
        self.aspect_ratio_thresh = 1.6
        self.device = "cuda"
        
        # for field detection
        self.field_pretrained = "/mnt/12cf47fc-dac6-4abb-a06f-014c3d2e7d30/football_extension/football_combined/training_0/best.pth"
        self.field_input_size = (256, 256)
        self.raw_video_path = "/media/hcchen/307cc387-4594-4a7a-b426-1994a987f8c4/football_data/football_combined"

    
        