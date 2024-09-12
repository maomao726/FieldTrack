
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
        self.ckpt = "pretrained/ocsort_dance_model.pth.tar"
        self.conf = None
        self.nms = None
        self.tsize = None
        self.seed = None
        
        # tracking args
        self.track_thresh =  0.518
        self.dist_thresh = 50    # for mse, if dimstance > dist_thresh, then gain is set to -np.inf
        self.mse_tolerance = 1 # mse_tolerence, if mse > mse_tolerence, gain is set to 0
        self.min_hits = 3
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
        self.path = "/media/hcchen/data/data/football_data/videoData/motions/slowest"
        self.save_result = True
        self.save_frames = False
        self.aspect_ratio_thresh = 1.6
        self.device = "cuda"
        
        # for field detection
        self.field_pretrained = "/media/hcchen/data/FieldTrack/football_field/football_field/best.pth"
        self.field_input_size = (256, 256)
        self.raw_video_path = "/media/hcchen/data/data/football_data/maincam_momentum"

    
        