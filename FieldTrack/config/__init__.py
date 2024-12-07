import config.eval_my as eval_my
from loguru import logger



def get_args(method, fintuning=None):

    if method == "eval_my":
        arg = eval_my.Args()

    else:
        raise ValueError("Invalid argument file")
    
    if fintuning:
        arg.save_frames = False
        arg.save_result = False
        for k in fintuning.keys():
            if hasattr(arg, k):
                setattr(arg, k, fintuning[k])
            else:
                logger.warning(f"Invalid parameter {k} in fintuning")
    return arg
        
    
def get_search_space(method):
    if method == "eval_my":
        return {
            "track_thresh": {'_type': 'uniform', '_value': [0.1, 0.9]},
            "dist_thresh": {'_type': 'uniform', '_value': [80, 120]},
            #"mse_tolerance": {'_type': 'uniform', '_value': [0, 0.3]},
            "min_hits": {'_type': 'randint', '_value': [1, 5]},
            "inertia": {'_type': 'uniform', '_value': [0.1, 0.9]},
            "deltat": {'_type': 'randint', '_value': [1, 5]},
            #"min_box_area": self.min_box_area,
            #"use_byte": {'_type': 'choice', '_value': [True, False]},
            "weight_app_embed": {'_type': 'uniform', '_value': [0.1, 0.5]},
            "app_embed_alpha": {'_type': 'uniform', '_value': [0.1, 0.5]}
        }

