import nni
from nni.utils import OptimizeMode
from nni.experiment import Experiment
import config


search_space = config.get_search_space("eval_my")
print(search_space)
experiment = Experiment('local')
experiment.config.experiment_name = 'OC_SORT'
experiment.config.trial_command = 'python eval_football_track.py --finetune'
experiment.config.trial_code_directory = '.'

experiment.config.search_space = search_space
experiment.config.max_trial_number = 500
experiment.config.trial_concurrency = 1
experiment.config.tuner.name = 'DNGO'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.run(8088)

experiment.stop()