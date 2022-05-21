import os
import os.path as osp
from yacs.config import CfgNode as CN
from KIE.utils.data_root import DATA_ROOT

# It uses yacs which does the job perfectly. For further info,
# check this out: https://github.com/rbgirshick/yacs/blob/master/yacs/config.py

_C = CN()

# ---------------------------------------------------------------------------- #
#                                    Model                                     #
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = "charlm_cnn_highway_lstm"
_C.MODEL.PARAMS = [
    ["char_embedding_dim", 15],
    ["char_conv_kernel_sizes", [1, 2, 3, 4, 5, 6]],
    ["char_conv_feature_maps", [25, 50, 75, 100, 125, 150]],  # [25 * kernel_size]
    ["use_batch_norm", True],
    ["num_highway_layers", 1],
    ["num_lstm_layers", 2],
    ["hidden_size", 300],
    ["dropout", 0.5],
    ["padding_idx", 0]
]

# ---------------------------------------------------------------------------- #
#                                    Loss                                      #
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.IGNORE_INDEX = -1

# ---------------------------------------------------------------------------- #
#                                  Dataset                                     #
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
# The train dataset list that specifies the path to folders.
_C.DATASET.TRAIN = [["data-dir", osp.join(DATA_ROOT, "kie_split")], ["split", "train"]]
# The validation dataset list that specifies the path to folders.
_C.DATASET.VAL = [["data-dir", osp.join(DATA_ROOT, "kie_split")], ["split", "val"]]
# The test dataset list that specifies the path to folders.
_C.DATASET.TEST = [["data-dir", osp.join(DATA_ROOT, "kie_split")], ["split", "test"]]
# The name of the dataset. By default it is the DATASET
_C.DATASET.NAME = "DATASET"
# The class names associated to this dataset.
_C.DATASET.CLASS_NAMES = [
    ["none", 0],
    ["company", 1],
    ["address", 2],
    ["date", 3],
    ["total", 4]
]
# The number of classes for this dataset.
_C.DATASET.NUM_CLASSES = 5

# ---------------------------------------------------------------------------- #
#                                  Dataloader                                  #
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.TRAINING = [
    ["batch_size", 10],
    ["shuffle", True],
    ["num_workers", 0],
    ["pin_memory", False],
    ["drop_last", False]
]
_C.DATALOADER.EVALUATION = [
    ["batch_size", 10],
    ["shuffle", False],
    ["num_workers", 0],
    ["pin_memory", False],
    ["drop_last", False]
]

# ---------------------------------------------------------------------------- #
#                                AdamW Solver                                  #
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.LR = 1e-3
_C.SOLVER.GAMMA = 0.5
_C.SOLVER.AMSGRAD = True
_C.SOLVER.MAX_EPOCHS = 25
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.LR_DECAY_STEPS = [20, 25]

# The output dir for all files generated during either training, evaluation or prediction.
_C.OUTPUT_DIR = os.getcwd() +"\\KIE\\outputs"