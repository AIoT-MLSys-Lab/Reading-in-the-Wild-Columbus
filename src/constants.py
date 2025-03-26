from enum import Enum
from pathlib import Path


class ModelNames(Enum):
    pass

model_names = [f.stem for f in Path('../models').glob('*.pt')]
MODELS = Enum('ModelNames', model_names)

COMBINATIONS = {
    'gaze_only': (True, False, False),
    'imu_only': (False, True, False),
    'rgb_only': (False, False, True),
    'gaze_and_imu': (True, True, False),
    'gaze_and_rgb': (True, False, True),
    'imu_and_rgb': (False, True, True),
    'all': (True, True, True),
}