'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''

from .rl_3df import RL3DF
from .rl_3df_student import RL3DF_student

def build_skeleton(cfg):
    return __all__[cfg.MODEL.SKELETON](cfg)

def build_skeleton_teacher(cfg):
    return __all__[cfg.MODEL_T.SKELETON](cfg)

__all__ = {
    'RL3DF': RL3DF,
    'RL3DF_student': RL3DF_student,
}
