import torch.nn as nn

from models import pre_processor, backbone_3d, head, roi_head

class RL3DF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        load_teacher = cfg.OPTIMIZER.get('LOAD_TEACHER', None)
        if load_teacher is not None:
            self.cfg_model = cfg.MODEL_T
        else:
            self.cfg_model = cfg.MODEL
        
        self.list_module_names = [
            'pre_processor', 'pre_processor2', 'backbone', 'head'
        ]
        self.list_modules = []
        self.build_rl_detector()

    def build_rl_detector(self):
        for name_module in self.list_module_names:
            module = getattr(self, f'build_{name_module}')()
            if module is not None:
                self.add_module(name_module, module) # override nn.Module
                self.list_modules.append(module)

    def build_pre_processor(self):
        return pre_processor.__all__[self.cfg_model.PRE_PROCESSOR.NAME](self.cfg)
    
    def build_pre_processor2(self):
        return pre_processor.__all__[self.cfg_model.PRE_PROCESSOR2.NAME](self.cfg)

    def build_backbone(self):
        return backbone_3d.__all__[self.cfg_model.BACKBONE.NAME](self.cfg)

    def build_head(self):
        return head.__all__[self.cfg_model.HEAD.NAME](self.cfg)

    def forward(self, x):
        for module in self.list_modules:
            x = module(x)
        return x