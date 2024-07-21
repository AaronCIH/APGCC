from .APGCC import Model_builder, SetCriterion_Crowd
from .matcher import build_matcher_crowd

# create the main model
def build_model(cfg, training):
   model = Model_builder(cfg)
   if not training: 
      return model

   weight_dict = cfg.MODEL.WEIGHT_DICT
   matcher = build_matcher_crowd(cfg)
   if not cfg.MODEL.AUX_EN:
      del weight_dict['loss_aux']
   criterion = SetCriterion_Crowd(num_classes=1, \
                                  matcher=matcher, weight_dict=weight_dict, \
                                  eos_coef=cfg.MODEL.EOS_COEF, \
                                  aux_kwargs = {'AUX_NUMBER': cfg.MODEL.AUX_NUMBER,
                                                'AUX_RANGE': cfg.MODEL.AUX_RANGE, 
                                                'AUX_kwargs': cfg.MODEL.AUX_kwargs})
   return model, criterion