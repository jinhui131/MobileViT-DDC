# from models.mobilevit import mobile_vit_x_small
# from models.mobilevit import mobile_vit_xx_small
# from models.mobilevit_dilate import  mobile_vit_dilate_xx_small
# from models.mobilevit import mobile_vit_small
# from models.mobilevit_dilate import mobile_vit_dilate_x_small
# from models.mobilevit_adda import mobile_vit_adda_xx_small
# from models.mobilenet_v3 import mobilenet_v3_large
# from models.shufflenet_v2 import shufflenet_v2_x0_5
# from models.xception import xception
# from models.edgevit import edgevit_s
# from models.ghostnetv2_torch import ghostnetv2
# from models.mobilenet_v3 import mobilenet_v3_small
# from models.edgevit import edgevit_xs
# from models.shufflenet_v2 import shufflenet_v2_x1_0
# from models.mobilenet_v3 import mobilenet_v3_small
# from models.mobilevit import mobile_vit_x_small
# from models.edgevit import edgevit_xs
# from models.shufflenet_v2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0
# from models.edgevit import edgevit_s
# from models.ghostnetv2_torch import ghostnetv2
# from models.ghostnetv2_torch import ghostnetv2
# from models.mobilevit_adda import mobile_vit_adda_x_small
# from models.resnet import resnet50
# from models.coatnet import coatnet_0

from models.mobilevit_adda import mobile_vit_adda_x_small
# from models.mobilevit import mobile_vit_small
# # from models.mobilevit_dilate import mobile_vit_dilate_small
# from models.mobilevit import mobile_vit_small
# from models.mobilevit_dilate import mobile_vit_dilate_small
# from models.efficientnet_v2 import efficientnetv2_m
cfgs = {
    # 'shufflenet_v2_x1_0':shufflenet_v2_x1_0
    #   'resnet50':resnet50
   # 'coatnet_0':coatnet_0
    # 'xception':xception
    # 'edgevit_s':edgevit_s
    # 'ghostnetv2':ghostnetv2
    #    'mobilenet_v3_large':mobilenet_v3_large
    # 'mobilenet_v3_small':mobilenet_v3_small
    # 'shufflenet_v2_x1_0':shufflenet_v2_x1_0
    # 'edgevit_s':edgevit_s
    # 'mobile_vit_x_small':mobile_vit_x_small
    'mobile_vit_adda_x_small':mobile_vit_adda_x_small
    # 'efficientnetv2_m':efficientnetv2_m
    # 'mobile_vit_dilate_xx_small': mobile_vit_dilate_xx_small
    #   'mobile_vit_xx_small': mobile_vit_xx_small
    #   'mobile_vit_small':mobile_vit_small
    #   'mobile_vit_dilate_small': mobile_vit_dilate_small
    #   'mobile_vit_dilate_x_small': mobile_vit_dilate_x_small
}

def find_model_using_name(model_name, num_classes):
    return cfgs[model_name](num_classes)
# def find_model_using_name(model_name, num_classes=1000, width=1.0, dropout=0.2, args=None):
#     if model_name == 'ghostnetv2':
#         return ghostnetv2(num_classes=num_classes, width=width, dropout=dropout, args=args)

 
