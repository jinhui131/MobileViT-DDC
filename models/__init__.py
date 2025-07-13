

from models.mobilevit_adda import mobile_vit_adda_x_small

cfgs = {

    'mobile_vit_adda_x_small':mobile_vit_adda_x_small

}

def find_model_using_name(model_name, num_classes):
    return cfgs[model_name](num_classes)

 
