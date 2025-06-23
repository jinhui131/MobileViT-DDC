import torch
from thop import profile, clever_format
from models import find_model_using_name  # 导入你的模型定义

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = find_model_using_name("mobile_vit_dilate_xx_small", num_classes=5).to(device)

# 加载已保存的权重
weights_path = "/home/home/jh/MobileVit/GuiZhou/weights/mobile_vit_dilate_xx_small/mobile_vit_dilate_xx_small/mobile_vit_dilate_xx_small.pth"
model.load_state_dict(torch.load(weights_path))

# 创建一个假输入张量用于计算
input_tensor = torch.randn(1, 3, 384, 384).to(device)  # 假设模型输入为224x224的RGB图像

# 计算 FLOPs 和 Params
flops, params = profile(model, inputs=(input_tensor,))
flops, params = clever_format([flops, params], "%.8f")  # 格式化输出
print(f"FLOPs: {flops}, Params: {params}")
