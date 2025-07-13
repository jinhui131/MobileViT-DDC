import torch
import torch.nn as nn
from einops import einops
from thop import profile
from timm.models.layers import DropPath
import torch.nn.functional as F
from torchvision.ops import DeformConv2d, deform_conv2d  # 引入可变形卷积



class CBSLayer(nn.Module):
    def __init__(self, num_features=None, eps=1e-6):
        super(CBSLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        # 初始化为 None，稍后会根据输入的特征动态调整
        self.scale = None

    def forward(self, x):
        # x 的形状假设为 [batch_size, height, width, num_features]
        if self.scale is None or self.scale.shape[-1] != x.shape[-1]:
            # 如果未初始化或特征数变化，动态初始化 scale
            self.num_features = x.shape[-1]
            self.scale = nn.Parameter(torch.ones(1, 1, 1, self.num_features).to(x.device))

        global_context, _ = torch.median(x, dim=1, keepdim=True)  # [batch_size, 1, width, num_features]

        # 确保 scale 的形状与 global_context 匹配
        scaled_context = global_context * self.scale.expand_as(global_context)

        # 广播加到每个 token 上
        x = x + scaled_context
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,cb_layer=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.cb_layer = cb_layer
    def forward(self, x):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if self.cb_layer:
            x = self.cb_layer(x)
        x = self.drop(x)

        return x


class ADDAAttention(nn.Module):
    "Implementation of Dilate-attention with dynamic convolution"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.dilation = dilation

        # 只有当膨胀率不为 1 时，才使用动态卷积
        if dilation != 1:
            self.offset_conv = nn.Conv2d(head_dim, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                         padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
            self.deform_conv = DeformConv2d(head_dim, head_dim, kernel_size=kernel_size,
                                            padding=dilation * (kernel_size - 1) // 2, dilation=dilation)
        else:
            self.offset_conv = None
            self.deform_conv = None

        # 滑动窗口操作（Unfold），用于生成 k 和 v 的展开
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation,
                                padding=dilation * (kernel_size - 1) // 2, stride=1)

        # Dropout for attention
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        B, d, H, W = q.shape  # 获取输入形状，B是batch size，d是维度，H和W是图像的高和宽

        # Detach the tensors to avoid tracking gradients during attention calculation
        q, k, v = q.detach(), k.detach(), v.detach()

        # Reshape q, k, v into the form [B, h, N, 1, d]
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B, h, N, 1, d

        # 如果膨胀率不为 1，使用动态卷积来生成偏移量
        if self.offset_conv is not None:
            offsets = self.offset_conv(k)  # 通过卷积学习偏移量

            # 使用可变形卷积对 k 和 v 进行处理，保持通道数一致
            k_deformed = self.deform_conv(k, offsets)  # 对 k 进行可变形卷积
            v_deformed = self.deform_conv(v, offsets)  # 对 v 进行可变形卷积
        else:
            k_deformed = k
            v_deformed = v

        # Unfold k and v (sliding window operation)
        k_unfold = self.unfold(k_deformed)  # 生成滑动窗口
        v_unfold = self.unfold(v_deformed)  # 生成滑动窗口

        # 对k和v的窗口进行偏移
        B, d, H, W = k.shape
        k_unfold_reshaped = k_unfold.view(B, d // self.head_dim, self.head_dim,
                                          self.kernel_size * self.kernel_size, H * W).permute(0, 1, 4, 2,
                                                                                              3)  # B, h, N, k*k, d
        v_unfold_reshaped = v_unfold.view(B, d // self.head_dim, self.head_dim,
                                          self.kernel_size * self.kernel_size, H * W).permute(0, 1, 4, 3,
                                                                                              2)  # B, h, N, k*k, d

        # 计算 q 和 k 的点积并进行 softmax 归一化得到注意力矩阵
        attn = (q @ k_unfold_reshaped) * self.scale  # B, h, N, 1, k*k
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 使用注意力矩阵加权 value (v)
        x = (attn @ v_unfold_reshaped).transpose(1, 2).reshape(B, H, W, d)  # B, H, W, d

        return x



class MultiADDALocalAttention(nn.Module):
    "膨胀注意力的实现，结合可变形卷积"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 2, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0, f"num_heads {num_heads} 必须是 num_dilation {self.num_dilation} 的整数倍!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)

        # 使用 ADDAAttention 代替原始的 ADDAAttention
        self.ADDA_attention = nn.ModuleList(
            [ADDAAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)]
        )

        # 每层注意力后加 CBS 层
        self.cbs_layers = nn.ModuleList([CBSLayer(head_dim) for _ in range(self.num_dilation - 1)])

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # B, C, H, W

        # 生成 qkv
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)
        # qkv 形状为 [num_dilation, 3, B, C//num_dilation, H, W]

        # 对输入 x 进行重构，适应多膨胀率注意力的输入
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)
        # x 形状为 [num_dilation, B, H, W, C//num_dilation]

        # 遍历不同的膨胀率，分别处理特征
        for i in range(self.num_dilation):
            # 使用可变形卷积进行膨胀注意力处理
            x[i] = self.ADDA_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])  # 使用  attention

            # if i < self.num_dilation - 1:
            #     # 通过 CBS 层进一步处理膨胀注意力输出
            #     x[i + 1] = self.cbs_layers[i](x[i + 1])

        # 恢复到 [B, H, W, C] 形状
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)  # 投影层
        x = self.proj_drop(x)  # Dropout
        return x



# 修改后的 ADDABlock，支持不同层使用不同的卷积
class ADDABlock(nn.Module):
    "实现带有膨胀注意力的 Transformer Block"

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 kernel_size=3, dilation=[1, 2, 3], cpe_per_block=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block

        # 如果启用了 CPE（位置编码），我们为每个 block 添加位置编码层
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        # 归一化层，处理输入特征
        self.norm1 = norm_layer(dim)

        # 调用修改后的 MultiADDALocalAttention，内部已经使用了 ADDAAttention
        self.attn = MultiADDALocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              attn_drop=attn_drop, kernel_size=kernel_size, dilation=dilation)

        # 路径 dropout 机制（默认不开启）
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 第二个归一化层
        self.norm2 = norm_layer(dim)

        #MLP 层，用于特征的进一步处理，维度提升到 dim * mlp_ratio
        mlp_hidden_dim = int(dim * mlp_ratio)
        cb_layer = CBSLayer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, cb_layer=cb_layer)


    def forward(self, x):
        # 如果启用了 CPE，在每个 block 中进行位置编码的叠加
        if self.cpe_per_block:
            x = x + self.pos_embed(x)

        # 形状变换：B, C, H, W -> B, H, W, C，以便输入注意力模块
        x = x.permute(0, 2, 3, 1)

        # 执行多膨胀率的局部注意力机制
        x = x + self.drop_path(self.attn(self.norm1(x)))

       # 执行 MLP 层
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 3, 1, 2)

        return x




