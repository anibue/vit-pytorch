from random import randrange
import torch
from torch import nn, einsum
import torch.nn.functional as F

"""
引入einops库用于张量重排和重复操作
"""
from einops import rearrange, repeat    #  rearrange用于重排张量，repeat用于重复张量
from einops.layers.torch import Rearrange   #  Rearrange用于重排张量,区别于rearrange函数,前者都是张量操作,后者是函数调用

# helpers

def exists(val):
    return val is not None

def dropout_layers(layers: nn.ModuleList, dropout: float) -> list:
    """返回经过dropout处理后的层列表，确保总是返回可迭代的列表"""
    if dropout == 0:
        return list(layers)  # 转换为列表确保可迭代

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    # 确保返回标准的Python列表
    return list(layer for (layer, drop) in zip(layers, to_drop) if not drop)

# classes


"""
1. 深度自适应初始化
2. 逐通道宽度缩放
"""
class LayerScale(nn.Module):
    """
    LayerScale module applies a learnable scaling parameter to the output of a given function (fn).
    The initial value of the scaling parameter depends on the depth of the layer, as described in the paper.
    Args:
        dim (int): The dimension of the scaling parameter.
        fn (nn.Module or callable): The function or module whose output will be scaled.
        depth (int): The depth of the layer, used to determine the initial epsilon value for scaling.
    Attributes:
        scale (nn.Parameter): Learnable scaling parameter initialized based on depth.
        fn (nn.Module or callable): The function to apply to the input.
    Forward Args:
        x (Tensor): Input tensor.
        **kwargs: Additional arguments to pass to fn.
    Returns:
        Tensor: The output of fn(x, **kwargs) scaled by the learnable parameter.
    """
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1  # 误差
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5  # 误差
        else:
            init_eps = 1e-6  # 误差

        scale = torch.zeros(1, 1, dim).fill_(init_eps)  # 创建一个维度为dim的向量，并初始化为0;.fill_(init_eps)将其填充为init_eps的值
        self.scale = nn.Parameter(scale)    # 创建一个可学习参数
        self.fn = fn    
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



"""
1. talking heads
2. 层归一化
3. dropout
4. 线性变换
交叉注意力头交互(Cross-head interaction)：通过可学习矩阵让注意力头之间可以交换信息
两阶段设计：在注意力计算的不同阶段(预softmax和后softmax)应用不同的混合策略
参数化注意力增强：相比标准Transformer固定模式的注意力头交互，这种设计更灵活
"""

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5   # 缩放因子，防止点积过大导致梯度消失

        self.norm = nn.LayerNorm(dim)   # 层归一化
        self.to_q = nn.Linear(dim, inner_dim, bias = False) # 查询向量
        # 键值向量的线性变换，输出维度为inner_dim，维数与输入相同
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)    # 键值向量

        self.attend = nn.Softmax(dim = -1)  # 注意力机制，softmax归一化
        self.dropout = nn.Dropout(dropout)  # dropout层，防止过拟合

        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))   # 可学习的头部混合矩阵，预-softmax
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))  # 可学习的头部混合矩阵，后-softmax

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None):
        b, n, _, h = *x.shape, self.heads

        x = self.norm(x)
        context = x if context is None else torch.cat((x, context), dim = 1)    # 获取输入和上下文

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))   # 获取查询、键、值向量,chunks函数将一个张量切分为多个子张量
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)


        # 预softmax混合作用:
        # 1. 信息聚合:在归一化前允许注意力头之间交换信息
        # 2. 特征融合:允许不同特征之间的融合,共享特征模式
        # 3. 增强表达:扩大注意力模式的搜索空间
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale     # 点积注意力计算，使用爱因斯坦求和约定,标准的注意力计算方式是将查询向量和键向量进行点积，然后除以缩放因子。这里使用了einsum函数来实现这一点。
        """
        einsum函数用于执行爱因斯坦求和约定，'b h i j'表示输入张量的形状，'h g'表示输出张量的形状，'i'和'j'表示输入张量的两个维度，'d'表示输入张量的第三个维度，'g'表示输出张量的第二个维度。
        这里d不是公共维度，即每一个注意力头的维度是独立的，而不是共享的。
        通过这种方式，可以实现对每个注意力头的独立处理，从而实现更灵活的注意力机制。
        """
        # 后softmax混合作用:
        # 1. 权重重分配:归一化前允许注意力头之间交换信息
        # 2. 动态调整:允许注意力头之间的动态调整
        
        dots = einsum('b h i j, h g -> b g i j', dots, self.mix_heads_pre_attn)    # 注意力头之间的混合，预-softmax
        # 这里的h表示注意力头数量，g表示talking heads数量;
        attn = self.attend(dots)
        attn = self.dropout(attn)

        attn = einsum('b h i j, h g -> b g i j', attn, self.mix_heads_post_attn)   # talking heads, post-softmax

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., layer_dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout

        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                LayerScale(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout), depth = ind + 1),
                LayerScale(dim, FeedForward(dim, mlp_dim, dropout = dropout), depth = ind + 1)
            ]))
    def forward(self, x, context = None):
        layers = dropout_layers(self.layers, dropout = self.layer_dropout)  # 输出的layers是一个列表，包含了经过dropout处理的层
        """
        nn.ModuleList是一个特殊的容器，用于存储nn.Module的子模块。它可以像列表一样迭代，但在处理时会自动注册子模块。
        但,nn.Module本身不是可迭代的对象，因此不能直接迭代nn.ModuleList。
        于是设置dropout_layers函数返回一个列表，确保可以直接迭代。
        """
        for layer in layers:  # 直接迭代列表
            attn, ff = layer  # 使用解包方式访问子模块
            x = attn(x, context = context) + x
            x = ff(x) + x
        return x


"""
CaiT: Class-Attention-in-Transformers
@article{chen2021cait,
  title={CaiT: Class-Attention in Transformers},
  journal={arXiv preprint arXiv:2103.17239},
  year={2021}
}
# CaiT（Class-Attention in Transformers）是一种改进的视觉Transformer模型，专为提升深层Transformer在图像分类任务中的表现而设计。其主要创新点包括：
# 1. LayerScale：为每一层引入可学习的缩放参数，提升深层网络的稳定性。
# 2. Talking-head Attention：在注意力计算前后引入可学习的头部混合矩阵，增强不同注意力头之间的信息交互。
# 3. Class-Attention层（cls_transformer）：在Transformer后端引入专门的Class Attention模块，仅对类别token与patch token进行交互，提升分类性能。
# 4. 支持Layer Dropout，进一步提升模型泛化能力。
# CaiT在ImageNet等数据集上取得了优异的性能，尤其在模型加深时表现突出。
"""
class CaiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.,
        layer_dropout = 0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2   # 获取图像的像素数-图像可以分割为的图像块数量
        patch_dim = 3 * patch_size ** 2 # 获取图像块的维度,即每个图像块的像素数 3:三通道;** 2:每个图像块的宽度和高度之积

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # 初始化位置嵌入，用于捕捉序列中每个补丁的位置信息
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # 初始化分类令牌嵌入，作为序列的特殊标记，用于后续的分类任务
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 初始化嵌入层的dropout，用于防止过拟合
        self.dropout = nn.Dropout(emb_dropout)

        # 初始化补丁变换器，用于对补丁序列进行自注意力和前馈操作
        self.patch_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout)
        # 初始化分类变换器，专门用于处理添加了分类令牌的序列，以进行分类任务
        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        # 初始化多层感知器头部，用于将变换器的输出映射到分类的输出维度
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 只取前n个位置嵌入(n是当前batch的实际patch数量)
        # 因为pos_embedding预先计算了最大可能patch数(num_patches)
        # 但实际输入图像的patch数可能小于num_patches(如图像被裁剪时)
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.patch_transformer(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)   # 1, 1, dim -> b, 1, dim
        x = self.cls_transformer(cls_tokens, context = x)

        return self.mlp_head(x[:, 0])
