import torch
def talking_heads_attention_demo():
    # 1. 标准注意力计算
    q = torch.randn(2, 4, 6, 8)  # (batch, heads, seq_len, dim_head)
    k = torch.randn(2, 4, 6, 8)
    v = torch.randn(2, 4, 6, 8)
    
    # 计算注意力分数
    dots = torch.einsum('b h i d, b h j d -> b h i j', q, k)
    print(f"1. 原始注意力分数: {dots.shape}")
    
    # 2. 预softmax混合 (Talking Heads第一阶段)
    mix_pre = torch.randn(4, 4)
    dots_mixed_pre = torch.einsum('b h i j, h g -> b g i j', dots, mix_pre)
    print(f"2. 预softmax混合后: {dots_mixed_pre.shape}")
    import matplotlib.pyplot as plt

    # 可视化预softmax混合前后的注意力分数（以第一个batch和第一个token为例）
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(dots[0, :, 0, :].detach().numpy(), aspect='auto', cmap='viridis')
    axs[0].set_title('原始注意力分数\n[batch=0, :, token=0, :]')
    plt.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(dots_mixed_pre[0, :, 0, :].detach().numpy(), aspect='auto', cmap='viridis')
    axs[1].set_title('预softmax混合后\n[batch=0, :, token=0, :]')
    plt.colorbar(im1, ax=axs[1])
    plt.tight_layout()
    plt.show()
    
    # 3. Softmax归一化
    attn = torch.softmax(dots_mixed_pre, dim=-1)
    print(f"3. Softmax后: {attn.shape}")
    
    # 4. 后softmax混合 (Talking Heads第二阶段)
    mix_post = torch.randn(4, 4)
    attn_mixed_post = torch.einsum('b h i j, h g -> b g i j', attn, mix_post)
    print(f"4. 后softmax混合后: {attn_mixed_post.shape}")
    
    # 5. 应用到值向量
    out = torch.einsum('b h i j, b h j d -> b h i d', attn_mixed_post, v)
    print(f"5. 最终输出: {out.shape}")

talking_heads_attention_demo()