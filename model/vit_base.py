import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, DropPath
from collections import OrderedDict
from functools import partial


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_num = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.patch_projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # [B, C, W, H] -> [B, D, Pn_W, Pn_H] -> [B, D, Pn] -> [B, Pn, D]
        x = self.patch_projection(x).flatten(2).transpose(1, 2)
        return x
    

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features  # if out_features=None
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        batch, patch_num, patch_dim = x.shape
        # [B, Pn, D] -> [B, Pn, 3D] -> [B, Pn, 3, H, D/H] -> [3, B, H, Pn, D/H]
        qkv = self.qkv(x).reshape(batch, patch_num, 3, self.num_heads, patch_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        # [3, B, H, Pn, D/H] -> [B, H, Pn, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]
        # [B, H, Pn, D/H] @ [B, H, D/H, Pn] -> [B, H, Pn, Pn]
        attn_score = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn_score.softmax(dim=-1)
        attn_score = self.attn_drop(attn_score)
        # [B, H, Pn, Pn] @ [B, H, Pn, D/H] -> [B, H, Pn, D/H]
        attn = attn_score @ v
        # multi-head attention fusion: [B, Pn, H, D/H] -> [B, Pn, D]
        x = attn.transpose(1, 2).reshape(batch, patch_num, patch_dim)
        # mlp module
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # pre-normalization
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # add + norm
        self.norm2 = norm_layer(dim)
        # FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        # MHSA + Res_connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # FFN + Res_connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        # patch embedding
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_channels=in_chans, embed_dim=embed_dim)
        patch_num = self.patch_embed.patch_num
        # embedding tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_num + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        # Stochastic Depth Decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # ViT encoders
        self.blocks = nn.Sequential(*[
            ViTBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, 
                     attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        # last norm
        self.norm = norm_layer(embed_dim)
        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        # classifier head
        self.head = nn.Linear(self.num_features, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.head_dist = None
    
    def forward_feature(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # [B, 1, dim]
        x = torch.cat((cls_token, x), dim=1) # [B, Pn+1, dim]
        x = self.pos_drop(x + self.pos_embed) # [B, Pn+1, dim]
        # ViT encoders
        x = self.blocks(x)
        x = self.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_feature(x)
        x = self.pre_logits(x[:, 0])  # use cls token for cls head [B,1,dim]
        x = self.head(x)
        return x


if __name__=='__main__':
    '''my vit model'''
    x = torch.randn(8, 3, 224, 224)
    model = ViT()
    y = model(x)
    print(y.shape)
    '''pre train model'''
    from transformers import ViTModel, ViTConfig
    model_config = 'google/vit-base-patch16-224-in21k'
    model = ViTModel.from_pretrained(model_config, add_pooling_layer=False)
    y = model(x)
    last_embed = y['last_hidden_state']
    print(last_embed.size()) # [8, 197, 768]