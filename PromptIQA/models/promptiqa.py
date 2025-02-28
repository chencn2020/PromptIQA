"""
    The completion for Mean-opinion Network(MoNet)
"""
import torch
import torch.nn as nn
import timm

from timm.models.vision_transformer import Block
from functools import partial
from einops import rearrange

class Attention_Block(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class Self_Attention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()

        self.qConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.kConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.vConv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inFeature):
        bs, C, w, h = inFeature.size()

        proj_query = self.qConv(inFeature).view(bs, -1, w * h).permute(0, 2, 1).contiguous()
        proj_key = self.kConv(inFeature).view(bs, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.vConv(inFeature).view(bs, -1, w * h)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1).contiguous())
        out = out.view(bs, C, w, h)

        out = self.gamma * out + inFeature

        return out


class MAL(nn.Module):
    """
        Multi-view Attention Learning (MAL) module
    """

    def __init__(self, in_dim=768, feature_num=4, feature_size=28):
        super().__init__()

        self.channel_attention = Attention_Block(in_dim * feature_num)  # Channel-wise self attention
        self.feature_attention = Attention_Block(feature_size ** 2 * feature_num)  # Pixel-wise self attention

        # Self attention module for each input feature
        self.attention_module = nn.ModuleList()
        for _ in range(feature_num):
            self.attention_module.append(Self_Attention(in_dim))

        self.feature_num = feature_num
        self.in_dim = in_dim

    def forward(self, features):
        feature = torch.tensor([]).cuda()
        for index, _ in enumerate(features):
            feature = torch.cat((feature, self.attention_module[index](features[index]).unsqueeze(0)), dim=0)
        features = feature

        input_tensor = rearrange(features, 'n b c w h -> b (n c) (w h)')  # bs, 768 * feature_num, 28 * 28
        bs, _, _ = input_tensor.shape  # [2, 3072, 784]

        in_feature = rearrange(input_tensor, 'b (w c) h -> b w (c h)', w=self.in_dim,
                               c=self.feature_num)  # bs, 768, 28 * 28 * feature_num
        feature_weight_sum = self.feature_attention(in_feature)  # bs, 768, 768

        in_channel = input_tensor.permute(0, 2, 1).contiguous()  # bs, 28 * 28, 768 * feature_num
        channel_weight_sum = self.channel_attention(in_channel)  # bs, 28 * 28, 28 * 28

        weight_sum_res = (rearrange(feature_weight_sum, 'b w (c h) -> b (w c) h', w=self.in_dim,
                                    c=self.feature_num) + channel_weight_sum.permute(0, 2, 1).contiguous()) / 2  # [2, 3072, 784]

        weight_sum_res = torch.mean(weight_sum_res.view(bs, self.feature_num, self.in_dim, -1), dim=1)

        return weight_sum_res  # bs, 768, 28 * 28


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class PromptIQA(nn.Module):
    def __init__(self, patch_size=8, drop=0.1, dim_mlp=768, img_size=224):
        super().__init__()
        self.img_size = img_size
        self.input_size = img_size // patch_size
        self.dim_mlp = dim_mlp

        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.vit.norm = nn.Identity()
        self.vit.head = nn.Identity()

        self.save_output = SaveOutput()

        # Register Hooks
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        self.MALs = nn.ModuleList()
        for _ in range(3):
            self.MALs.append(MAL())

        # Image Quality Score Regression
        self.fusion_mal = MAL(feature_num=3)
        self.block = Block(dim_mlp, 12)
        self.cnn = nn.Sequential(
            nn.Conv2d(dim_mlp, 256, 5),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2, 2)),
            nn.Conv2d(256, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((2, 2)),
            nn.Conv2d(128, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((3, 3)),
        )

        self.i_p_fusion = nn.Sequential(
            Block(128, 4),
            Block(128, 4),
            Block(128, 4),
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 128),
        )
        
        self.prompt_fusion = nn.Sequential(
            Block(128, 4),
            Block(128, 4),
            Block(128, 4),
        )
        
        dpr = [x.item() for x in torch.linspace(0, 0, 8)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=128, num_heads=4, mlp_ratio=4, qkv_bias=True, drop=0,
                attn_drop=0, drop_path=dpr[i], norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU)
            for i in range(8)])
        self.norm = nn.LayerNorm(128)
        
        self.score_block = nn.Sequential(
            nn.Linear(128, 128 // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(128 // 2, 1),
            nn.Sigmoid()
        )
        
        self.prompt_feature = {}

    @torch.no_grad()
    def clear(self):
        self.prompt_feature = {}
        
    @torch.no_grad()
    def inference(self, x, data_type):
        prompt_feature = self.prompt_feature[data_type] # 1, n, 128

        _x = self.vit(x)
        x = self.extract_feature(self.save_output)  # bs, 28 * 28, 768 * 4
        self.save_output.outputs.clear()

        x = x.permute(0, 2, 1).contiguous()  # bs, 768 * 4, 28 * 28
        x = rearrange(x, 'b (d n) (w h) -> b d n w h', d=4, n=self.dim_mlp, w=self.input_size, h=self.input_size)  # bs, 4, 768, 28, 28
        x = x.permute(1, 0, 2, 3, 4).contiguous()  # bs, 4, 768, 28 * 28

        # Different Opinion Features (DOF)
        DOF = torch.tensor([]).cuda()
        for index, _ in enumerate(self.MALs):
            DOF = torch.cat((DOF, self.MALs[index](x).unsqueeze(0)), dim=0)
        DOF = rearrange(DOF, 'n c d (w h) -> n c d w h', w=self.input_size, h=self.input_size)  # M, bs, 768, 28, 28

        # Image Quality Score Regression
        fusion_mal = self.fusion_mal(DOF).permute(0, 2, 1).contiguous()  # bs, 28 * 28 768
        IQ_feature = self.block(fusion_mal).permute(0, 2, 1).contiguous() # bs, 768, 28 * 28
        IQ_feature = rearrange(IQ_feature, 'c d (w h) -> c d w h', w=self.input_size, h=self.input_size) # bs, 768, 28, 28
        img_feature = self.cnn(IQ_feature).squeeze(-1).squeeze(-1).unsqueeze(1) # bs, 1, 128

        prompt_feature = prompt_feature.repeat(img_feature.shape[0], 1, 1) # bs, n, 128
        prompt_feature = self.prompt_fusion(prompt_feature) # bs, n, 128

        fusion = self.blocks(torch.cat((img_feature, prompt_feature), dim=1)) # bs, 2, 1
        fusion = self.norm(fusion)
        fusion = self.score_block(fusion)

        iq_res = fusion[:, 0].view(-1)
        
        return iq_res

    @torch.no_grad()
    def forward_prompt(self, x, score, data_type):
        _x = self.vit(x)
        x = self.extract_feature(self.save_output)  # bs, 28 * 28, 768 * 4
        self.save_output.outputs.clear()

        x = x.permute(0, 2, 1).contiguous()  # bs, 768 * 4, 28 * 28
        x = rearrange(x, 'b (d n) (w h) -> b d n w h', d=4, n=self.dim_mlp, w=self.input_size, h=self.input_size)  # bs, 4, 768, 28, 28
        x = x.permute(1, 0, 2, 3, 4).contiguous()  # 4, bs, 768, 28, 28

        # Different Opinion Features (DOF)
        DOF = torch.tensor([]).cuda()
        for index, _ in enumerate(self.MALs):
            DOF = torch.cat((DOF, self.MALs[index](x).unsqueeze(0)), dim=0)
        DOF = rearrange(DOF, 'n c d (w h) -> n c d w h', w=self.input_size, h=self.input_size)  # M, bs, 768, 28, 28

        # Image Quality Score Regression
        fusion_mal = self.fusion_mal(DOF).permute(0, 2, 1).contiguous()  # bs, 28 * 28 768
        IQ_feature = self.block(fusion_mal).permute(0, 2, 1).contiguous() # bs, 768, 28 * 28
        IQ_feature = rearrange(IQ_feature, 'c d (w h) -> c d w h', w=self.input_size, h=self.input_size) # bs, 768, 28, 28
        img_feature = self.cnn(IQ_feature).squeeze(-1).squeeze(-1).unsqueeze(1) # bs, 1, 128

        score_feature = score.expand(-1, 128)

        funsion_feature = self.i_p_fusion(torch.cat((img_feature, score_feature.unsqueeze(1)), dim=1)) # bs, 2, 128
        funsion_feature = self.mlp(torch.mean(funsion_feature, dim=1)).unsqueeze(0) # 1, n, 128

        self.prompt_feature[data_type] = funsion_feature.clone()
    
    def forward(self, x, score):
        _x = self.vit(x)
        x = self.extract_feature(self.save_output)  # bs, 28 * 28, 768 * 4
        self.save_output.outputs.clear()

        x = x.permute(0, 2, 1).contiguous()  # bs, 768 * 4, 28 * 28
        x = rearrange(x, 'b (d n) (w h) -> b d n w h', d=4, n=self.dim_mlp, w=self.input_size, h=self.input_size)  # bs, 4, 768, 28, 28
        x = x.permute(1, 0, 2, 3, 4).contiguous()  # 4, bs, 768, 28, 28

        # Different Opinion Features (DOF)
        DOF = torch.tensor([]).cuda()
        for index, _ in enumerate(self.MALs):
            DOF = torch.cat((DOF, self.MALs[index](x).unsqueeze(0)), dim=0)
        DOF = rearrange(DOF, 'n c d (w h) -> n c d w h', w=self.input_size, h=self.input_size)  # M, bs, 768, 28, 28

        # Image Quality Score Regression
        fusion_mal = self.fusion_mal(DOF).permute(0, 2, 1).contiguous()  # bs, 28 * 28 768
        IQ_feature = self.block(fusion_mal).permute(0, 2, 1).contiguous() # bs, 768, 28 * 28
        IQ_feature = rearrange(IQ_feature, 'c d (w h) -> c d w h', w=self.input_size, h=self.input_size) # bs, 768, 28, 28
        img_feature = self.cnn(IQ_feature).squeeze(-1).squeeze(-1).unsqueeze(1) # bs, 1, 128

        score_feature = score.expand(-1, 128) # bs, 128

        funsion_feature = self.i_p_fusion(torch.cat((img_feature, score_feature.unsqueeze(1)), dim=1)) # bs, 2, 128
        funsion_feature = self.mlp(torch.mean(funsion_feature, dim=1)) #bs, 128
        funsion_feature = self.expand(funsion_feature) # bs, bs - 1, 128
        funsion_feature = self.prompt_fusion(funsion_feature) # bs, bs - 1, 128

        fusion = self.blocks(torch.cat((img_feature, funsion_feature), dim=1)) # bs, 2, 1
        fusion = self.norm(fusion)
        fusion = self.score_block(fusion)
        iq_res = fusion[:, 0].view(-1)
        gt_res = score.view(-1)
        
        return iq_res, gt_res
    
    def extract_feature(self, save_output, block_index=[2, 5, 8, 11]):
        x1 = save_output.outputs[block_index[0]][:, 1:]
        x2 = save_output.outputs[block_index[1]][:, 1:]
        x3 = save_output.outputs[block_index[2]][:, 1:]
        x4 = save_output.outputs[block_index[3]][:, 1:]
        x = torch.cat((x1, x2, x3, x4), dim=2)
        return x

    def expand(self, A):
        A_expanded = A.unsqueeze(0).expand(A.size(0), -1, -1)

        B = None
        for index, i in enumerate(A_expanded):
            rmv = torch.cat((i[:index], i[index + 1:])).unsqueeze(0)
            if B is None:
                B = rmv
            else:
                B = torch.cat((B, rmv), dim=0)

        return B
