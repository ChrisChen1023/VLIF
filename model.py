from spikingjelly.activation_based.neuron import LIFNode
from spikingjelly.activation_based import functional, layer
import torch
import torch.nn as nn
import torch.nn.functional as F

v_th = 0.15

alpha = 1 / (2 ** 0.5)

# Constants and classes from multispike4.py
decay = 0.25  # decay constants

class MultiSpike4(nn.Module):
    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            # First quantize in the range 0-4, then normalize by dividing by 4 to make the maximum value 1
            quantized = torch.round(torch.clamp(input, min=0, max=4))
            return quantized / 4.0

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            # Consider gradient scaling for normalization
            return grad_input / 4.0

    def forward(self, x):
        return self.quant4.apply(x)

class mem_update(nn.Module):
    def __init__(self):
        super(mem_update, self).__init__()
        self.qtrick = MultiSpike4()  # change the max value

    def forward(self, x):
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]
            else:
                mem = x[i]
            spike = self.qtrick(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output

##### PixelShuffleLIFBlock #####
class PixelShuffleLIFBlock(nn.Module):
    def __init__(self, in_channels, downsample_factor=2):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.r = downsample_factor
        
        # Pixel unshuffle for downsampling
        self.pixel_unshuffle = nn.PixelUnshuffle(downsample_factor)
        
        # Use mem_update instead of LIFNode to process each patch position
        self.lif_node = mem_update()
        
        # Optional: Channel adjustment layer to ensure output channels match input
        self.channel_adjust = layer.Conv2d(
            in_channels * (downsample_factor ** 2), 
            in_channels, 
            kernel_size=1, 
            bias=False,
            step_mode='m'
        )
        functional.set_step_mode(self.channel_adjust, step_mode='m')
        
    def forward(self, x):
        # Input: [T, B, C, H, W]
        T, B, C, H, W = x.shape
        
        # 1. Flatten time and batch dimensions for 2D operations
        # [T, B, C, H, W] -> [T*B, C, H, W]
        x_flat = x.reshape(T * B, C, H, W)
        
        # 2. Pixel unshuffle downsampling
        # [T*B, C, H, W] -> [T*B, C*r^2, H//r, W//r]
        x_downsampled = self.pixel_unshuffle(x_flat)
        
        # 3. Restore time and batch dimensions
        # [T*B, C*r^2, H//r, W//r] -> [T, B, C*r^2, H//r, W//r]
        x_reshaped = x_downsampled.reshape(T, B, C * self.r**2, H//self.r, W//self.r)
        
        # 4. Rearrange: separate patch information into independent dimensions
        # [T, B, C*r^2, H//r, W//r] -> [T, B, C, r^2, H//r, W//r]
        x_patch_separated = x_reshaped.reshape(T, B, C, self.r**2, H//self.r, W//self.r)
        
        # 5. Merge patch dimension with time dimension to create a new time series
        # [T, B, C, r^2, H//r, W//r] -> [T*r^2, B, C, H//r, W//r]
        x_temporal = x_patch_separated.permute(0, 3, 1, 2, 4, 5).contiguous()
        x_temporal = x_temporal.reshape(T * self.r**2, B, C, H//self.r, W//self.r)
        
        # 6. Process through LIF neurons
        # Now time steps become T*r^2, each time step corresponds to a patch position in the original image
        x_lif_output = self.lif_node(x_temporal)
        
        # 7. Reshape back to original shape directly, without using channel_adjust
        # [T*r^2, B, C, H//r, W//r] -> [T, r^2, B, C, H//r, W//r]
        x_reorganized = x_lif_output.reshape(T, self.r**2, B, C, H//self.r, W//self.r)
        
        # 8. Rearrange back to original spatial layout
        # [T, r^2, B, C, H//r, W//r] -> [T, B, C, r^2, H//r, W//r]
        x_reorganized = x_reorganized.permute(0, 2, 3, 1, 4, 5).contiguous()
        
        # 9. Reshape back to original spatial dimensions directly
        # [T, B, C, r^2, H//r, W//r] -> [T, B, C*r^2, H//r, W//r]
        x_reorganized = x_reorganized.reshape(T, B, C * self.r**2, H//self.r, W//self.r)
        
        # 10. Use bilinear interpolation to restore original spatial dimensions
        # [T*B, C*r^2, H//r, W//r] -> [T*B, C*r^2, H, W]
        x_flat_reorganized = x_reorganized.reshape(T * B, C * self.r**2, H//self.r, W//self.r)
        x_upsampled = F.interpolate(x_flat_reorganized, size=(H, W), mode='bilinear', align_corners=False)
        
        # 11. Restore time and batch dimensions
        # [T*B, C*r^2, H, W] -> [T, B, C*r^2, H, W]
        x_restored = x_upsampled.reshape(T, B, C * self.r**2, H, W)
        
        # 12. Use channel_adjust to adjust channel count back to original dimension
        # [T, B, C*r^2, H, W] -> [T, B, C, H, W]
        x_final = self.channel_adjust(x_restored)
        
        return x_final

# TimeAttention from layer.py
class TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(TimeAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SUNet_Level1_Block(nn.Module):
    """
    SUNet module for Level 1
    Flow:
    1. Spiking_Residual_Block
    2. PixelShuffleLIFBlock downsampling (maintain extended time dimension)
    3. TimeAttention + 3D Conv to compress time dimension
    4. 2 [LIF-Conv2d-BatchNorm-MultiDimensionalAttention] blocks
    5. Upsample back
    6. Skip connection + MultiDimensionalAttention
    """
    def __init__(self, dim):
        super(SUNet_Level1_Block, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        
        self.downsample_factor = 2
        self.r = 2
        
        # 1. Initial Spiking_Residual_Block
        self.initial_residual = Spiking_Residual_Block(dim=dim)
        
        # 2. Custom PixelShuffleLIF, maintaining extended time dimension
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downsample_factor)
        
        self.lif_node = mem_update()

        # 3. TimeAttention processing extended time dimension (T*r^2 = 4*4 = 16)
        self.time_attention = TimeAttention(in_planes=16, ratio=4)  # T*r^2 = 16
        
        # 4. 3D convolution compresses time dimension from T*r^2 back to T
        self.temporal_compress = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(4, 1, 1), stride=(4, 1, 1), padding=0, bias=False),  # Compress time dimension
            nn.ReLU(inplace=True)
        )
        
        # 5. Two processing blocks: [LIF-Conv2d-BatchNorm-MultiDimensionalAttention]
        # First processing block
        self.lif_1 = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.conv_1 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.bn_1 = layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True)
        self.attn_1 = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)
        
        # Second processing block
        self.lif_2 = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.conv_2 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.bn_2 = layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True)
        self.attn_2 = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)
        
        # 6. Upsampling layer (using bilinear interpolation)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # 7. Final MultiDimensionalAttention
        self.final_attn = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)
    
    def forward(self, x):
        # 1. Pass through Initial Spiking_Residual_Block
        residual_out = self.initial_residual(x)
        
        # Save features for skip connection
        skip_features = residual_out.clone()
        
        # 2. Custom PixelShuffleLIF downsampling, maintaining extended time dimension
        T, B, C, H, W = residual_out.shape
        
        # Flatten time and batch dimensions for 2D operations
        x_flat = residual_out.reshape(T * B, C, H, W)
        
        # Pixel unshuffle downsampling
        x_downsampled = self.pixel_unshuffle(x_flat)  # [T*B, C*r^2, H//r, W//r]
        
        # Restore time and batch dimensions
        x_reshaped = x_downsampled.reshape(T, B, C * self.r**2, H//self.r, W//self.r)
        
        # Rearrange: separate patch information into independent dimensions
        x_patch_separated = x_reshaped.reshape(T, B, C, self.r**2, H//self.r, W//self.r)
        
        # Merge patch dimension with time dimension to create a new time series
        x_temporal = x_patch_separated.permute(0, 3, 1, 2, 4, 5).contiguous()
        downsampled = x_temporal.reshape(T * self.r**2, B, C, H//self.r, W//self.r)
        
        # Process through LIF neurons, maintaining extended time dimension
        downsampled = self.lif_node(downsampled)
        # downsampled shape: [T*r^2, B, C, H//r, W//r] = [16, B, C, H//2, W//2]
        
        # 3. TimeAttention processing extended time dimension
        # Need to convert to [B, T*r^2, C, H//r, W//r] format for TimeAttention
        T_extended, B, C, H_down, W_down = downsampled.shape
        downsampled_transposed = downsampled.transpose(0, 1)  # [B, T*r^2, C, H//r, W//r]
        
        # Apply TimeAttention
        time_att = self.time_attention(downsampled_transposed)  # [B, T*r^2, C, H//r, W//r]
        attended = downsampled_transposed * time_att
        
        # Convert back to [T*r^2, B, C, H//r, W//r] format
        attended = attended.transpose(0, 1)  # [T*r^2, B, C, H//r, W//r]
        
        # 4. Use 3D convolution to compress time dimension from T*r^2 back to T
        # Convert to format required by 3D convolution: [B, C, T*r^2, H//r, W//r]
        attended_3d = attended.permute(1, 2, 0, 3, 4)  # [B, C, T*r^2, H//r, W//r]
        
        # Apply 3D convolution to compress time dimension
        compressed = self.temporal_compress(attended_3d)  # [B, C, T, H//r, W//r]
        
        # Convert back to [T, B, C, H//r, W//r] format
        out = compressed.permute(2, 0, 1, 3, 4)  # [T, B, C, H//r, W//r]
        
        # 5. First processing block
        out = self.lif_1(out)
        out = self.conv_1(out)
        out = self.bn_1(out)
        out = self.attn_1(out)
        
        # Second processing block
        out = self.lif_2(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.attn_2(out)
        
        # 6. Upsample back to original size
        # Need to handle time dimension
        T, B, C, H, W = out.shape
        # Flatten time and batch dimensions for upsampling
        out_flat = out.reshape(T * B, C, H, W)
        upsampled_flat = self.upsample(out_flat)
        # Restore time and batch dimensions
        upsampled = upsampled_flat.reshape(T, B, C, upsampled_flat.shape[2], upsampled_flat.shape[3])
        
        # 7. Skip connection
        combined = upsampled + skip_features
        
        # 8. Final MultiDimensionalAttention
        final_out = self.final_attn(combined)
        
        return final_out

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32, spike_mode="lif", LayerNorm_type='WithBias', bias=False, T=4):
        super(OverlapPatchEmbed, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.proj = layer.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        # x shape: [T, N, C, H, W]
        x = self.proj(x)
        return x

class Spiking_Residual_Block(nn.Module):
    def __init__(self, dim):
        super(Spiking_Residual_Block, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        
        # Use traditional LIF as high-frequency filter
        self.lif_1 = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.conv1 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.bn1 = layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True)
        
        # Learnable parameters to scale high and low frequency features (1st group)
        self.high_freq_scale_1 = nn.Parameter(torch.ones(1))
        self.low_freq_scale_1 = nn.Parameter(torch.ones(1))
        
        # Use PixelShuffleLIFBlock as high-frequency filter (2nd group)
        self.lif_2 = PixelShuffleLIFBlock(in_channels=dim, downsample_factor=2)
        self.conv2 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.bn2 = layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th * 0.2, affine=True)
        
        # Learnable parameters to scale high and low frequency features (2nd group)
        self.high_freq_scale_2 = nn.Parameter(torch.ones(1))
        self.low_freq_scale_2 = nn.Parameter(torch.ones(1))
        
        self.shortcut = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                         bias=False, step_mode='m'),
            layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha,
                                                v_th=v_th, affine=True),
        )

        self.attn = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)

    def forward(self, x):
        # 1st group: Use LIF as high-frequency filter
        # 1. Use LIF as high-frequency filter
        x_h_1 = self.lif_1(x)  # High-frequency features
        
        # 2. Obtain low-frequency features by X - x_h
        x_l_1 = x - x_h_1  # Low-frequency features
        
        # 3. Use learnable parameters to scale high and low frequency features
        x_h_1_scaled = self.high_freq_scale_1 * x_h_1
        x_l_1_scaled = self.low_freq_scale_1 * x_l_1
        
        # 4. Element-wise enhancement of input feature X using original high-frequency feature x_h
        x_enhanced_1 = x * x_h_1  # element-wise multiplication
        
        # 5. Sum all features
        combined_features_1 = x_h_1_scaled + x_l_1_scaled + x_enhanced_1
        
        # 6. Continue through convolution and batchnorm
        out = self.conv1(combined_features_1)
        out = self.bn1(out)
        
        # 2nd group: Apply same high/low frequency processing to out
        # 1. Use LIF as high-frequency filter
        x_h_2 = self.lif_2(out)  # High-frequency features
        
        # 2. Obtain low-frequency features by out - x_h_2
        x_l_2 = out - x_h_2  # Low-frequency features
        
        # 3. Use learnable parameters to scale high and low frequency features
        x_h_2_scaled = self.high_freq_scale_2 * x_h_2
        x_l_2_scaled = self.low_freq_scale_2 * x_l_2
        
        # 4. Element-wise enhancement of input feature out using original high-frequency feature x_h_2
        x_enhanced_2 = out * x_h_2  # element-wise multiplication
        
        # 5. Sum all features
        combined_features_2 = x_h_2_scaled + x_l_2_scaled + x_enhanced_2
        
        # 6. Continue through convolution and batchnorm
        out = self.conv2(combined_features_2)
        out = self.bn2(out)
        
        # Final combination
        shortcut = torch.clone(x)
        out = out + self.shortcut(shortcut)
        out = self.attn(out) + shortcut
        return out


class DownSampling(nn.Module):
    def __init__(self, dim):
        super(DownSampling, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        
        # Use traditional LIF instead of PixelShuffleLIFBlock
        self.lif = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.conv = layer.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, step_mode='m', bias=False)
        self.bn = layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim * 2, affine=True)

    def forward(self, x):
        x = self.lif(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class UpSampling(nn.Module):
    def __init__(self, dim):
        super(UpSampling, self).__init__()
        self.scale_factor = 2
        
        # Use traditional LIF instead of PixelShuffleLIFBlock
        self.lif = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.conv = layer.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, step_mode='m', bias=False)
        self.bn = layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim // 2, affine=True)

    def forward(self, input):
        # First perform bilinear interpolation upsampling
        temp = torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3] * self.scale_factor,
                            input.shape[4] * self.scale_factor)).cuda()
        output = []
        for i in range(input.shape[0]):
            temp[i] = F.interpolate(input[i], scale_factor=self.scale_factor, mode='bilinear')
            output.append(temp[i])
        out = torch.stack(output, dim=0)
        
        # Then pass through traditional LIF and convolution layers
        out = self.lif(out)
        out = self.conv(out)
        out = self.bn(out)
        return out


class VLIFNet(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=24, en_num_blocks=[4, 4, 6, 6], de_num_blocks=[4, 4, 6, 6],
                 bias=False, T=4, use_refinement=False):
        super(VLIFNet, self).__init__()

        functional.set_backend(self, backend='cupy')
        functional.set_step_mode(self, step_mode='m')

        self.T = T
        self.use_refinement = use_refinement
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, T=T)
        # Use SUNet_Level1_Block instead of the original Spiking_Residual_Block sequence
        self.encoder_level1 = SUNet_Level1_Block(dim=int(dim * 1))

        self.down1_2 = DownSampling(dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 1)) for i in range(en_num_blocks[1])])

        self.down2_3 = DownSampling(int(dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 2)) for i in range(en_num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 2)) for i in range(de_num_blocks[2])])

        self.up3_2 = UpSampling(int(dim * 2 ** 2))  ## From Level 3 to Level 2

        # Use traditional LIF instead of PixelShuffleLIFBlock
        self.lif_level2 = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.reduce_conv_level2 = layer.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias, step_mode='m')
        self.reduce_bn_level2 = layer.ThresholdDependentBatchNorm2d(num_features=int(dim * 2 ** 1), alpha=alpha, v_th=v_th)

        self.decoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 1)) for i in range(de_num_blocks[1])])

        self.up2_1 = UpSampling(int(dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        # Use traditional LIF instead of PixelShuffleLIFBlock
        self.lif_level1 = LIFNode(v_threshold=v_th, backend='cupy', step_mode='m', decay_input=False)
        self.reduce_conv_level1 = layer.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 0), kernel_size=1, bias=bias, step_mode='m')
        self.reduce_bn_level1 = layer.ThresholdDependentBatchNorm2d(num_features=int(dim * 2 ** 0), alpha=alpha, v_th=v_th)

        # Use SUNet_Level1_Block instead of the original Spiking_Residual_Block sequence
        self.decoder_level1 = SUNet_Level1_Block(dim=int(dim * 2 ** 0))
        
        # Add extra SUNet_Level1_Block
        self.additional_sunet_level1 = SUNet_Level1_Block(dim=int(dim * 2 ** 0))
        
        # Add refinement blocks only if use_refinement is True (for RainL200)
        if self.use_refinement:
            self.refinement_blocks = nn.Sequential(*[
                Spiking_Residual_Block(dim=int(dim * 2 ** 0)) for i in range(4)
            ])

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=int(dim * 2 ** 0), out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1)
        )

    def forward(self, inp_img):
        short = inp_img.clone()
        ############ Repeat Feature  ################
        if len(inp_img.shape) < 5:
            inp_img = (inp_img.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)

        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=2)

        # Use traditional LIF structure
        inp_dec_level2 = self.lif_level2(inp_dec_level2)
        inp_dec_level2 = self.reduce_conv_level2(inp_dec_level2)
        inp_dec_level2 = self.reduce_bn_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=2)
        
        # Use traditional LIF structure
        inp_dec_level1 = self.lif_level1(inp_dec_level1)
        inp_dec_level1 = self.reduce_conv_level1(inp_dec_level1)
        inp_dec_level1 = self.reduce_bn_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        
        # Pass through SUNet_Level1_Block again
        out_dec_level1 = self.additional_sunet_level1(out_dec_level1)
        
        # Pass through refinement blocks if use_refinement is True (for RainL200)
        if self.use_refinement:
            out_dec_level1 = self.refinement_blocks(out_dec_level1)

        ############ Image Reconstruction  ################
        # Skip refinement, go directly through output layer and add short
        out_dec_level1 = (self.output(out_dec_level1.mean(0))) + short
        return out_dec_level1


def model(use_refinement=False):
    return VLIFNet(dim=48, en_num_blocks=[4, 4, 8, 8], de_num_blocks=[2, 2, 2, 2], T=4, use_refinement=use_refinement).cuda()
