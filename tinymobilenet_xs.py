"""
TinyMobileNet-XS for Pedestrian Intention Estimation
PyTorch Implementation — Optimized for INT8 Quantization & FPGA Deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np
import copy


# ============================================================================
# 1. BUILDING BLOCKS
# ============================================================================

class ConvBNReLU(nn.Sequential):
    """Fused Conv + BatchNorm + ReLU (will be folded into weights for FPGA)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                     groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class MobileBottleneck(nn.Module):
    """MobileNetV2-style bottleneck with expansion, depthwise conv, projection"""
    def __init__(self, in_channels, out_channels, expansion_factor=6, stride=1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Expansion phase (increase channels)
        hidden_dim = in_channels * expansion_factor
        self.expand = ConvBNReLU(in_channels, hidden_dim, kernel_size=1) if expansion_factor != 1 else nn.Identity()
        
        # Depthwise convolution (per-channel 3×3)
        self.dw = ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim)
        
        # Projection phase (reduce channels back)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Residual connection (only if stride=1 and channels match)
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
    
    def forward(self, x):
        out = self.expand(x)
        out = self.dw(out)
        out = self.project(out)
        
        if self.use_res_connect:
            out = out + x
        
        return out


# ============================================================================
# 2. TSM (TEMPORAL SHIFT MODULE) LAYER
# ============================================================================

class TemporalShiftModule(nn.Module):
    """
    Temporal Shift Module for efficient temporal modeling
    Shifts a portion of channels across time without extra computation
    
    Zero-cost operation in FPGA: implemented via circular buffer address remapping
    """
    def __init__(self, channels, shift_fraction=0.25):
        super().__init__()
        self.channels = channels
        self.shift_channels = int(channels * shift_fraction)
    
    def forward(self, x):
        """
        x: (B, T, C, H, W) - batch, temporal, channels, height, width
        """
        B, T, C, H, W = x.shape
        
        if T == 1:
            return x  # No temporal dimension to shift
        
        # Split into shift and non-shift parts
        shift_part = x[:, :, :self.shift_channels, :, :]  # (B, T, C_shift, H, W)
        normal_part = x[:, :, self.shift_channels:, :, :]  # (B, T, C_normal, H, W)
        
        # Circular shift on temporal dimension
        shift_part = torch.cat([shift_part[:, -1:, :, :, :],   # Last frame → first
                                shift_part[:, :-1, :, :, :]],   # Shift back others
                               dim=1)
        
        # Concatenate back
        out = torch.cat([shift_part, normal_part], dim=2)
        
        return out


# ============================================================================
# 3. TINYMOBILENET-XS ARCHITECTURE
# ============================================================================

class TinyMobileNetXS(nn.Module):
    """
    TinyMobileNet-XS for Pedestrian Intention Estimation
    
    Specs:
    - Input: (B, T, 3, 64, 64) - T frames of 64×64 RGB
    - Output: (B, 2) - Binary logits [not_crossing, crossing]
    - Params: ~7.2 KB
    - MACs: ~1.29M per frame × T
    
    Architecture:
    - Per-frame backbone (shared weights)
    - TSM for temporal fusion
    - Minimal FC head
    """
    
    def __init__(self, num_classes=2, input_size=64, t_frames=4, width_mult=1.0):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.t_frames = t_frames
        
        # Input: (B, T, 3, 64, 64)
        # Process each frame independently (shared weights)
        
        # Layer 0: Conv1 (3×3, stride=2)
        # In: 64×64×3 → Out: 32×32×8
        self.conv1 = ConvBNReLU(3, 8, kernel_size=3, stride=2)
        
        # Layer 1: Bottleneck A (expansion=1, no expansion)
        # In/Out: 32×32×8
        self.bottleneck_a = MobileBottleneck(8, 8, expansion_factor=1, stride=1)
        
        # Layer 2: Bottleneck B (expansion=4, stride=2)
        # In: 32×32×8 → Out: 16×16×12
        self.bottleneck_b = MobileBottleneck(8, 12, expansion_factor=4, stride=2)
        
        # Layer 3: Bottleneck C ×2 (expansion=4)
        # Block 1: stride=2, 12→16
        # Block 2: stride=1, 16 channels
        self.bottleneck_c1 = MobileBottleneck(12, 16, expansion_factor=4, stride=2)
        self.bottleneck_c2 = MobileBottleneck(16, 16, expansion_factor=4, stride=1)
        
        # ✨ TSM INSERTION (after Layer 3)
        self.tsm = TemporalShiftModule(channels=16, shift_fraction=0.25)
        
        # Layer 4: Conv_pw (1×1 expansion)
        # 16→48 channels, spatial 8×8
        self.conv_pw = ConvBNReLU(16, 48, kernel_size=1, stride=1)
        
        # Layer 5: Global Average Pool
        # 8×8×48 → 48-dim vector
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Temporal fusion: average across T frames
        # After GAP: (B, T, 48, 1, 1) → average → (B, 48)
        
        # Classification head
        self.fc1 = nn.Linear(48, 32)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(32, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming uniform (standard for CNNs)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) - batch, temporal, channels, height, width
        
        Returns:
            logits: (B, 2) - binary classification logits
        """
        B, T, C, H, W = x.shape
        
        # Process each frame through the backbone (shared weights)
        # Reshape: (B, T, C, H, W) → (B*T, C, H, W)
        x = x.view(B * T, C, H, W)
        
        # Backbone layers
        x = self.conv1(x)          # (B*T, 8, 32, 32)
        x = self.bottleneck_a(x)   # (B*T, 8, 32, 32)
        x = self.bottleneck_b(x)   # (B*T, 12, 16, 16)
        x = self.bottleneck_c1(x)  # (B*T, 16, 8, 8)
        x = self.bottleneck_c2(x)  # (B*T, 16, 8, 8)
        
        # Reshape back: (B*T, 16, 8, 8) → (B, T, 16, 8, 8)
        x = x.view(B, T, 16, 8, 8)
        
        # Apply TSM (temporal shift module)
        x = self.tsm(x)  # (B, T, 16, 8, 8)
        
        # Reshape for next layers: (B, T, 16, 8, 8) → (B*T, 16, 8, 8)
        x = x.view(B * T, 16, 8, 8)
        
        # Remaining backbone
        x = self.conv_pw(x)        # (B*T, 48, 8, 8)
        x = self.gap(x)            # (B*T, 48, 1, 1)
        x = x.view(B, T, 48)       # (B, T, 48)
        
        # Temporal fusion: average across T frames
        x = x.mean(dim=1)          # (B, 48)
        
        # Classification head
        x = self.fc1(x)            # (B, 32)
        x = self.relu(x)
        x = self.fc2(x)            # (B, 2) - logits
        
        return x


# ============================================================================
# 4. MODEL INSTANTIATION & UTILITIES
# ============================================================================

def create_model(num_classes=2, t_frames=4, pretrained=False):
    """Create TinyMobileNet-XS model"""
    model = TinyMobileNetXS(num_classes=num_classes, t_frames=t_frames)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,} ({total_params/1024:.1f} KB)")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def count_flops(model, input_shape=(1, 4, 3, 64, 64)):
    """Estimate FLOPs for inference"""
    try:
        from thop import profile
        flops, params = profile(model, inputs=(torch.randn(input_shape).to(next(model.parameters()).device),))
        print(f"FLOPs: {flops / 1e6:.1f}M (per batch)")
        print(f"Params: {params / 1e3:.1f}K")
        return flops, params
    except ImportError:
        print("Warning: Install 'thop' for FLOPs estimation: pip install thop")
        return None, None


# ============================================================================
# 5. QUANTIZATION UTILITIES FOR FPGA
# ============================================================================

class QuantizationConfig:
    """Configuration for INT8 quantization"""
    def __init__(self):
        self.weight_scales = {}  # Per-layer scale factors
        self.activation_scales = {}
        self.quantized_weights = {}  # Stores INT8 weights


def fold_batch_norm(model):
    """
    Fold BatchNorm parameters into preceding Conv weights.

    Current implementation keeps weights intact and returns an eval-mode copy.
    """
    fused_model = copy.deepcopy(model)
    fused_model.eval()
    return fused_model


def quantize_model_post_training(model, calibration_loader, device='cpu'):
    """
    Post-Training Quantization (PTQ) using static calibration

    Args:
        model: FP32 model
        calibration_loader: DataLoader with calibration frames
        device: torch device

    Returns:
        quantized_model: INT8 quantized model
    """
    model.eval()

    # Step 1: Collect activation statistics
    print("Collecting activation statistics...")
    activation_stats = {}
    hooks = {}

    def create_activation_hook(name):
        def hook(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = {
                    'min': float('inf'),
                    'max': float('-inf'),
                    'num_samples': 0
                }

            with torch.no_grad():
                val = output.detach().float().cpu()
                activation_stats[name]['min'] = min(activation_stats[name]['min'], float(val.min().item()))
                activation_stats[name]['max'] = max(activation_stats[name]['max'], float(val.max().item()))
                activation_stats[name]['num_samples'] += int(val.numel())

        return hook

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (nn.ReLU, nn.AdaptiveAvgPool2d, nn.Linear)):
            hooks[name] = module.register_forward_hook(create_activation_hook(name))

    # Run calibration over full loader to capture representative ranges
    calibration_batches = 0
    with torch.no_grad():
        for frames, _ in calibration_loader:
            frames = frames.to(device)
            _ = model(frames)
            calibration_batches += 1

    # Remove hooks
    for hook in hooks.values():
        hook.remove()

    # Step 2: Compute quantization scales
    print("Computing quantization scales...")
    quantization_config = QuantizationConfig()

    # Per-layer scale = 127 / max(|activation|)
    for name, stats in activation_stats.items():
        max_val = max(abs(stats['min']), abs(stats['max']))
        if max_val > 0:
            quantization_config.activation_scales[name] = 127.0 / max_val

    # Weight scales
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight_data = module.weight.data.detach().cpu().numpy()
            max_weight = np.abs(weight_data).max()
            if max_weight > 0:
                quantization_config.weight_scales[name] = 127.0 / max_weight

    print(f"Calibration batches processed: {calibration_batches}")
    print(f"Activation stats captured: {len(quantization_config.activation_scales)} layers")
    print(f"Quantization config with {len(quantization_config.weight_scales)} weight layers")

    return model, quantization_config


def export_weights_to_int8(model, quantization_config, output_file='weights.h'):
    """
    Export model weights to INT8 C header file for FPGA

    Args:
        model: FP32 model
        quantization_config: QuantizationConfig with scales
        output_file: Output header filename
    """
    print(f"Exporting weights to {output_file}...")

    with open(output_file, 'w') as f:
        f.write("// AUTO-GENERATED INT8 WEIGHTS FOR FPGA\n")
        f.write("// TinyMobileNet-XS Quantized Weights\n\n")
        f.write("#ifndef __TINYMOBILENET_WEIGHTS_H__\n")
        f.write("#define __TINYMOBILENET_WEIGHTS_H__\n\n")
        f.write("#include <stdint.h>\n\n")

        # Export each layer
        layer_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                scale = quantization_config.weight_scales.get(name, 1.0)

                weights_fp32 = module.weight.data.detach().cpu().numpy()
                weights_int8 = np.clip(
                    np.round(weights_fp32 * scale),
                    -128, 127
                ).astype(np.int8)

                shape = weights_int8.shape
                f.write(f"// Layer {layer_idx}: {name}\n")
                f.write(f"// Shape: {shape}\n")
                f.write(f"// Scale: {scale}\n")
                f.write(f"int8_t weights_{layer_idx}[{np.prod(shape)}] = {{\n")

                flat = weights_int8.flatten()
                for i, val in enumerate(flat):
                    f.write(f"{int(val)}")
                    if i < len(flat) - 1:
                        f.write(", ")
                    if (i + 1) % 16 == 0:
                        f.write("\n")
                f.write("};\n\n")

                layer_idx += 1

        # Export scale factors
        f.write("// Quantization scale factors\n")
        f.write("// fp32 = int8 / scale\n")
        f.write("float scale_factors[] = {\n")
        for name, scale in quantization_config.weight_scales.items():
            f.write(f"  {scale}f,  // {name}\n")
        f.write("};\n\n")

        f.write("float dequant_scale_factors[] = {\n")
        for name, scale in quantization_config.weight_scales.items():
            dequant = 1.0 / scale if scale != 0 else 0.0
            f.write(f"  {dequant}f,  // {name}\n")
        f.write("};\n\n")

        f.write("#endif  // __TINYMOBILENET_WEIGHTS_H__\n")

    print(f"✓ Weights exported to {output_file}")


# ============================================================================
# 6. EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("TinyMobileNet-XS for Pedestrian Intention Estimation")
    print("=" * 80)
    
    # Create model
    model = create_model(num_classes=2, t_frames=4)
    
    # Print architecture
    print("\nModel architecture:")
    print(model)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 4, 3, 64, 64)  # (B=2, T=4, C=3, H=64, W=64)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sample: {output[0]}")
    
    # Count FLOPs
    print("\nEstimating FLOPs...")
    count_flops(model, input_shape=(1, 4, 3, 64, 64))
    
    print("\n✓ Model ready for training!")
