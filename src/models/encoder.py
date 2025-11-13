"""
Module 5: U-Net Encoder
Learns to extract structural features (F_Tensor) from rendered designs (P_Image)

Architecture: Standard U-Net with skip connections
Input: P_Image [B, 3, 256, 256]
Output: F_Tensor [B, 4, 256, 256]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double Convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downsampling block: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upsampling block: Upsample -> Conv -> Concat with skip -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch (if any)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output convolution layer
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetEncoder(nn.Module):
    """
    U-Net for extracting F_Tensor from P_Image

    Output channels:
        K=0: Text Mask (sigmoid)
        K=1: Image Mask (sigmoid)
        K=2: Color ID Map (logits, 18 classes)
        K=3: Hierarchy Map (sigmoid)
    """
    def __init__(self, n_channels=3, n_color_classes=18, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_color_classes = n_color_classes
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # Output heads for each channel
        self.out_text_mask = OutConv(64, 1)      # K=0: Text Mask
        self.out_image_mask = OutConv(64, 1)     # K=1: Image Mask
        self.out_color_id = OutConv(64, n_color_classes)  # K=2: Color ID logits
        self.out_hierarchy = OutConv(64, 1)      # K=3: Hierarchy Map

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 64 channels, 256x256
        x2 = self.down1(x1)   # 128 channels, 128x128
        x3 = self.down2(x2)   # 256 channels, 64x64
        x4 = self.down3(x3)   # 512 channels, 32x32
        x5 = self.down4(x4)   # 512 channels, 16x16 (bottleneck)

        # Decoder with skip connections
        x = self.up1(x5, x4)  # 256 channels, 32x32
        x = self.up2(x, x3)   # 128 channels, 64x64
        x = self.up3(x, x2)   # 64 channels, 128x128
        x = self.up4(x, x1)   # 64 channels, 256x256

        # Output heads
        text_mask = torch.sigmoid(self.out_text_mask(x))      # [B, 1, 256, 256]
        image_mask = torch.sigmoid(self.out_image_mask(x))    # [B, 1, 256, 256]
        color_logits = self.out_color_id(x)                   # [B, 18, 256, 256]
        hierarchy = torch.sigmoid(self.out_hierarchy(x))      # [B, 1, 256, 256]

        # Stack outputs: [B, 4, 256, 256]
        # Note: For color_id, we return logits during training, argmax during inference
        return {
            'text_mask': text_mask,
            'image_mask': image_mask,
            'color_logits': color_logits,
            'hierarchy': hierarchy
        }

    def predict(self, x):
        """
        Inference mode: return F_Tensor in standard format
        """
        outputs = self.forward(x)

        # Convert color logits to class predictions
        color_ids = torch.argmax(outputs['color_logits'], dim=1, keepdim=True).float()

        # Stack into F_Tensor format [B, 4, 256, 256]
        f_tensor = torch.cat([
            outputs['text_mask'],
            outputs['image_mask'],
            color_ids,
            outputs['hierarchy']
        ], dim=1)

        return f_tensor


class CompositeLoss(nn.Module):
    """
    Multi-task loss for U-Net Encoder

    Components:
    1. DiceLoss for binary masks (K=0, K=1)
    2. CrossEntropyLoss for color IDs (K=2)
    3. MSELoss for hierarchy map (K=3)
    """
    def __init__(self, dice_weight=1.0, ce_weight=0.5, mse_weight=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.mse_weight = mse_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        """
        Dice loss for binary segmentation
        pred: [B, 1, H, W]
        target: [B, 1, H, W]
        """
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

        return 1 - dice

    def forward(self, predictions, targets):
        """
        predictions: dict with keys ['text_mask', 'image_mask', 'color_logits', 'hierarchy']
        targets: F_Tensor [B, 4, H, W]
        """
        # Extract target channels
        target_text = targets[:, 0:1, :, :]      # [B, 1, H, W]
        target_image = targets[:, 1:2, :, :]     # [B, 1, H, W]
        target_color = targets[:, 2, :, :].long()  # [B, H, W] (class indices)
        target_hierarchy = targets[:, 3:4, :, :]  # [B, 1, H, W]

        # 1. Dice loss for masks
        dice_text = self.dice_loss(predictions['text_mask'], target_text)
        dice_image = self.dice_loss(predictions['image_mask'], target_image)
        total_dice = dice_text + dice_image

        # 2. Cross-entropy for color IDs
        ce_color = self.ce_loss(predictions['color_logits'], target_color)

        # 3. MSE for hierarchy
        mse_hierarchy = self.mse_loss(predictions['hierarchy'], target_hierarchy)

        # Weighted combination
        total_loss = (
            self.dice_weight * total_dice +
            self.ce_weight * ce_color +
            self.mse_weight * mse_hierarchy
        )

        # Return loss components for logging
        return total_loss, {
            'total': total_loss.item(),
            'dice': total_dice.item(),
            'dice_text': dice_text.item(),
            'dice_image': dice_image.item(),
            'ce_color': ce_color.item(),
            'mse_hierarchy': mse_hierarchy.item()
        }


def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics

    Returns:
        dict with IoU, accuracy, MSE for each component
    """
    metrics = {}

    with torch.no_grad():
        # Text mask IoU
        pred_text = (predictions['text_mask'] > 0.5).float()
        target_text = targets[:, 0:1, :, :]
        intersection = (pred_text * target_text).sum()
        union = pred_text.sum() + target_text.sum() - intersection
        metrics['text_iou'] = (intersection / (union + 1e-6)).item()

        # Image mask IoU
        pred_image = (predictions['image_mask'] > 0.5).float()
        target_image = targets[:, 1:2, :, :]
        intersection = (pred_image * target_image).sum()
        union = pred_image.sum() + target_image.sum() - intersection
        metrics['image_iou'] = (intersection / (union + 1e-6)).item()

        # Color ID accuracy
        pred_color = torch.argmax(predictions['color_logits'], dim=1)
        target_color = targets[:, 2, :, :].long()
        correct = (pred_color == target_color).float().sum()
        total = target_color.numel()
        metrics['color_accuracy'] = (correct / total).item()

        # Hierarchy MSE
        metrics['hierarchy_mse'] = F.mse_loss(
            predictions['hierarchy'],
            targets[:, 3:4, :, :]
        ).item()

    return metrics


if __name__ == '__main__':
    # Test the encoder
    print("=" * 60)
    print("U-Net Encoder Test")
    print("=" * 60)

    # Create model
    model = UNetEncoder(n_channels=3, n_color_classes=18)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    print(f"\nInput shape: {x.shape}")

    outputs = model(x)
    print(f"\nOutput shapes:")
    for key, tensor in outputs.items():
        print(f"  {key}: {tensor.shape}")

    # Test predict mode
    f_tensor = model.predict(x)
    print(f"\nF_Tensor (predict mode): {f_tensor.shape}")

    # Test loss
    criterion = CompositeLoss()
    fake_target = torch.randn(batch_size, 4, 256, 256)
    fake_target[:, 0:2, :, :] = (fake_target[:, 0:2, :, :] > 0).float()  # Binary masks
    fake_target[:, 2, :, :] = torch.randint(0, 18, (batch_size, 256, 256)).float()  # Color IDs
    fake_target[:, 3:4, :, :] = torch.sigmoid(fake_target[:, 3:4, :, :])  # Hierarchy [0,1]

    loss, loss_dict = criterion(outputs, fake_target)
    print(f"\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # Test metrics
    metrics = calculate_metrics(outputs, fake_target)
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("Encoder architecture test passed!")
    print("=" * 60)
