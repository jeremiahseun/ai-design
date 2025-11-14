"""
Abstractor Model (Module 6)
Predicts V_Grammar and V_Meta from F_Tensor

Architecture:
    F_Tensor [B,4,256,256] → ResNet-18 → Global Pool → FC-512
                              ├→ MLP Head 1 → V_Meta (goal, tone, format)
                              └→ MLP Head 2 → V_Grammar [4]

Input:  F_Tensor [B, 4, 256, 256]
Output: V_Grammar [B, 4] + V_Meta components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict, Tuple


class Abstractor(nn.Module):
    """
    ResNet-18 based abstractor with dual MLP heads

    Predicts:
    - V_Grammar: [alignment, contrast, whitespace, hierarchy] (4 continuous values)
    - V_Meta: goal (classification), tone (regression), format (classification)
    """

    def __init__(self,
                 n_goal_classes: int = 4,      # inform, persuade, entertain, inspire
                 n_format_classes: int = 3,    # poster, social, flyer
                 pretrained: bool = True):
        super().__init__()

        self.n_goal_classes = n_goal_classes
        self.n_format_classes = n_format_classes

        # ResNet-18 backbone (pretrained on ImageNet)
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # Modify first conv layer to accept 4 channels (F_Tensor) instead of 3
        # We'll initialize the 4th channel with average of RGB weights
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            # Copy RGB weights and initialize 4th channel as average
            with torch.no_grad():
                self.conv1.weight[:, :3] = original_conv1.weight
                self.conv1.weight[:, 3] = original_conv1.weight.mean(dim=1)

        # Use remaining ResNet layers (excluding first conv and fc)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Feature dimension after ResNet
        feature_dim = 512  # ResNet-18 output dimension

        # Shared FC layer
        self.fc_shared = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Head 1: V_Meta prediction
        self.meta_head = nn.ModuleDict({
            'goal': nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_goal_classes)
            ),
            'tone': nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1),
                nn.Sigmoid()  # Tone is in [0, 1]
            ),
            'format': nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, n_format_classes)
            )
        })

        # Head 2: V_Grammar prediction (4 continuous values)
        self.grammar_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),
            nn.Sigmoid()  # Grammar scores are in [0, 1]
        )

    def forward(self, f_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            f_tensor: [B, 4, 256, 256]

        Returns:
            Dictionary with predictions:
            - 'v_goal': [B, n_goal_classes] (logits)
            - 'v_tone': [B, 1]
            - 'v_format': [B, n_format_classes] (logits)
            - 'v_grammar': [B, 4]
        """
        # ResNet forward pass
        x = self.conv1(f_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling
        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]

        # Shared features
        features = self.fc_shared(x)  # [B, 512]

        # V_Meta predictions
        v_goal_logits = self.meta_head['goal'](features)  # [B, n_goal_classes]
        v_tone = self.meta_head['tone'](features)  # [B, 1]
        v_format_logits = self.meta_head['format'](features)  # [B, n_format_classes]

        # V_Grammar predictions
        v_grammar = self.grammar_head(features)  # [B, 4]

        return {
            'v_goal': v_goal_logits,
            'v_tone': v_tone,
            'v_format': v_format_logits,
            'v_grammar': v_grammar
        }


class AbstractorLoss(nn.Module):
    """
    Composite loss for Abstractor training

    Loss = CE(v_Goal) + CE(v_Format) + MSE(v_Tone) + MSE(v_Grammar)
    """

    def __init__(self,
                 weight_goal: float = 1.0,
                 weight_format: float = 1.0,
                 weight_tone: float = 1.0,
                 weight_grammar: float = 1.0):
        super().__init__()

        self.weight_goal = weight_goal
        self.weight_format = weight_format
        self.weight_tone = weight_tone
        self.weight_grammar = weight_grammar

        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate composite loss

        Args:
            predictions: Output from Abstractor.forward()
            targets: Dictionary with:
                - 'v_goal': [B] (class indices)
                - 'v_tone': [B, 1]
                - 'v_format': [B] (class indices)
                - 'v_grammar': [B, 4]

        Returns:
            total_loss, loss_dict
        """
        # Classification losses (CrossEntropy)
        loss_goal = self.ce_loss(predictions['v_goal'], targets['v_goal'].long())
        loss_format = self.ce_loss(predictions['v_format'], targets['v_format'].long())

        # Regression losses (MSE)
        loss_tone = self.mse_loss(predictions['v_tone'], targets['v_tone'])
        loss_grammar = self.mse_loss(predictions['v_grammar'], targets['v_grammar'])

        # Total weighted loss
        total_loss = (
            self.weight_goal * loss_goal +
            self.weight_format * loss_format +
            self.weight_tone * loss_tone +
            self.weight_grammar * loss_grammar
        )

        # Return loss breakdown
        loss_dict = {
            'total': total_loss.item(),
            'goal': loss_goal.item(),
            'format': loss_format.item(),
            'tone': loss_tone.item(),
            'grammar': loss_grammar.item()
        }

        return total_loss, loss_dict


def calculate_metrics(predictions: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Calculate evaluation metrics

    Returns:
        Dictionary with:
        - goal_accuracy: Classification accuracy for goal
        - format_accuracy: Classification accuracy for format
        - tone_mae: Mean absolute error for tone
        - grammar_mae: Mean absolute error for grammar (per dimension)
        - grammar_mae_total: Overall MAE for all grammar dimensions
    """
    with torch.no_grad():
        # Goal accuracy
        goal_pred = predictions['v_goal'].argmax(dim=1)
        goal_acc = (goal_pred == targets['v_goal'].long()).float().mean().item()

        # Format accuracy
        format_pred = predictions['v_format'].argmax(dim=1)
        format_acc = (format_pred == targets['v_format'].long()).float().mean().item()

        # Tone MAE
        tone_mae = F.l1_loss(predictions['v_tone'], targets['v_tone']).item()

        # Grammar MAE (per dimension)
        grammar_mae_per_dim = F.l1_loss(
            predictions['v_grammar'],
            targets['v_grammar'],
            reduction='none'
        ).mean(dim=0).cpu().numpy()

        grammar_mae_total = grammar_mae_per_dim.mean()

        metrics = {
            'goal_accuracy': goal_acc,
            'format_accuracy': format_acc,
            'tone_mae': tone_mae,
            'grammar_mae_alignment': float(grammar_mae_per_dim[0]),
            'grammar_mae_contrast': float(grammar_mae_per_dim[1]),
            'grammar_mae_whitespace': float(grammar_mae_per_dim[2]),
            'grammar_mae_hierarchy': float(grammar_mae_per_dim[3]),
            'grammar_mae_total': float(grammar_mae_total)
        }

        return metrics


if __name__ == '__main__':
    """Test the Abstractor model"""
    print("=" * 60)
    print("Testing Abstractor Model")
    print("=" * 60)

    # Create model
    model = Abstractor(n_goal_classes=4, n_format_classes=3, pretrained=True)
    print(f"\nModel created")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Test forward pass
    batch_size = 8
    f_tensor = torch.randn(batch_size, 4, 256, 256)

    print(f"\nInput shape: {f_tensor.shape}")

    outputs = model(f_tensor)

    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Test loss
    print(f"\nTesting loss function...")
    criterion = AbstractorLoss()

    targets = {
        'v_goal': torch.randint(0, 4, (batch_size,)),
        'v_tone': torch.rand(batch_size, 1),
        'v_format': torch.randint(0, 3, (batch_size,)),
        'v_grammar': torch.rand(batch_size, 4)
    }

    loss, loss_dict = criterion(outputs, targets)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # Test metrics
    print(f"\nTesting metrics...")
    metrics = calculate_metrics(outputs, targets)
    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("Abstractor model test passed!")
    print("=" * 60)
