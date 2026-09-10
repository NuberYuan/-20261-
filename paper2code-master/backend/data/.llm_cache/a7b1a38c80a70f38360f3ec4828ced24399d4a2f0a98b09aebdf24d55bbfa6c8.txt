import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F


_DEFAULT_CONFIG_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'config.yaml')
)


def _resolve_config(config):
    """Return a dict from config.yaml, a provided dict, or an empty dict."""
    if isinstance(config, dict):
        return config

    if isinstance(config, str):
        if os.path.isfile(config):
            with open(config, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    if os.path.isfile(_DEFAULT_CONFIG_PATH):
        with open(_DEFAULT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    return {}


class ConvBlock(nn.Module):
    """Convolutional block: Conv2d -> BN -> ReLU -> [Dropout] -> Conv2d -> BN -> ReLU."""

    def __init__(self, in_channels, out_channels, dropout_ratio=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0.0 else None

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.relu2(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    """UNet++ (nested U-Net) architecture.

    This implementation follows the UNet++ paper's nested dense skip connections.
    The class is named ``UNet`` to match the project's file/module layout, while
    the architecture itself is the nested UNet++ structure.
    """

    def __init__(self, config=None, in_channels=3, deep_supervision=False):
        super().__init__()

        cfg = _resolve_config(config)
        model_cfg = cfg.get('model', {}) if isinstance(cfg, dict) else {}

        # Binary segmentation default is 1 foreground channel, as described in the paper.
        self.num_classes = int(model_cfg.get('num_classes') or 1)
        if self.num_classes < 1:
            self.num_classes = 1

        dropout_cfg = model_cfg.get('dropout_ratio')
        try:
            dropout_ratio = float(dropout_cfg)
            if not (0.0 <= dropout_ratio < 1.0):
                dropout_ratio = 0.0
        except (TypeError, ValueError):
            dropout_ratio = 0.0
        self.dropout_ratio = dropout_ratio

        self.in_channels = int(in_channels)
        self.deep_supervision = bool(deep_supervision)

        # Base feature widths used by the nested decoder.  Starting at 32 keeps
        # the parameter count in the same order of magnitude as the UNet++
        # reference implementation for medical image segmentation.
        self.features = [32, 64, 128, 256, 512]
        f = self.features

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path (j = 0)
        self.conv00 = ConvBlock(self.in_channels, f[0], self.dropout_ratio)
        self.conv10 = ConvBlock(f[0], f[1], self.dropout_ratio)
        self.conv20 = ConvBlock(f[1], f[2], self.dropout_ratio)
        self.conv30 = ConvBlock(f[2], f[3], self.dropout_ratio)
        self.conv40 = ConvBlock(f[3], f[4], self.dropout_ratio)

        # Decoder path (j = 1)
        self.conv01 = ConvBlock(f[0] + f[1], f[0], self.dropout_ratio)
        self.conv11 = ConvBlock(f[1] + f[2], f[1], self.dropout_ratio)
        self.conv21 = ConvBlock(f[2] + f[3], f[2], self.dropout_ratio)
        self.conv31 = ConvBlock(f[3] + f[4], f[3], self.dropout_ratio)

        # Decoder path (j = 2)
        self.conv02 = ConvBlock(f[0] * 2 + f[1], f[0], self.dropout_ratio)
        self.conv12 = ConvBlock(f[1] * 2 + f[2], f[1], self.dropout_ratio)
        self.conv22 = ConvBlock(f[2] * 2 + f[3], f[2], self.dropout_ratio)

        # Decoder path (j = 3)
        self.conv03 = ConvBlock(f[0] * 3 + f[1], f[0], self.dropout_ratio)
        self.conv13 = ConvBlock(f[1] * 3 + f[2], f[1], self.dropout_ratio)

        # Decoder path (j = 4)
        self.conv04 = ConvBlock(f[0] * 4 + f[1], f[0], self.dropout_ratio)

        # Deep supervision heads, one for each decoder node at resolution level i=0.
        self.out1 = nn.Conv2d(f[0], self.num_classes, kernel_size=1)
        self.out2 = nn.Conv2d(f[0], self.num_classes, kernel_size=1)
        self.out3 = nn.Conv2d(f[0], self.num_classes, kernel_size=1)
        self.out4 = nn.Conv2d(f[0], self.num_classes, kernel_size=1)

    @staticmethod
    def _upsample(x, target_size):
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)

    def _activate(self, x):
        if self.num_classes == 1:
            return torch.sigmoid(x)
        return torch.softmax(x, dim=1)

    def forward(self, x, deep_supervision=None):
        if deep_supervision is None:
            deep_supervision = self.deep_supervision

        # Encoder pathway
        x00 = self.conv00(x)
        x10 = self.conv10(self.maxpool(x00))
        x20 = self.conv20(self.maxpool(x10))
        x30 = self.conv30(self.maxpool(x20))
        x40 = self.conv40(self.maxpool(x30))

        # Nested decoder pathway, level 1
        x01 = self.conv01(torch.cat([x00, self._upsample(x10, x00.shape[2:])], dim=1))
        x11 = self.conv11(torch.cat([x10, self._upsample(x20, x10.shape[2:])], dim=1))
        x21 = self.conv21(torch.cat([x20, self._upsample(x30, x20.shape[2:])], dim=1))
        x31 = self.conv31(torch.cat([x30, self._upsample(x40, x30.shape[2:])], dim=1))

        # Nested decoder pathway, level 2
        x02 = self.conv02(torch.cat([x00, x01, self._upsample(x11, x00.shape[2:])], dim=1))
        x12 = self.conv12(torch.cat([x10, x11, self._upsample(x21, x10.shape[2:])], dim=1))
        x22 = self.conv22(torch.cat([x20, x21, self._upsample(x31, x20.shape[2:])], dim=1))

        # Nested decoder pathway, level 3
        x03 = self.conv03(torch.cat([x00, x01, x02, self._upsample(x12, x00.shape[2:])], dim=1))
        x13 = self.conv13(torch.cat([x10, x11, x12, self._upsample(x22, x10.shape[2:])], dim=1))

        # Nested decoder pathway, level 4
        x04 = self.conv04(torch.cat([x00, x01, x02, x03, self._upsample(x13, x00.shape[2:])], dim=1))

        out1 = self._activate(self.out1(x01))
        out2 = self._activate(self.out2(x02))
        out3 = self._activate(self.out3(x03))
        out4 = self._activate(self.out4(x04))

        if deep_supervision:
            return [out1, out2, out3, out4]
        return out4