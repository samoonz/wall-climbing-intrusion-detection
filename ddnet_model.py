import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    def __init__(self):
        self.frame_l = 40      # Số lượng frame
        self.joint_n = 17      # Số khớp
        self.joint_d = 2       # Chiều không gian (x, y)
        self.cls_num = 2       # Số lớp phân loại
        self.feat_d = 136      # Đặc trưng JCD  
        self.filters = 16      # Filters khởi đầu

# --- Layers ---
class SpatialDropout1D(nn.Dropout1d):
    def forward(self, x):
        # Input: (B, T, C)
        x = x.permute(0, 2, 1)  # (B, C, T)
        x = super().forward(x)
        return x.permute(0, 2, 1)  # (B, T, C)

class C1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding="same"):
        super(C1D, self).__init__()
        if padding == "same":
            pad = kernel_size // 2
        else:
            pad = 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        # Input: (B, T, C) → (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x.permute(0, 2, 1)  # Output: (B, T, C)
    
class Block(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super(Block, self).__init__()
        self.c1 = C1D(in_channels, filters, kernel_size)
        self.c2 = C1D(filters, filters, kernel_size)

    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        return x

class D1D(nn.Module):
    def __init__(self, in_features, out_features):
        super(D1D, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.linear(x)
        x = self.bn(x)
        x = self.act(x)
        return x

def poses_motion(P):
    # P shape: (B, F, J, D)
    diff_slow = P[:, 4:, :, :] - P[:, :-4, :, :]
    diff_fast = P[:, 1:, :, :] - P[:, :-1, :, :]

    # Padding để khớp lại độ dài ban đầu
    pad_slow = torch.zeros_like(P[:, :4, :, :])
    pad_fast = torch.zeros_like(P[:, :1, :, :])
    diff_slow = torch.cat([pad_slow, diff_slow], dim=1)
    diff_fast = torch.cat([pad_fast, diff_fast], dim=1)

    B, F, J, D = P.shape
    diff_slow = diff_slow.reshape(B, F, J * D)
    diff_fast = diff_fast.reshape(B, F, J * D)

    return diff_slow, diff_fast


class DDNetOriginal(nn.Module):
    def __init__(self, config, use_attention=True):
        super(DDNetOriginal, self).__init__()
        self.use_attention = use_attention
        frame_l = config.frame_l
        feat_d = config.feat_d
        joint_n = config.joint_n
        joint_d = config.joint_d
        class_num = config.cls_num
        filters = config.filters

        # --- JCD branch ---
        self.jcd_conv1 = nn.Sequential(C1D(feat_d, 2*filters, 1), SpatialDropout1D(0.1))
        self.jcd_conv2 = nn.Sequential(C1D(2*filters, filters, 3), SpatialDropout1D(0.1))
        self.jcd_conv3 = C1D(filters, filters, 1)
        self.jcd_pool = nn.Sequential(nn.MaxPool1d(2), SpatialDropout1D(0.1))

        # --- Slow branch ---
        self.slow_conv1 = nn.Sequential(C1D(joint_n*joint_d, 2*filters, 1), SpatialDropout1D(0.1))
        self.slow_conv2 = nn.Sequential(C1D(2*filters, filters, 3), SpatialDropout1D(0.1))
        self.slow_conv3 = C1D(filters, filters, 1)
        self.slow_pool = nn.Sequential(nn.MaxPool1d(2), SpatialDropout1D(0.1))

        # --- Fast branch ---
        self.fast_conv1 = nn.Sequential(C1D(joint_n*joint_d, 2*filters, 1), SpatialDropout1D(0.1))
        self.fast_conv2 = nn.Sequential(C1D(2*filters, filters, 3), SpatialDropout1D(0.1))
        self.fast_conv3 = nn.Sequential(C1D(filters, filters, 1), SpatialDropout1D(0.1))

        # --- Attention ---
        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=3*filters, num_heads=1, batch_first=True)

        # --- Blocks ---
        self.block1 = Block(3*filters, 2*filters, 3)
        self.block_pool1 = nn.Sequential(nn.AdaptiveMaxPool1d(frame_l//2), SpatialDropout1D(0.1))
        self.block2 = Block(2*filters, 4*filters, 3)
        self.block_pool2 = nn.Sequential(nn.AdaptiveMaxPool1d(frame_l//4), SpatialDropout1D(0.1))
        self.block3 = nn.Sequential(Block(4*filters, 8*filters, 3), SpatialDropout1D(0.1))

        # --- Fully connected ---
        self.linear1 = nn.Sequential(D1D(8*filters, 128), nn.Dropout(0.5))
        self.linear2 = nn.Sequential(D1D(128, 128), nn.Dropout(0.5))
        self.linear3 = nn.Linear(128, class_num)

    def forward(self, M, P):
        # --- JCD branch ---
        x = self.jcd_conv1(M)
        x = self.jcd_conv2(x)
        x = self.jcd_conv3(x)
        x = x.permute(0,2,1)
        x = self.jcd_pool(x)
        x = x.permute(0,2,1)

        # --- Pose motion branch ---
        diff_slow, diff_fast = poses_motion(P)

        # Slow branch
        x_slow = self.slow_conv1(diff_slow)
        x_slow = self.slow_conv2(x_slow)
        x_slow = self.slow_conv3(x_slow)
        x_slow = x_slow.permute(0,2,1)
        x_slow = self.slow_pool(x_slow)
        x_slow = x_slow.permute(0,2,1)

        # Fast branch
        x_fast = self.fast_conv1(diff_fast)
        x_fast = self.fast_conv2(x_fast)
        x_fast = self.fast_conv3(x_fast)

        # --- Adaptive pooling để concat không lỗi ---
        target_len = min(x.shape[1], x_slow.shape[1], x_fast.shape[1])

        x = F.adaptive_max_pool1d(x.permute(0,2,1), target_len).permute(0,2,1)
        x_slow = F.adaptive_max_pool1d(x_slow.permute(0,2,1), target_len).permute(0,2,1)
        x_fast = F.adaptive_max_pool1d(x_fast.permute(0,2,1), target_len).permute(0,2,1)

        x = torch.cat([x, x_slow, x_fast], dim=2)

        # --- Attention ---
        if self.use_attention:
            x,_ = self.attention(x,x,x)

        # --- Blocks ---
        x = self.block1(x)
        x = x.permute(0,2,1)
        x = self.block_pool1(x)
        x = x.permute(0,2,1)

        x = self.block2(x)
        x = x.permute(0,2,1)
        x = self.block_pool2(x)
        x = x.permute(0,2,1)

        x = self.block3(x)

        # --- Global max pooling ---
        x = torch.max(x, dim=1).values

        # --- Fully connected ---
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x