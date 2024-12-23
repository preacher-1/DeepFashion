import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import T_co
from torchvision import models
from resnet_module import ResNetModule, resnet34
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence

"""
网格表示+Transformer 编码器+解码器
使用ResNet18作为特征提取器，对原始图像进行特征提取，得到多尺度的特征图
对较大的特征图进行下采样后，将所有特征图进行网格划分，拼接后经过特征融合后送入Transformer编码器，得到全局特征表示（或不进行特征融合，直接经过位置编码输入Transformer编码器）
Transformer编码器输出的全局特征表示送入解码器，从<start>开始解码，生成图像描述文本。
"""


# 使用ResNet50作为特征提取器
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(512, 512)

    def forward(self, x):
        x = self.resnet(x)
        return x


# Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_encoder_layers)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        return output


# Transformer解码器
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerDecoder, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_decoder_layers)

    def forward(self, tgt, memory):
        tgt = tgt.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        output = self.transformer_decoder(tgt, memory)
        output = output.permute(1, 0, 2)
        return output


# 网格划分模块
class GridModule(nn.Module):
    def __init__(self, grid_size, feature_size):
        super(GridModule, self).__init__()
        self.grid_size = grid_size
        self.feature_size = feature_size
        self.grid_embedding = nn.Embedding(grid_size ** 2, feature_size)

    def forward(self, x):
        batch_size = x.size(0)
        feature_map_size = x.size(2)
        grid_size = self.grid_size
        feature_size = self.feature_size
        x = x.view(batch_size, feature_size, feature_map_size ** 2)
        x = x.permute(0, 2, 1)
        x = F.interpolate(x, size=grid_size, mode='bilinear', align_corners=True)
        x = x.permute(0, 2, 1)
        x = x.view(batch_size, grid_size ** 2, feature_size)
        x = self.grid_embedding(torch.arange(0, grid_size ** 2).long().to(x.device))
        x = x.view(1, grid_size ** 2, feature_size)
        x = x.repeat(batch_size, 1, 1)
        return x


# 特征融合模块
class FeatureFusionModule(nn.Module):
    def __init__(self, feature_size):
        super(FeatureFusionModule, self).__init__()
        self.feature_size = feature_size
        self.linear = nn.Linear(feature_size * 2, feature_size)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.linear(x)
        return x


# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# 图像描述生成模型
class ImageCaptionModel(nn.Module):
    def __init__(self, feature_size, grid_size, d_model, nhead, num_encoder_layers, dim_feedforward, dropout,
                 num_decoder_layers):
        super(ImageCaptionModel, self).__init__()
        self.feature_size = feature_size
        self.grid_size = grid_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.num_decoder_layers = num_decoder_layers

        self.resnet_module = ResNetModule()
        self.grid_module = GridModule(grid_size, feature_size)
        self.feature_fusion_module = FeatureFusionModule(feature_size)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.transformer_encoder = TransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.transformer_decoder = TransformerDecoder(d_model, nhead, num_decoder_layers, dim_feedforward, dropout)
        self.linear = nn.Linear(d_model, feature_size)
        self.linear2 = nn.Linear(feature_size, 512)
        self.linear3 = nn.Linear(512, 1)

    def forward(self, x, tgt, src_mask, tgt_mask):
        # 特征提取
        feature_map = self.resnet_module(x)
        # 网格划分
        grid_embedding = self.grid_module(feature_map)
        # 特征融合
        feature_fusion = self.feature_fusion_module(grid_embedding, feature_map)
        # 位置编码
        src = self.positional_encoding(feature_fusion)
        # Transformer编码器
        memory = self.transformer_encoder(src, src_mask)
        # Transformer解码器
        output = self.transformer_decoder(tgt, memory, tgt_mask)
        # 输出层
        output = self.linear(output)
        output = self.linear2(output)
        output = self.linear3(output)
        return output

    def generate_caption(self, x, max_len=20):
        # 特征提取
        feature_map = self.resnet_module(x)
        # 网格划分
        grid_embedding = self.grid_module(feature_map)
        # 特征融合
        feature_fusion = self.feature_fusion_module(grid_embedding, feature_map)
        # 位置编码
        src = self.positional_encoding(feature_fusion)
        # Transformer编码器
        memory = self.transformer_encoder(src)
        # 解码器初始化
        tgt = torch.zeros(1, 1).long().to(x.device)
        tgt_mask = self.transformer_decoder.generate_square_subsequent_mask(1).to(x.device)
        # 解码器
        for i in range(max_len):
            output = self.transformer_decoder(tgt, memory, tgt_mask)
            output = self.linear(output)
            output = self.linear2(output)
            output = self.linear3(output)
            output = F.softmax(output, dim=1)
            _, predicted_index = torch.max(output, dim=1)
            tgt = torch.cat((tgt, predicted_index.unsqueeze(0)), dim=1)
            if predicted_index == 1:
                break
        return tgt.squeeze(0)


# # 训练模型
# def train_model(model, train_loader, optimizer, criterion, device):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data, target[:-1], src_mask=None, tgt_mask=None)
#         loss = criterion(output.reshape(-1, 1), target[1:].reshape(-1))
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))
#
#
# # 验证模型
# def validate_model(model, val_loader, criterion, device):
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for data, target in val_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data, target[:-1], src_mask=None, tgt_mask=None)
#             val_loss += criterion(output.reshape(-1, 1), target[1:].reshape(-1)).item()
#     val_loss /= len(val_loader)
#     print('\nValidation set: Average loss: {:.4f}\n'.format(val_loss))
#     return val_loss
#
#
# # 训练模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = ImageCaptionModel(feature_size=512, grid_size=16, d_model=512, nhead=8, num_encoder_layers=6,
#                           dim_feedforward=2048, dropout=0.1, num_decoder_layers=6).to(device)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# for epoch in range(1, 10):
#     train_model(model, train_loader, optimizer, criterion, device)
#     val_loss = validate_model(model, val_loader, criterion, device)

# 模型实例化

# # 损失函数
# class PackedCrossEntropyLoss(nn.Module):
#     def __init__(self):
#         super(PackedCrossEntropyLoss, self).__init__()
#         self.loss_fn = nn.CrossEntropyLoss()
#
#     def forward(self, predictions, targets, lengths):
#         packed_predictions = pack_padded_sequence(predictions, lengths, batch_first=True, enforce_sorted=False)[0]
#         packed_targets = pack_padded_sequence(targets, lengths, batch_first=True, enforce_sorted=False)[0]
#
#         # 计算损失，忽略填充的部分
#         loss = self.loss_fn(packed_predictions, packed_targets)
#         return loss

# class ImageCaptionDataset(Dataset):
#     def __init__(self,root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#
#         self.
#
#     def __getitem__(self, index) -> T_co:
#         pass
#
#
# train_dataset = ImageCaptionDataset()
# x_train = DataLoader()
