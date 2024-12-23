import torch
import torch.nn as nn
import torchvision
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class ImageCaptioningModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_dim=512,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
    ):
        super().__init__()

        # CNN Encoder (使用预训练的ResNet)
        resnet = torchvision.models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # 冻结CNN参数
        for param in self.cnn.parameters():
            param.requires_grad = False

        # 特征映射层
        self.feature_projection = nn.Linear(2048, hidden_dim)

        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )

        # 输出层
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def generate_square_subsequent_mask(self, sz):
        """生成前瞻掩码"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
                .masked_fill(mask == 0, float("-inf"))
                .masked_fill(mask == 1, float(0.0))
        )
        return mask.bool()

    def create_padding_mask(self, seq):
        """生成填充掩码"""
        return seq == 0

    def forward(self, images, captions):
        """
        :param images: [B, 3, 224, 224]
        :param captions: [B, seq_len]
        :return:
        """
        # CNN特征提取
        batch_size = images.size(0)
        cnn_features = self.cnn(images)  # [B, 2048, 7, 7]
        cnn_features = cnn_features.view(batch_size, 2048, -1).permute(
            2, 0, 1
        )  # [49, B, 2048]

        # 特征映射和位置编码
        encoder_features = self.feature_projection(cnn_features)  # [49, B, 512]
        encoder_features = self.pos_encoder(encoder_features)

        # 准备解码器输入
        tgt = captions[:, :-1]  # [B, T] 去除[END]标记
        tgt_embeddings = self.embedding(tgt)  # [B, T, 512]
        tgt_embeddings = tgt_embeddings.permute(1, 0, 2)  # [T, B, 512]
        tgt_embeddings = self.pos_encoder(tgt_embeddings)

        # 创建掩码
        tgt_len = tgt.size(1)
        # 自注意力掩码 [T, T]
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(images.device)
        # 填充掩码 [B, T]
        tgt_padding_mask = self.create_padding_mask(tgt).to(images.device)

        # Transformer前向传播
        output = self.transformer(
            src=encoder_features,  # [49, B, 512]
            tgt=tgt_embeddings,  # [T, B, 512]
            tgt_mask=tgt_mask,  # [T, T]
            tgt_key_padding_mask=tgt_padding_mask,  # [B, T]
        )

        # 输出层
        output = self.fc_out(output)  # [T, B, vocab_size]

        return output.permute(1, 0, 2)

    @torch.no_grad()
    def generate(
            self, image, max_len=50, temperature=1.0, vocab=None
    ) -> list | torch.Tensor:
        """
        生成图像描述
        Args:
            image (torch.Tensor): 输入图像 [1, 3, 224, 224]
            max_len (int): 生成序列的最大长度
            temperature (float): 采样温度
            vocab (dict): 词表字典，用于将ID转换为词
        Returns:
            list: 生成的词序列
            torch.Tensor: 生成的token ID序列
        """
        self.eval()
        device = next(self.parameters()).device

        # 确保图像维度正确
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(device)

        # CNN特征提取
        batch_size = image.size(0)
        cnn_features = self.cnn(image)
        cnn_features = cnn_features.view(batch_size, 2048, -1).permute(2, 0, 1)

        # 特征映射并添加位置编码
        encoder_features = self.feature_projection(cnn_features)
        encoder_features = self.pos_encoder(encoder_features)

        # 初始化解码序列
        current_token = torch.tensor([[1]]).to(device)  # [START] token
        output_tokens = [1]  # 存储生成的token ID

        # 逐词生成
        for i in range(max_len):
            # 准备decoder输入
            tgt = self.embedding(current_token).permute(1, 0, 2)
            tgt = self.pos_encoder(tgt)

            # 创建掩码
            tgt_mask = self.generate_square_subsequent_mask(current_token.size(1)).to(
                device
            )
            tgt_padding_mask = self.create_padding_mask(current_token).to(device)

            # Transformer解码
            output = self.transformer(
                encoder_features,
                tgt,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_padding_mask,
            )

            # 获取下一个词的预测
            logits = self.fc_out(output[-1, :])

            # 应用温度
            if temperature != 1.0:
                logits = logits / temperature

            # 采样下一个词
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)

            # 添加到输出序列
            output_tokens.append(next_token.item())
            current_token = torch.cat([current_token, next_token], dim=1)

            # 如果生成了[END]标记，停止生成
            if next_token.item() == 2:  # [END] token
                break

        # 如果提供了词表，转换为词序列
        if vocab is not None:
            id2word = {v: k for k, v in vocab.items()}
            words = [id2word.get(idx, "[UNK]") for idx in output_tokens]
            # 移除[START]和[END]标记
            words = words[1:]
            if words[-1] == "[END]":
                words = words[:-1]
            return words

        return output_tokens
