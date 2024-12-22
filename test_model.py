import torch
import torch.nn as nn
import torchvision


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # CNN Encoder (使用预训练模型)
        resnet = torchvision.models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # 移除最后的全连接层

        # 特征映射层
        self.feature_projection = nn.Linear(2048, hidden_dim)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

        # 输出层
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, images, captions):
        """

        :param images: [B, 3, 224, 224]
        :param captions: [B, 60]
        :return:
        """
        # CNN特征提取
        batch_size = images.size(0)
        cnn_features = self.cnn(images)  # [B, 2048, H', W']
        cnn_features = cnn_features.view(batch_size, 2048, -1).permute(2, 0, 1)  # [H'W', B, 2048]

        # 特征映射
        encoder_features = self.feature_projection(cnn_features)  # [H'W', B, hidden_dim]

        # 词嵌入
        caption_embeddings = self.embedding(captions).permute(1, 0, 2)  # [seq_len, B, hidden_dim]

        # Transformer
        output = self.transformer(encoder_features, caption_embeddings)

        # 输出层
        output = self.fc_out(output)  # [seq_len, B, vocab_size]

        return output

