import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter


class CaptioningTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(self.device)
        self.model.global_step = 0

    def train_epoch(self, dataloader, writer: SummaryWriter, epoch):
        self.model.train()
        total_loss, steps_loss = 0, []

        for i, batch in enumerate(dataloader):
            images, captions = batch
            images = images.to(self.device)
            captions = captions.to(self.device)

            # 前向传播
            outputs = self.model(images, captions)  # [B, T, vocab_size]
            
            # 准备目标序列
            targets = captions[:, 1:]  # 去除[START]标记
            
            # 计算损失
            loss = self.criterion(
                outputs.reshape(-1, outputs.size(-1)),  # [B*T, vocab_size]
                targets.reshape(-1)  # [B*T]
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
            self.optimizer.step()

            steps_loss.append(loss.item())
            total_loss += loss.item()

            # 记录日志
            writer.add_scalar('train_step_loss', loss.item(), epoch * len(dataloader) + i)

        # 记录日志
        writer.add_scalar('train_epoch_loss', total_loss / len(dataloader), epoch + 1)
        return total_loss / len(dataloader), steps_loss

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                images, captions = batch
                images = images.to(self.device)
                captions = captions.to(self.device)

                # 前向传播
                outputs = self.model(images, captions)  # [B, T, vocab_size]
                
                # 准备目标序列
                targets = captions[:, 1:]  # 去除[START]标记
                
                # 计算损失 - 使用reshape代替view
                loss = self.criterion(
                    outputs.reshape(-1, outputs.size(-1)),  # [B*T, vocab_size]
                    targets.reshape(-1)  # [B*T]
                )
                
                total_loss += loss.item()

        return total_loss / len(dataloader)
