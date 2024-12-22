class CaptioningTrainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, steps_loss = 0, []

        for batch in dataloader:
            images, captions = batch
            images = images.to(self.device)
            captions = captions.to(self.device)

            # 教师强制训练
            outputs = self.model(images, captions[:, :-1])  # 去除[END]标记

            # 计算损失
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                captions[:, 1:].contiguous().view(-1)  # 去除[START]标记
            )

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            steps_loss.append(loss.item())
            total_loss += loss.item()

        return total_loss / len(dataloader), steps_loss
