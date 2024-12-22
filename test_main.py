import torch
import torch.nn as nn
from datasets import get_dataloaders
from test_model import ImageCaptioningModel
from test_trainer import CaptioningTrainer
from utils import generate_caption, decode_sequence
# from torch.utils.tensorboard import SummaryWriter


# writer = SummaryWriter('runs/fashion_mnist_experiment_1')

# 配置参数
config = {
    'image_dir': 'data/deepfashion-multimodal/images',
    'train_json_path': 'data/deepfashion-multimodal/train_captions.json',
    'test_json_path': 'data/deepfashion-multimodal/test_captions.json',
    'vocab_file': 'data/deepfashion-multimodal/vocab.json',
    'batch_size': 32,
    'hidden_dim': 512,
    'nhead': 8,
    'num_encoder_layers': 6,
    'num_decoder_layers': 6,
    'learning_rate': 1e-4,
    'epochs': 50,
    'clip_grad': 5.0
}


# 训练流程
def main():
    # 1. 数据加载
    train_loader, val_loader = get_dataloaders(train_json_path=config['train_json_path'],
                                               test_json_path=config['test_json_path'],
                                               image_dir=config['image_dir'],
                                               batch_size=config['batch_size'])
    vocab = train_loader.dataset.vocab
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 模型初始化
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        hidden_dim=config['hidden_dim'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers']
    ).to(device)

    # 3. 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['[PAD]'])

    # 4. 训练器
    trainer = CaptioningTrainer(model, optimizer, criterion, device)

    # 5. 训练循环，同时记录损失，并实现早停
    loss_history, steps_loss_history = [], []
    for epoch in range(config['epochs']):
        loss, steps_loss = trainer.train_epoch(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

        # # 验证
        # if (epoch + 1) % 5 == 0:
        #     validate(model, val_loader)

        # 记录损失
        loss_history.append(loss)
        steps_loss_history.update(steps_loss)

        # 保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'.checkpoints/model_{epoch + 1}.pth')
            print(f'Model saved to .checkpoints/model_{epoch + 1}.pth')

        # 早停
        if len(loss_history) > 1 and loss_history[-1] > loss_history[-2]:
            print('Early stopping')
            break

    # 6. 预测
    # model.load_state_dict(torch.load('.checkpoints/model_30.pth'))
    # model.eval()



if __name__ == '__main__':
    main()
