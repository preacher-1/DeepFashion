import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize


# nltk.download('punkt_tab')


class ImageCaptioningDataset(Dataset):
    def __init__(self, json_path, image_dir, transform=None, max_length=60, vocab=None, is_training=True):
        """
        Args:
            json_path (str): 标注文件路径
            image_dir (str): 图片文件夹路径
            transform: 图像预处理变换
            max_length (int): 句子最大长度
            vocab (Vocabulary): 词表对象，训练时为None会自动构建
            is_training (bool): 是否为训练模式
        """
        self.image_dir = image_dir
        self.max_length = max_length
        self.is_training = is_training

        # 读取标注文件
        with open(json_path, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)

        self.image_paths = list(self.annotations.keys())

        # 设置图像预处理
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        # 构建或加载词表
        self.vocab = vocab
        if is_training and vocab is None:
            self.vocab = self.build_vocabulary()

    def build_vocabulary(self, min_freq=5):
        """构建词表"""
        word_freq = Counter()

        # 统计词频
        for caption in self.annotations.values():
            tokens = word_tokenize(caption.lower())
            word_freq.update(tokens)

        # 创建词表
        vocab = {'[PAD]': 0, '[START]': 1, '[END]': 2, '[UNK]': 3}
        idx = 4

        for word, freq in word_freq.items():
            if freq >= min_freq:
                vocab[word] = idx
                idx += 1

        return vocab

    def tokenize_caption(self, caption):
        """将描述文本转换为token序列"""
        tokens = word_tokenize(caption.lower())
        tokens = ['[START]'] + tokens + ['[END]']

        # 截断或填充到指定长度
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend(['[PAD]'] * (self.max_length - len(tokens)))

        # 将tokens转换为索引
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab['[UNK]'])

        return torch.LongTensor(token_ids)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 获取图片路径和描述
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, image_name)
        caption = self.annotations[image_name]

        # 读取并预处理图片
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # 处理描述文本
        caption_tensor = self.tokenize_caption(caption)

        return image, caption_tensor


class PreserveAspectRatioTransform:
    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, img):
        # 计算缩放比例
        w, h = img.size
        ratio = min(self.target_size / w, self.target_size / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)

        transform = transforms.Compose([
            transforms.Resize((new_h, new_w)),
            transforms.Pad(
                padding=(
                    (self.target_size - new_w) // 2,
                    (self.target_size - new_h) // 2,
                    (self.target_size - new_w + 1) // 2,
                    (self.target_size - new_h + 1) // 2
                ),
                fill=0
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(img)


def get_dataloaders(train_json_path, test_json_path, image_dir, batch_size=32, num_workers=4, transform=None):
    """创建数据加载器"""
    # 创建训练集
    train_dataset = ImageCaptioningDataset(
        json_path=train_json_path,
        image_dir=image_dir,
        transform=transform,
        is_training=True
    )

    # 创建测试集，使用训练集的词表
    test_dataset = ImageCaptioningDataset(
        json_path=test_json_path,
        image_dir=image_dir,
        transform=transform,
        vocab=train_dataset.vocab,
        is_training=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset.vocab


# 使用示例
def main():
    # 配置参数
    config = {
        'train_json': 'data/deepfashion-multimodal/train_captions.json',
        'test_json': 'data/deepfashion-multimodal/test_captions.json',
        'image_dir': 'data/deepfashion-multimodal/images',
        'batch_size': 32,
        'num_workers': 4
    }

    # 创建数据加载器
    train_loader, test_loader, vocab = get_dataloaders(
        train_json_path=config['train_json'],
        test_json_path=config['test_json'],
        image_dir=config['image_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # 保存词表（可选）
    vocab_path = 'data/deepfashion-multimodal/vocab.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f'词表大小: {len(vocab)}')
    print(f'训练集大小: {len(train_loader.dataset)}')
    print(f'测试集大小: {len(test_loader.dataset)}')

    # 测试数据加载
    for images, captions in train_loader:
        print(f'图像批次形状: {images.shape}')
        print(f'描述批次形状: {captions.shape}')
        break


if __name__ == '__main__':
    main()

# 词表大小: 109
# 训练集大小: 10155
# 测试集大小: 2538
# 图像批次形状: torch.Size([32, 3, 224, 224])
# 描述批次形状: torch.Size([32, 60])
