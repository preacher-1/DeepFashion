import os
import json
from collections import Counter
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ImageDescriptionDataset(Dataset):
    def __init__(self, json_path, img_folder, transform=None):
        """
        Args:
            json_path (str): Path to the JSON file containing filename-description mapping.
            img_folder (str): Path to the folder containing images.
            transform (callable, optional): Transform to apply to the images.
        """
        self.word_to_idx = {}
        self.vocab = Counter()
        self.img_folder = img_folder
        self.transform = transform

        # Load filename-description mapping
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Get a list of filenames
        self.filenames = list(data.keys())
        self.captions = list(data.values())
        self.caption_lengths = []
        self.length = len(self.filenames)

        self.textdata_transform()
        self.vocal_size = len(self.word_to_idx)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get the filename and description
        filename = self.filenames[idx]
        caption = self.captions[idx]
        caption_length = self.caption_lengths[idx]

        # Load the image
        img_path = os.path.join(self.img_folder, filename)
        image = Image.open(img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, caption, caption_length

    def textdata_transform(self, max_length=128):
        for description in self.captions:
            self.vocab.update(description.lower().split())

        # 移除低频词
        # self.vocab = {k: v for k, v in self.vocab.items() if v >= 5}

        # 构建词典
        self.word_to_idx = {word: idx + 4 for idx, word in enumerate(self.vocab)}
        self.word_to_idx['<pad>'] = 0
        self.word_to_idx['<start>'] = 1
        self.word_to_idx['<end>'] = 2
        self.word_to_idx['<unk>'] = 3

        # 将文本描述逐词替换为词索引
        for idx, caption in enumerate(self.captions):
            words = caption.lower().split()
            encoded_caption = [self.word_to_idx.get(word, self.word_to_idx['<unk>']) for word in words]
            # 加2是因为要加上<start>和<end>，但最终caplen应该减去1
            # caplen = min(len(encoded_caption) + 2, max_length) - 1
            encoded_caption = [self.word_to_idx['<start>']] + encoded_caption + [self.word_to_idx['<end>']]
            encoded_caption += [self.word_to_idx['<pad>']] * (max_length - len(encoded_caption))
            self.captions[idx] = encoded_caption
            self.caption_lengths.append(len(encoded_caption) - 1)


def build_dataloader(json_path, img_folder, transform, batch_size, shuffle=True, num_workers=4):
    dataset = ImageDescriptionDataset(json_path, img_folder, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# # Example usage
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize to 224x224 for consistency
#     transforms.ToTensor(),  # Convert image to PyTorch tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
# ])
#
# # Paths to your data
# train_json_path = "data/deepfashion-multimodal/train_captions.json"
# test_json_path = "data/deepfashion-multimodal/test_captions.json"
# img_folder = "data/deepfashion-multimodal/images"
#
# # Create the full dataset
# train_dataset = ImageDescriptionDataset(json_path=train_json_path, img_folder=img_folder, transform=transform)
# test_dataset = ImageDescriptionDataset(json_path=test_json_path, img_folder=img_folder, transform=transform)
#
# # Create DataLoaders
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# if __name__ == '__main__':
#     # Example usage
#     for images, descriptions in train_loader:
#         # Do something with the data
#         print(images.shape)
#         print(type(descriptions), len(descriptions), descriptions[0])
#         break
