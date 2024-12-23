from resnet_module import resnet34
from torchvision.models import resnet18
from NVAE_residual_module import ConvTower
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch
from torch.nn import CrossEntropyLoss
from utils import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    resnet = resnet18(num_classes=10, pretrained=False)
    nvae = ConvTower(in_channels=3, out_channels=10)  # 10 = outchannels 不影响num_classes
    print(f"ResNet18 has {get_parameter_number(resnet)['Total'] / 1e6:.3f}M parameters")
    print(f"NVAE has {get_parameter_number(nvae)['Total'] / 1e6:.3f}M parameters")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet.to(device)
    nvae.to(device)

    # Train both models on CIFAR10
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    optimizer_resnet = Adam(resnet.parameters(), lr=1e-3)
    optimizer_nvae = Adam(nvae.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()

    resnet_train_loss = []
    nvae_train_loss = []
    resnet_test_loss = []
    nvae_test_loss = []
    resnet_test_acc = []
    nvae_test_acc = []
    for epoch in range(20):
        resnet.train()
        nvae.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer_resnet.zero_grad()
            optimizer_nvae.zero_grad()

            # ResNet34
            outputs_resnet = resnet(images)
            loss_resnet = criterion(outputs_resnet, labels)
            loss_resnet.backward()
            optimizer_resnet.step()

            # NVAE
            outputs_nvae = nvae(images)
            loss_nvae = criterion(outputs_nvae, labels)
            loss_nvae.backward()
            optimizer_nvae.step()

            if i % 100 == 0:
                print(
                    f"Epoch {epoch}, Batch {i}, Loss ResNet34: {loss_resnet.item():.4f}, Loss NVAE: {loss_nvae.item():.4f}")
                resnet_train_loss.append(loss_resnet.item())
                nvae_train_loss.append(loss_nvae.item())

        # Test both models on CIFAR10
        resnet.eval()
        nvae.eval()
        test_loss_resnet = 0
        test_loss_nvae = 0
        correct_resnet = 0
        correct_nvae = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs_resnet = resnet(images)
                test_loss_resnet += criterion(outputs_resnet, labels).item()
                _, predicted_resnet = torch.max(outputs_resnet.data, 1)
                correct_resnet += (predicted_resnet == labels).sum().item()

                outputs_nvae = nvae(images)
                test_loss_nvae += criterion(outputs_nvae, labels).item()
                _, predicted_nvae = torch.max(outputs_nvae.data, 1)
                correct_nvae += (predicted_nvae == labels).sum().item()

        test_loss_resnet /= len(test_loader)
        test_loss_nvae /= len(test_loader)
        resnet_test_loss.append(test_loss_resnet)
        nvae_test_loss.append(test_loss_nvae)
        resnet_test_acc.append(100 * correct_resnet / len(test_dataset))
        nvae_test_acc.append(100 * correct_nvae / len(test_dataset))
        print(f"Epoch {epoch}, Test Loss ResNet34: {test_loss_resnet:.4f}, Test Loss NVAE: {test_loss_nvae:.4f}")
        print(
            f"Epoch {epoch}, Test Accuracy ResNet34: {100 * correct_resnet / len(test_dataset):.2f}%, Test Accuracy NVAE: {100 * correct_nvae / len(test_dataset):.2f}%")

    # Plot loss and accuracy curves
    plt.plot(resnet_train_loss, label='ResNet34 Train Loss')
    plt.plot(nvae_train_loss, label='NVAE Train Loss')
    plt.plot(resnet_test_loss, label='ResNet34 Test Loss')
    plt.plot(nvae_test_loss, label='NVAE Test Loss')
    plt.plot(resnet_test_acc, label='ResNet34 Test Accuracy')
    plt.plot(nvae_test_acc, label='NVAE Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.show()
