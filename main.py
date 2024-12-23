import torch
import torch.nn as nn
from datasets import get_dataloaders
from model import ImageCaptioningModel
from trainer import CaptioningTrainer
from utils import *
from llm import *
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import argparse

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 配置参数
config = {
    "image_dir": "data/deepfashion-multimodal/images",
    "train_json_path": "data/deepfashion-multimodal/train_captions.json",
    "test_json_path": "data/deepfashion-multimodal/test_captions.json",
    "vocab_file": "data/deepfashion-multimodal/vocab.json",
    "batch_size": 64,
    "hidden_dim": 512,
    "nhead": 8,
    "num_encoder_layers": 6,
    "num_decoder_layers": 6,
    "learning_rate": 2e-4,
    "epochs": 30,
    "clip_grad": 5.0,
    "api_key": "enteryour.zhipuapikey",
}


def test_generation(model, val_loader, vocab, device, num_samples=5):
    """测试模型生成能力"""
    model.eval()

    # 获取一些测试样本
    test_images, test_captions = next(iter(val_loader))
    test_images = test_images[:num_samples].to(device)

    id2word = {v: k for k, v in vocab.items()}

    print("\n=== 生成示例 ===")
    for i in range(num_samples):
        # 使用模型生成描述
        generated_words = model.generate(
            test_images[i: i + 1], max_len=50, temperature=0.7, vocab=vocab
        )

        # 获取真实描述
        true_caption = decode_sequence(test_captions[i], vocab, id2word)

        print(f"\n样本 {i + 1}:")
        print(f"生成描述: {' '.join(generated_words)}")
        print(f"真实描述: {true_caption}")


def check_frozen_params(model):
    """检查模型中的冻结参数"""
    for name, param in model.named_parameters():
        if "cnn" in name:
            assert not param.requires_grad, f"CNN参数 {name} 未被冻结"
        else:
            assert param.requires_grad, f"非CNN参数 {name} 被错误地冻结"
    print("参数冻结检查通过！")


# 训练流程
def main():
    writer = SummaryWriter("logs/deepfashion_multimodal_experiment_1")

    # 1. 数据加载
    train_loader, test_loader, vocab = get_dataloaders(
        train_json_path=config["train_json_path"],
        test_json_path=config["test_json_path"],
        image_dir=config["image_dir"],
        batch_size=config["batch_size"],
    )

    # 2. 模型初始化
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        hidden_dim=config["hidden_dim"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 检查参数冻结情况
    check_frozen_params(model)

    # 3. 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["[PAD]"])

    # 4. 训练器
    trainer = CaptioningTrainer(model, optimizer, criterion, device)

    # 5. 训练循环，同时记录损失，并实现早停
    loss_history, steps_loss_history = [], []
    val_loss_history = []
    best_loss = float("inf")
    best_model_weights = None
    epochs_without_improvement = 0
    patience = 5
    print(f"Training started with {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
        try:
            loss, steps_loss = trainer.train_epoch(train_loader, writer, epoch)
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

            # 验证
            val_loss = trainer.validate(test_loader)
            val_loss_history.append(val_loss)
            writer.add_scalar("val_loss", val_loss, epoch)

            # 记录损失
            loss_history.append(loss)
            steps_loss_history.extend(steps_loss)

            # 保存模型
            if (epoch + 1) % 3 == 0:
                torch.save(model.state_dict(), f"checkpoints/model_{epoch + 1}.pth")
                print(f"Model saved to checkpoints/model_{epoch + 1}.pth")

            # 早停
            # 如果验证集上的损失更好，则更新最佳模型参数
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
                best_model_weights = model.state_dict()
            else:
                epochs_without_improvement += 1

            # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
            if epochs_without_improvement == patience:
                model.load_state_dict(best_model_weights)
                print(f"Early stopping at epoch {epoch + 1}...")
                break
        except Exception as e:
            print(f"Training stopped at epoch {epoch + 1}, error: {e}")
            if not os.path.exists(f"checkpoints/model_{epoch + 1}.pth"):
                torch.save(model.state_dict(), f"checkpoints/model_{epoch + 1}.pth")
                print(f"Model saved to checkpoints/model_{epoch + 1}.pth")
            break


# 测试流程
def test():
    # 1. 数据加载
    train_loader, test_loader, vocab = get_dataloaders(
        train_json_path=config["train_json_path"],
        test_json_path=config["test_json_path"],
        image_dir=config["image_dir"],
        batch_size=config["batch_size"],
    )

    # 2. 模型初始化
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        hidden_dim=config["hidden_dim"],
        nhead=config["nhead"],
        num_encoder_layers=config["num_encoder_layers"],
        num_decoder_layers=config["num_decoder_layers"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 3. 载入权重
    model.load_state_dict(torch.load("checkpoints/model_12.pth", map_location=device))

    # 4. 测试生成
    # test_generation(model, test_loader, vocab, device)

    # 5. 进行ROUGE-L评估
    print("\n开始ROUGE-L评估...")
    rouge_scores, predictions, references = evaluate_model(
        model, test_loader, vocab, device, num_samples=50  # 可以调整评估样本数量
    )

    # 打印ROUGE-L分数
    print("\nROUGE-L Scores:")
    print(f"F1: {rouge_scores['rouge-l']['f']:.4f}")
    print(f"Precision: {rouge_scores['rouge-l']['p']:.4f}")
    print(f"Recall: {rouge_scores['rouge-l']['r']:.4f}")

    # ROUGE-L Scores:
    # F1: 0.6302
    # Precision: 0.6496
    # Recall: 0.6244

    # 保存评估结果
    current_time = datetime.now().strftime("%y%m%d%H%M%S")
    evaluation_results = {
        "rouge_scores": rouge_scores,
        "predictions": predictions[:10],  # 保存前10个预测样例
        "references": references[:10],  # 保存前10个参考样例
        "timestamp": current_time,
    }

    with open(f"docs/evaluation_results_{current_time}.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a LLM-based chatbot.")
    parser.add_argument(
        "--test", "-t", action="store_true", help="Evaluate the model on test set."
    )
    parser.add_argument("--llm", "-l", action="store_true", help="Call the ZhipuAI LLM API to generate captions.")

    args = parser.parse_args()

    if args.test:
        test()
    elif args.llm:
        llm(api_key=config["api_key"])
    else:
        main()
