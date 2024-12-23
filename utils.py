import torch
from rouge import Rouge
import numpy as np


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"Total": total_num, "Trainable": trainable_num}


def decode_sequence(
    seq: torch.Tensor, vocab_word2id: dict, vocab_idx2word: dict
) -> str:
    decoded_seq = []
    for i in seq:
        if i.item() == vocab_word2id["[END]"]:
            break
        decoded_seq.append(vocab_idx2word[i.item()])
    return " ".join(decoded_seq[1:])


def generate_caption(model, image, vocab, vocab_idx2word, device, max_length=50):
    model.eval()
    with torch.no_grad():
        # 提取图像特征
        features = model.cnn(image.unsqueeze(0))
        features = model.feature_projection(features.view(1, 2048, -1).permute(2, 0, 1))

        # 初始化序列
        seq = torch.LongTensor([[vocab["[START]"]]]).to(device)

        # 逐词生成
        for i in range(max_length):
            out = model.transformer(features, model.embedding(seq).permute(1, 0, 2))
            pred = model.fc_out(out[-1, :])
            next_word = pred.argmax(dim=1)

            seq = torch.cat([seq, next_word.unsqueeze(0)], dim=1)

            if next_word.item() == vocab["[END]"]:
                break

        return decode_sequence(seq.squeeze(), vocab_idx2word)


def calculate_rouge_scores(predictions, references):
    """
    计算ROUGE-L分数
    Args:
        predictions: 预测的描述列表
        references: 参考描述列表
    Returns:
        dict: 包含ROUGE-L分数的字典
    """
    rouge = Rouge()
    # 确保输入不为空
    filtered_pairs = [
        (p, r)
        for p, r in zip(predictions, references)
        if len(p.strip()) > 0 and len(r.strip()) > 0
    ]

    if not filtered_pairs:
        return {"rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}}

    predictions, references = zip(*filtered_pairs)

    try:
        scores = rouge.get_scores(predictions, references, avg=True)
        return scores
    except Exception as e:
        print(f"Error calculating ROUGE scores: {e}")
        return {"rouge-l": {"f": 0.0, "p": 0.0, "r": 0.0}}


def evaluate_model(model, dataloader, vocab, device, num_samples=None):
    """
    使用ROUGE-L评估模型性能
    Args:
        model: 图像描述模型
        dataloader: 数据加载器
        vocab: 词表
        device: 设备
        num_samples: 评估样本数量（None表示使用全部数据）
    Returns:
        dict: ROUGE-L分数
    """
    model.eval()
    predictions = []
    references = []

    # 创建词表的反向映射
    id2word = {v: k for k, v in vocab.items()}

    with torch.no_grad():
        for i, (images, captions) in enumerate(dataloader):
            if num_samples and i * dataloader.batch_size >= num_samples:
                break

            images = images.to(device)
            batch_size = images.size(0)

            # 生成描述
            for j in range(batch_size):
                if num_samples and len(predictions) >= num_samples:
                    break

                # 生成预测描述
                generated_words = model.generate(
                    images[j : j + 1], max_len=50, temperature=1.0, vocab=vocab
                )
                pred_caption = " ".join(generated_words)

                # 获取参考描述
                ref_caption = decode_sequence(captions[j], vocab, id2word)

                predictions.append(pred_caption)
                references.append(ref_caption)

    # 计算ROUGE分数
    rouge_scores = calculate_rouge_scores(predictions, references)

    return rouge_scores, predictions, references
