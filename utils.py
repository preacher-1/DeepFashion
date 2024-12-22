import torch


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def decode_sequence(seq, vocab_idx2word):
    decoded_seq = []
    for i in seq:
        if i.item() == vocab_idx2word['[END]']:
            break
        decoded_seq.append(vocab_idx2word[i.item()])
    return ''.join(decoded_seq)


def generate_caption(model, image, vocab, vocab_idx2word, device, max_length=50):
    model.eval()
    with torch.no_grad():
        # 提取图像特征
        features = model.cnn(image.unsqueeze(0))
        features = model.feature_projection(features.view(1, 2048, -1).permute(2, 0, 1))

        # 初始化序列
        seq = torch.LongTensor([[vocab['[START]']]]).to(device)

        # 逐词生成
        for i in range(max_length):
            out = model.transformer(features, model.embedding(seq).permute(1, 0, 2))
            pred = model.fc_out(out[-1, :])
            next_word = pred.argmax(dim=1)

            seq = torch.cat([seq, next_word.unsqueeze(0)], dim=1)

            if next_word.item() == vocab['[END]']:
                break

        return decode_sequence(seq.squeeze(), vocab_idx2word)
