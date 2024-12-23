import json
import numpy as np
import matplotlib.pyplot as plt


def caption_length_count(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    caption_lengths = {}
    for _, caption in data.items():
        length = len(caption.split())
        if length in caption_lengths:
            caption_lengths[length] += 1
        else:
            caption_lengths[length] = 1

    return caption_lengths


def plot_caption_length_count(caption_lengths):
    plt.bar(caption_lengths.keys(), caption_lengths.values())
    plt.xlabel('Caption Length')
    plt.ylabel('Frequency')
    plt.title('Caption Length Distribution')
    plt.show()


def plot_caption_length_cumulative_count(caption_lengths):
    """
    Plots the cumulative distribution curve of caption lengths.
    """
    cumulative_count = []
    total_count = sum(caption_lengths.values())
    current_count = 0
    max_length = max(caption_lengths.keys())
    for length in range(1, max_length + 1):
        current_count += caption_lengths.get(length, 0)
        cumulative_count.append(current_count / total_count)

    plt.plot(np.arange(1, len(cumulative_count) + 1), cumulative_count)
    plt.plot([1, max_length], [0.8, 0.8], '--', color='gray')
    plt.xlabel('Caption Length')
    plt.ylabel('Cumulative Frequency')
    plt.title('Caption Length Cumulative Distribution')
    plt.show()


if __name__ == '__main__':
    json_file = '../data/deepfashion-multimodal/train_captions.json'
    caption_lengths = caption_length_count(json_file)
    plot_caption_length_count(caption_lengths)
    plot_caption_length_cumulative_count(caption_lengths)

    # Find maximum length of captions
    max_length = max(caption_lengths.keys())
    print(f'Maximum length of captions: {max_length}')

    # Find most frequent caption length
    most_frequent_length = max(caption_lengths, key=caption_lengths.get)
    print(f'Most frequent caption length: {most_frequent_length}')

    # Maximum length of captions: 93
    # Most frequent caption length: 51
