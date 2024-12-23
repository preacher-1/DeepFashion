import os
from zhipuai import ZhipuAI
from PIL import Image
import base64
import json
import matplotlib.pyplot as plt
from main import config

prompt = """
## Task

- Describe the clothing features of the person in the given image.
- Focus on details such as clothing type, fabric, pattern, length, and any visible accessories.

## Constraints

- Description must be in English.
- Keep your description length between 40 and 60 words.
"""

example_image_path = os.path.join(
    config["image_dir"], "WOMEN-Pants-id_00003334-01_1_front.jpg"
)

example_caption = "This person is wearing a long-sleeve sweater with pure color patterns. The sweater is with \
cotton fabric and its neckline is v-shape. This person wears a long trousers. The trousers are with cotton fabric and \
pure color patterns. The lady also wears an outer clothing, with knitting fabric and solid color patterns. This \
person wears a hat. There is an accessory on her wrist. The person is wearing a ring on her finger."


def load_test_image(test_idx=42):
    """
    return:
    """
    with open(config["test_json_path"], "r", encoding="utf-8") as f:
        test_json = json.load(f)

    image_path = list(test_json.keys())[test_idx]
    image_caption = list(test_json.values())[test_idx]

    image_path = os.path.join(config["image_dir"], image_path)

    with open(image_path, "rb") as imgf:
        image_base = base64.b64encode(imgf.read()).decode("utf-8")
    image = Image.open(image_path)
    return image, image_base, image_caption


def make_message(image_base, one_shot=False):
    if not one_shot:
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_base}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        with open(example_image_path, "rb") as img_file:
            example_image_base = base64.b64encode(img_file.read()).decode("utf-8")
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": example_image_base}},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "The upper clothing has long sleeves, cotton fabric and solid color patterns. The "
                                "neckline of it is v-shape. The lower clothing is of long length. The fabric is denim "
                                "and it has solid color patterns. This lady also wears an outer clothing, with cotton "
                                "fabric and complicated patterns. This female is wearing a ring on her finger. This "
                                "female has neckwear.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_base}},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    return message


def chat_to_generate(
        client: ZhipuAI, image_base: str, model="glm-4v-flash", one_shot=False
) -> str:
    """
    :param ZhipuAI client: client
    :param str image_base: base64 encoded image
    :param str model: model name, one of "glm-4v-flash", "glm-4v-plus", "glm-4v"
    :param bool one_shot: one-shot or not
    :return: str generated caption
    """
    model_list = ["glm-4v-plus", "glm-4v-flash", "glm-4v"]
    assert model.lower() in model_list, f"model should be in {model_list}"
    if one_shot and model.lower() == "glm-4v-flash":
        print("glm-4v-flash not support one-shot")
        return ""

    try:
        response = client.chat.completions.create(
            model=model,  # glm-4v-flash, glm-4v-plus
            messages=make_message(image_base, one_shot),
        )
    except Exception as e:
        print(f"Error: {e}")
        return ""
    return response.choices[0].message.content


def llm(api_key):
    client = ZhipuAI(api_key=api_key)
    test_image, image_base, target_caption = load_test_image()

    # 显示测试图片
    plt.figure(figsize=(12, 4))
    plt.imshow(test_image)
    plt.title('测试图片')
    plt.axis('off')
    plt.show()

    for i, model in enumerate(["glm-4v-flash", "glm-4v-plus"]):
        for j, one_shot in enumerate([False, True], start=1):
            print(f"模型：{model}, one-shot：{one_shot}")
            caption = chat_to_generate(client, image_base, model=model, one_shot=one_shot)
            print(f"生成描述:\n{caption}")
            print(f"参考描述:\n{target_caption}")
            print("=" * 50)


if __name__ == "__main__":
    llm()

"""
模型：glm-4v-flash, one-shot：False
生成描述:
The individual is wearing a fitted blue camisole with thin straps. The pants are form-fitting black jeans that appear 
to have a high waist and button closure at the front. They're paired with classic black lace-up shoes featuring 
brogue detailing for an elegant touch.
参考描述:
This female wears a sleeveless tank top with solid color patterns and long trousers. The tank top is with cotton 
fabric and it has a suspenders neckline. The trousers are with cotton fabric and pure color patterns.
==================================================
模型：glm-4v-flash, one-shot：True
glm-4v-flash not support one-shot
生成描述:

参考描述:
This female wears a sleeveless tank top with solid color patterns and long trousers. The tank top is with cotton 
fabric and it has a suspenders neckline. The trousers are with cotton fabric and pure color patterns.
==================================================
模型：glm-4v-plus, one-shot：False
生成描述:
The person is wearing a fitted, sleeveless blue top with thin spaghetti straps. The fabric appears to be smooth and 
stretchy. They are also wearing form-fitting black pants with a visible button closure at the waist. The outfit is 
completed with black lace-up shoes.
参考描述:
This female wears a sleeveless tank top with solid color patterns and long trousers. The tank top is with cotton 
fabric and it has a suspenders neckline. The trousers are with cotton fabric and pure color patterns.
==================================================
模型：glm-4v-plus, one-shot：True
生成描述:
The woman is wearing a sleeveless, blue top with thin straps, paired with form-fitting, black jeans. She is also 
wearing black, lace-up shoes. Her hair is styled in loose waves, and she has a natural makeup look.
参考描述:
This female wears a sleeveless tank top with solid color patterns and long trousers. The tank top is with cotton 
fabric and it has a suspenders neckline. The trousers are with cotton fabric and pure color patterns.
==================================================
"""
