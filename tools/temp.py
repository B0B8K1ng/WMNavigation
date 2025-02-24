from openai import OpenAI
import base64
import numpy as np
from PIL import Image
import io


# 从文件路径读取图片并转换为Base64编码
def encode_image_from_file(image_path):
    # 使用PIL库读取图片
    image = Image.open(image_path)

    # 将图片转换为numpy数组（H*W*C格式的RGB数组）
    image_array = np.array(image)

    # 将numpy数组转换回PIL图像
    image = Image.fromarray(image_array[:, :, :3], mode='RGB')

    # 将图像保存到字节流中
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")

    # 将字节流编码为Base64
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# 输入转发API Key
client = OpenAI(
    api_key="sk-8Quae3SbmJiVzQxQpOz937STnHLYdGJQiEQwboQMytdwMDfr",
    base_url="https://aigptapi.com/v1"
)

# 本地图片路径
image_path = "/file_system/vepfs/algorithm/dujun.nie/1.png"

# 从文件路径读取图片并转换为Base64编码
base64_image = encode_image_from_file(image_path)

response = client.chat.completions.create(
    model="gemini-2.0-flash-001",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "利用图中的刻度线，输出电视机的位置坐标，用<x, y>形式输出，其中x是水平坐标，y是竖直坐标"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{base64_image}",
                },
            ],
        }
    ],
    max_tokens=500,
    temperature=0,
    top_p=1,
    stream=False  # 是否开启流式输出
)

# 非流式输出获取结果
print(response.choices[0].message.content)