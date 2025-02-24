# import matplotlib
# matplotlib.use('Agg')  # 或者 'Agg'，'Qt5Agg'，根据你的需求
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
#
# # 加载图片
# image_path = "/file_system/vepfs/algorithm/dujun.nie/code/WMNav/VLMnav/logs/ObjectNav_hm3dv1_v7/2_of_50/117_853/step0/color_sensor.png"  # 替换为你的图片路径
# image = Image.open(image_path)
# width, height = image.size
#
# # 创建一个图像和坐标轴
# fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)  # 设置画布大小和DPI
# ax.imshow(image)
#
# # 获取图片的中心点
# center_x = width / 2
# center_y = height / 2
#
# # 设置坐标轴的范围
# ax.set_xlim(0, width)
# ax.set_ylim(height, 0)  # 注意：图像的y轴是从上到下的，所以需要反转
#
# # 设置坐标轴的原点在图片中心
# ax.spines['left'].set_position(('data', center_x))
# ax.spines['bottom'].set_position(('data', center_y))
#
# # 隐藏上边和右边的轴线
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
#
# # 设置x轴和y轴的刻度
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
#
# # 设置刻度的步长为0.2（按比例缩放）
# step = 0.1
# x_ticks = np.arange(-1, 1.1, step) * (width / 2) + center_x
# y_ticks = np.arange(-1, 1.1, step) * (height / 2) + center_y
#
# # 设置刻度标签
# x_labels = [f'{x:.1f}' for x in np.arange(0, 1.1, step)]
# y_labels = [f'{y:.1f}' for y in np.arange(0, 1.1, step)]
#
# ax.set_xticks(x_ticks)
# ax.set_xticklabels(x_labels, color='red')
# ax.set_yticks(y_ticks)
# ax.set_yticklabels(y_labels, color='red')
#
# # 绘制x轴和y轴
# ax.axhline(center_y, color='red', linewidth=1)
# ax.axvline(center_x, color='red', linewidth=1)
#
# # 设置坐标轴线颜色为红色
# ax.spines['left'].set_color('red')
# ax.spines['bottom'].set_color('red')
#
# # 设置刻度线颜色为红色
# ax.tick_params(axis='x', colors='red', which='both', direction='inout', length=5, width=1)
# ax.tick_params(axis='y', colors='red', which='both', direction='inout', length=5, width=1)
#
# # 显示图像
# output_path = "/file_system/vepfs/algorithm/dujun.nie/1.png"  # 替换为你希望保存的路径
# plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)  # 保持原始尺寸



from PIL import Image, ImageDraw, ImageFont

# 打开图片
image_path = "/file_system/nas/algorithm/dujun.nie/logs/ObjectNav_hm3dv1_v7/2_of_50/117_853/step0/color_sensor.png"  # 替换为你的图片路径
image = Image.open(image_path)
width, height = image.size

# 创建一个绘图对象
draw = ImageDraw.Draw(image)

# 定义颜色和字体
color = (255, 0, 0)  # 红色
try:
    font = ImageFont.truetype("arial.ttf", size=15)  # 使用系统字体
except IOError:
    font = ImageFont.load_default()  # 如果找不到字体，使用默认字体

# 绘制水平轴（X轴）
y_center = height // 2
draw.line((0, y_center, width, y_center), fill=color)  # 绘制水平轴线

# 绘制竖直轴（Y轴）
x_center = width // 2
draw.line((x_center, 0, x_center, height), fill=color)  # 绘制竖直轴线

# 在水平轴上绘制刻度
for i in range(1, 10):
    x = int(width * (i / 10))
    draw.line((x, y_center - 5, x, y_center + 5), fill=color)  # 绘制刻度线
    text = f"{i/10:.1f}"
    # 使用 getbbox 计算文本尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text((x - text_width // 2, y_center + 10), text, fill=color, font=font)  # 绘制刻度值

# 在竖直轴上绘制刻度
for i in range(1, 10):
    y = int(height * (i / 10))
    draw.line((x_center - 5, y, x_center + 5, y), fill=color)  # 绘制刻度线
    text = f"{i/10:.1f}"
    # 使用 getbbox 计算文本尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    draw.text((x_center + 10, y - text_height // 2), text, fill=color, font=font)  # 绘制刻度值

# 保存结果
output_path = "/file_system/vepfs/algorithm/dujun.nie/1.png"  # 替换为你想保存的路径
image.save(output_path)

print(f"图片已保存到 {output_path}")