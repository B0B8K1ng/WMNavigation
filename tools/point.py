from PIL import Image, ImageDraw
import math


def draw_star(draw, center, size, color):
    """
    在指定位置绘制一个五角星

    参数:
    - draw: ImageDraw 对象，用于绘制图形
    - center: 五角星中心的 (x, y) 坐标
    - size: 五角星的大小
    - color: 五角星的颜色
    """
    # 五角星的角度
    angle = math.pi / 5  # 每个角之间的角度
    points = []

    # 计算五角星的五个点
    for i in range(5):
        # 外部点
        outer_x = center[0] + size * math.cos(i * 2 * angle)
        outer_y = center[1] - size * math.sin(i * 2 * angle)
        points.append((outer_x, outer_y))
        # 内部点
        inner_x = center[0] + (size / 2) * math.cos((i * 2 + 1) * angle)
        inner_y = center[1] - (size / 2) * math.sin((i * 2 + 1) * angle)
        points.append((inner_x, inner_y))

    # 绘制五角星
    draw.polygon(points, fill=color)


def add_star_to_image(image_path, relative_position, star_size, star_color, output_path):
    """
    在图片中指定的相对位置绘制一个五角星，并保存结果图片

    参数:
    - image_path: 输入图片路径
    - relative_position: 五角星的相对位置 [w, h]，w 为相对宽度比例，h 为相对高度比例
    - star_size: 五角星的大小
    - star_color: 五角星的颜色 (例如: 'red', 'blue', (255, 0, 0) 等)
    - output_path: 输出图片路径
    """
    # 打开图片
    image = Image.open(image_path)
    width, height = image.size

    # 将相对坐标转换为像素坐标
    x = relative_position[0] * width
    y = relative_position[1] * height

    # 创建一个 ImageDraw 对象用于绘制
    draw = ImageDraw.Draw(image)

    # 绘制五角星
    draw_star(draw, (x, y), star_size, star_color)

    # 保存修改后的图片
    image.save(output_path)
    image.show()  # 可选，显示修改后的图片


# 使用示例
image_path = "/file_system/vepfs/algorithm/dujun.nie/1.png"  # 输入图片路径
relative_position = [0.2, 0.1]  # 五角星的相对位置，例如图片中央
star_size = 10  # 五角星的大小
star_color = 'red'  # 五角星的颜色
output_path = '/file_system/vepfs/algorithm/dujun.nie/3.jpg'  # 输出图片路径

# 在图片上添加五角星
add_star_to_image(image_path, relative_position, star_size, star_color, output_path)
