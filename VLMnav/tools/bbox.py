import cv2

# 读取图片
image = cv2.imread("/file_system/vepfs/algorithm/dujun.nie/code/WMNav/VLMnav/logs/ObjectNav_version_7_pro_improve_reset/4_of_50/83_890/step0/color_sensor.png")

# 定义检测框的位置和尺寸
# 假设左上角坐标为 (450, 100)，宽度为 200，高度为 100
top_left = (450, 100)  # 左上角坐标
bottom_right = (450 + 200, 100 + 100)  # 右下角坐标

# 在图片上绘制矩形框 (参数：图片, 左上角, 右下角, 颜色, 线宽)
cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # 绿色矩形框，线宽为2


# 如果需要保存修改后的图片
cv2.imwrite('/file_system/vepfs/algorithm/dujun.nie/image_with_box.jpg', image)
