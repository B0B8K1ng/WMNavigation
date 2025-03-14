import matplotlib
matplotlib.use('Agg')  # 或者 'Agg'，'Qt5Agg'，根据你的需求
import matplotlib.pyplot as plt
import numpy as np

# 假设你的 txt 文件路径
file_path = '/file_system/vepfs/algorithm/dujun.nie/data/vlmnav/8_hm3dv1_realFail_fakeSuccess.txt'

# 读取 txt 文件
with open(file_path, 'r') as file:
    # 读取文件中的所有数字，去掉多余的换行符，并过滤掉非整数行
    data = []
    for line in file.readlines():
        try:
            # 尝试将每行转换为整数
            num = int(line.strip())
            data.append(num)
        except ValueError:
            # 如果不能转换为整数，跳过该行
            continue

# 如果没有有效数据，提前退出
if not data:
    print("No valid numbers found in the file.")
    exit()

# 设置区间长度
bin_width = 40

# 计算区间的范围
min_value = min(data)
max_value = max(data)

# 设置区间的边界
bins = np.arange(min_value, max_value + bin_width, bin_width)

# 绘制直方图
plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)

# 添加标签
plt.xlabel('Value Ranges')
plt.ylabel('Frequency')
plt.title('Frequency Distribution of Indices')

# 保存图表为 PNG 文件
plt.savefig('/file_system/vepfs/algorithm/dujun.nie/histogram.png', bbox_inches='tight')

# 关闭图表（不显示）
plt.close()

