import os
import re
import matplotlib
matplotlib.use('Agg')  # 或者 'Agg'，'Qt5Agg'，根据你的需求
import matplotlib.pyplot as plt
import numpy as np


# 定义函数：检查文件中的运行数据并判定成功或失败
def analyze_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 用来存储每个单元的结果，字典结构 {unit_id: result}
    results = {}

    current_unit = None
    found_step_39 = False  # 标记是否找到 Step 39

    for line in lines:
        # 检查是否是运行单元的开始
        start_match = re.search(r'===================STARTING RUN: (\d+_\d+) ===================', line)
        step_match = re.search(r'Step (\d+)', line)
        run_complete_match = re.search(r'===================RUN COMPLETE===================', line)

        if start_match:
            # 获取当前单元编号
            current_unit = start_match.group(1).split('_')[0]  # 提取编号部分
            found_step_39 = False  # 重置找到了 Step 39 的标记

        elif step_match and current_unit:
            # 检查是否出现了 Step 39
            step = int(step_match.group(1))
            if step == 39:
                found_step_39 = True  # 找到 Step 39，标记为失败

        elif run_complete_match and current_unit:
            # 到达 "RUN COMPLETE" 行时，决定成功或失败
            if found_step_39:
                results[current_unit] = 0  # 失败
            else:
                results[current_unit] = 1  # 成功

    return results


# 主函数：遍历指定文件夹，分析所有txt文件
def analyze_folder(folder_path, output_file):
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # 用来存储所有文件分析结果
    all_results = {}

    for txt_file in txt_files:
        file_path = os.path.join(folder_path, txt_file)
        print(f"Analyzing {txt_file}...")
        file_results = analyze_txt_file(file_path)

        # 将分析结果合并
        all_results.update(file_results)

    # 按照数字顺序对单元编号排序
    sorted_results = sorted(all_results.items(), key=lambda x: int(x[0]))

    # 计算成功和失败的数量
    total_units = len(sorted_results)
    success_count = sum(1 for result in sorted_results if result[1] == 1)
    failure_count = total_units - success_count
    success_rate = success_count / total_units * 100 if total_units > 0 else 0

    # 打印总共的单元数量和成功率
    print(f"\nTotal units: {total_units}")
    print(f"Success count: {success_count}")
    print(f"Failure count: {failure_count}")
    print(f"Success rate: {success_rate:.2f}%")

    # 将结果写入输出文件
    with open(output_file, 'w') as output:
        for unit, result in sorted_results:
            output.write(f"{unit}:{result}\n")

    # 可视化：绘制热力图
    #plot_heatmap(sorted_results, plot_path)



# # 可视化函数：绘制热力图
# def plot_heatmap(sorted_results, plot_path):
#     # 创建一个25x40的网格，初始化为0
#     grid = np.zeros((25, 40))
#
#     # 填充网格，1表示成功，0表示失败
#     for idx, (unit, result) in enumerate(sorted_results):
#         row = idx // 40  # 每40个单元一行
#         col = idx % 40   # 每行40个单元
#         grid[row, col] = result  # 1 表示成功，0 表示失败
#
#     # 绘制热力图
#     plt.figure(figsize=(12, 8))
#     plt.imshow(grid, cmap='RdYlGn', interpolation='nearest')
#     plt.colorbar(label='Success (green) / Failure (red)')
#
#     # 在每个网格上标注单元编号
#     for idx, (unit, result) in enumerate(sorted_results):
#         row = idx // 40
#         col = idx % 40
#         plt.text(col, row, unit, ha='center', va='center', color='black', fontsize=6)
#
#     # 设置标题
#     plt.title('Heatmap of Run Results (Success/Failure)')
#
#     # 保存图像到指定路径
#     plt.tight_layout()
#     plt.savefig(plot_path, dpi=300)
#
#     # 关闭图形以节省内存
#     plt.close()

def plot_heatmap(sorted_results, plot_path):
    # 创建一个20x50的网格，初始化为0
    grid = np.zeros((20, 50))

    # 填充网格，1表示成功，0表示失败
    for idx, (unit, result) in enumerate(sorted_results):
        row = idx % 20  # 每50个单元一行（修改为20行，50列）
        col = idx // 20   # 每行50个单元
        grid[row, col] = result  # 修改为row, col坐标填充

    plt.figure(figsize=(12, 5))
    # 使用imshow绘制热力图，旋转90度
    plt.imshow(grid, cmap='RdYlGn', interpolation='nearest')

    # 在每个网格上标注单元编号
    for idx, (unit, result) in enumerate(sorted_results):
        row = idx % 20
        col = idx // 20
        plt.text(col, row, unit, ha='center', va='center', color='black', fontsize=6)

    # 设置标题
    plt.title('Heatmap of Run Results (Success/Failure)', fontsize=16)


    # 保存图像到指定路径
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)

    # 关闭图形以节省内存
    plt.close()



# 设置文件夹路径和输出文件路径
folder_path = '/file_system/vepfs/algorithm/dujun.nie/code/WMNav/VLMnav/logs/ObjectNav_version_8_hm3dv1/'  # 替换为实际的文件夹路径
output_file = '/file_system/vepfs/algorithm/dujun.nie/data/vlmnav/baseline_v8_hm3dv1.txt'  # 输出结果文件
plot_path = '/file_system/vepfs/algorithm/dujun.nie/data/vlmnav/baseline_v8_hm3dv1.png'  # 可视化图像保存路径

# 调用主函数
analyze_folder(folder_path, output_file)
