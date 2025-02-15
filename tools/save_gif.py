import os
import shutil
import glob
import concurrent.futures


def process_line(line, success_folder, fail_folder):
    # 分割行数据，获取id和status
    parts = line.strip().split(':')
    if len(parts) != 2:
        return  # 跳过无效行
    id_str, status_str = parts
    try:
        id = int(id_str)  # 解析id
        status = int(status_str)  # 解析status
    except ValueError:
        return  # 如果解析失败，跳过此行

    # 目标路径模板
    gif_path = f'/file_system/vepfs/algorithm/dujun.nie/code/WMNav/VLMnav/logs/ObjectNav_version_8_hm3dv1/*/{id}_*/animation.gif'

    # 根据status选择目标文件夹
    if status == 1:
        target_folder = success_folder
    else:
        target_folder = fail_folder

    # 构造新的文件名
    target_filename = f'animation_{id}.gif'
    target_path = os.path.join(target_folder, target_filename)

    # 复制文件
    try:
        gif_file_path = glob.glob(gif_path)  # 获取路径匹配的文件
        if gif_file_path:
            shutil.copy(gif_file_path[0], target_path)  # 复制文件
            print(f"Successfully copied: {gif_file_path[0]} to {target_path}")
        else:
            print(f"GIF not found for id {id}")
    except Exception as e:
        print(f"Error processing id {id}: {e}")


def process_file(txt_file, success_folder, fail_folder):
    # 确保目标文件夹存在，不存在则创建
    os.makedirs(success_folder, exist_ok=True)
    os.makedirs(fail_folder, exist_ok=True)

    with open(txt_file, 'r') as f:
        lines = f.readlines()

    # 使用ThreadPoolExecutor来并行处理每一行
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交每一行的任务到线程池
        futures = [executor.submit(process_line, line, success_folder, fail_folder) for line in lines]

        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            future.result()  # 获取任务结果，捕获可能的异常


if __name__ == '__main__':

    # 使用示例
    txt_file = '/file_system/vepfs/algorithm/dujun.nie/data/vlmnav/baseline_v8_hm3dv1_real.txt'  # 输入的txt文件路径
    success_folder = '/file_system/vepfs/algorithm/dujun.nie/data/vlmnav/baseline_v8_hm3dv1/success_gif'  # 成功的GIF文件保存目录
    fail_folder = '/file_system/vepfs/algorithm/dujun.nie/data/vlmnav/baseline_v8_hm3dv1/fail_gif'  # 失败的GIF文件保存目录

    process_file(txt_file, success_folder, fail_folder)
