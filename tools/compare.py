# 读取文件内容并解析成字典
def read_file(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # 去掉行尾换行符并分割成序号和数据
            index, value = line.strip().split(':')
            data_dict[int(index)] = int(value)  # 将序号和数据存储为整数
    return data_dict


# 找出在 file1 为 1 而在 file2 为 0 的序号，和在 file1 为 0 而在 file2 为 1 的序号
def compare_files(file1_path, file2_path):
    # 读取两个文件
    data1 = read_file(file1_path)
    data2 = read_file(file2_path)

    # 获取在 file1 为 1 而在 file2 为 0 的序号
    file1_1_file2_0 = [idx for idx in data1 if data1[idx] == 1 and data2.get(idx, 0) == 0]

    # 获取在 file1 为 0 而在 file2 为 1 的序号
    file1_0_file2_1 = [idx for idx in data1 if data1[idx] == 0 and data2.get(idx, 0) == 1]

    return file1_1_file2_0, file1_0_file2_1


# 将不同的序号写入结果文件
def write_results_to_file(different_indices, output_path):
    with open(output_path, 'w') as output_file:
        output_file.write("Data differences found at indices:\n")
        for idx in different_indices:
            output_file.write(f"{idx}\n")


# 主程序
def main():
    v0_path = "/file_system/vepfs/algorithm/dujun.nie/data/vlmnav/baseline_v8_hm3dv1_real.txt"  # 第一个文件路径
    v1_path = "/file_system/vepfs/algorithm/dujun.nie/data/vlmnav/baseline_v8_hm3dv1.txt"  # 第二个文件路径

    # 获取两个文件中不同情况的序号
    file1_1_file2_0, file1_0_file2_1 = compare_files(v0_path, v1_path)

    # 输出结果到文件
    write_results_to_file(file1_1_file2_0, '/file_system/vepfs/algorithm/dujun.nie/data/vlmnav/8_hm3dv1_realSuccess_fakeFail.txt')
    write_results_to_file(file1_0_file2_1, '/file_system/vepfs/algorithm/dujun.nie/data/vlmnav/8_hm3dv1_realFail_fakeSuccess.txt')


if __name__ == '__main__':
    main()

