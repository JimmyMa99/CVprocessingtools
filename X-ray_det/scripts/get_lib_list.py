import subprocess

# 获取已安装包的名称列表
completed_process = subprocess.run(['pip', 'list', '--format=freeze'], capture_output=True, text=True)
output = completed_process.stdout
packages = output.strip().split('\n')

# 创建并打开输出文件
output_file = open('lib_list.txt', 'w')

# 遍历每个已安装的包并获取其安装路径
for package in packages:
    package_name = package.split('==')[0]
    completed_process = subprocess.run(['pip', 'show', '-f', package_name], capture_output=True, text=True)
    output = completed_process.stdout

    # 写入包的详细信息到输出文件
    output_file.write(output)
    output_file.write('\n')

# 关闭输出文件
output_file.close()

print("Library list has been written to 'lib_list.txt' file.")
