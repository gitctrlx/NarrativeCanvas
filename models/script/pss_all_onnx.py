import os
import subprocess

def process_onnx_files(source_folder, output_folder):
    # 定义日志文件夹路径
    log_folder = os.path.join(output_folder, "logs")

    # 确保输出文件夹和日志文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith(".onnx"):
            original_path = os.path.join(source_folder, filename)
            modified_filename = filename.replace(".onnx", "-pss.onnx")
            modified_path = os.path.join(output_folder, modified_filename)
            log_path = os.path.join(log_folder, filename.replace(".onnx", ".log"))

            # 构建 Polygraphy 命令
            command = [
                "polygraphy", "surgeon", "sanitize", original_path,
                "--fold-constant",
                "-o", modified_path
            ]

            # 运行命令并将输出重定向到日志文件
            with open(log_path, "w") as log_file:
                subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)

# 使用示例
# 替换 'source_folder_path' 为onnx的源文件夹路径
# 替换 'output_folder_path' 为onnx的输出文件夹路径
process_onnx_files('onnx', 'pt')

