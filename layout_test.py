import subprocess

# 定义 .sh 脚本的路径
script_path = '/home/mlx/affordance_handover/scripts.sh'

# 使用 subprocess.run 运行 .sh 脚本
result = subprocess.run(['bash', script_path], capture_output=True, text=True)

# 检查返回码
if result.returncode == 0:
    print("Script executed successfully")
else:
    print(f"Script failed with return code {result.returncode}")

# 打印脚本的输出和错误输出
print(result.stdout)
print(result.stderr)
