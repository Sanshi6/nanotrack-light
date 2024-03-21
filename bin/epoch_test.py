from multiprocessing import Pool
import subprocess


# 定义一个函数来加载模型和计算AUC
def evaluate_model(checkpoint_path):
    # 调用subprocess运行外部命令，运行模型评估
    print(checkpoint_path)

    command = f"python tools/test.py --snapshot {checkpoint_path} --config models/config/SubNet.yaml --dataset OTB100"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:  # 非零返回码通常表示发生了错误
        print(f"error： {command}")
        print(f"print：\n{result.stderr}")

    # command = f"python tools/eval.py"
    # result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # if result.returncode != 0:  # 非零返回码通常表示发生了错误
    #     print(f"error： {command}")
    #     print(f"print：\n{result.stderr}")

        
def main():
    # 生成你要测试的模型文件列表
    checkpoints = [f"snapshot/checkpoint_e{i}.pth" for i in range(5, 51)]

    # 最大并发进程数
    max_processes = 6

    # 创建一个进程池
    with Pool(processes=max_processes) as pool:
        # pool.map函数会阻塞直到所有结果都完成
        auc_list = pool.map(evaluate_model, checkpoints)

    print("all test success")
    # print(f"AUC列表: {auc_list}")


# 保护程序的入口点
if __name__ == '__main__':
    # 这是Windows上使用多进程所必需的
    main()
