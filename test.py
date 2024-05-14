import numpy as np
from multiprocessing import Pool, cpu_count

# 矩阵加法函数
def matrix_add_subsection(a, b, start, end):
    return np.add(a[start:end], b[start:end])

# 将矩阵划分为多个子部分并分配给不同的进程
def parallel_matrix_add(a, b):
    num_processes = cpu_count()  # 获取CPU核心数

    print(num_processes)
    # exit(0)
    rows = a.shape[0]
    step = rows // num_processes

    # 创建进程池
    pool = Pool(processes=num_processes)

    # 分配任务
    results = [pool.apply_async(matrix_add_subsection, args=(a, b, i, i + step)) for i in range(0, rows, step)]

    # 收集结果
    results = [p.get() for p in results]

    # 合并结果
    result_matrix = np.concatenate(results, axis=0)

    # 关闭进程池
    pool.close()
    pool.join()

    return result_matrix

# 创建两个矩阵
a = np.random.rand(1000, 1000)
b = np.random.rand(1000, 1000)

# 使用多进程进行矩阵加法
result = parallel_matrix_add(a, b)

# 输出结果的一部分以验证
print(result[:5, :5])
