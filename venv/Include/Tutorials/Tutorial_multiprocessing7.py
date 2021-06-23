import multiprocessing
import parmap
import numpy as np

num_cores = multiprocessing.cpu_count() # 12

def square(input_list):
    y = []
    for x in input_list:
        y.append(x*x)

    return y


if __name__ == '__main__':
    data = list(range(1, 25000000))
    splited_data = np.array_split(data, num_cores*100)
    splited_data = [x.tolist() for x in splited_data]

    result = parmap.map(square, splited_data, pm_pbar=True, pm_processes=num_cores)
    #print(result)
    #print(splited_data)