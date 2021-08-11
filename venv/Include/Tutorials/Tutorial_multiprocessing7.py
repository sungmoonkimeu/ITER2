import multiprocessing
import parmap
import numpy as np
import tqdm

num_cores = multiprocessing.cpu_count() # 12

def square(input_list):
    y = []
    for x in input_list:
        y.append(x*x)

    return y

def myfunction(x, y, param1, param2):
    return x + y**2 + 3.14 + 42

if __name__ == '__main__':
    data = list(range(1, 25000))
    splited_data = np.array_split(data, num_cores*100)
    splited_data = [x.tolist() for x in splited_data]

    result = parmap.map(square, splited_data, pm_pbar=True, pm_processes=num_cores)
    #print(result)
    #print(splited_data)


    # z = [myfunction(x, y, argument1, argument2) for (x,y) in mylist]
    # z = parmap.starmap(myfunction, mylist, argument1, argument2)
    # 당신이 원하는 것: 
    # listx = [1, 2, 3, 4, 5, 6]
    listx = np.arange(0,100000,1)
    # listy = [2, 3, 4, 5, 6, 7]
    listy = np.arange(0, 100000, 1)
    param1 = 3.14
    param2 = 42 
    listz = [] 

    # 병렬로 실행:
    
    listz = parmap.starmap(myfunction, zip(listx, listy), param1, param2,  pm_processes=num_cores)
    print(listz)
