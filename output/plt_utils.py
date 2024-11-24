import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib.ticker as ticker
import numpy as np

def parse_data(file_path, id):
    data = {
        "TopK": [], 
        "GCA": [],
        "MTR":[], 
        "STR": [],
        "STRS": [],
    }
    data_time = {
        "TopK": [], 
        "GCA": [],
        "MTR":[], 
        "STR": [],
        "STRS": [],
    }
    data_cost = {
        "TopK": [], 
        "GCA": [],
        "MTR":[], 
        "STR": [],
        "STRS": [],
    }
    data_apu = {
        "TopK": [], 
        "GCA": [],
        "MTR":[], 
        "STR": [],
        "STRS": [],
    }
    data_num_set = set()

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.split(' ')
            try:
                data_num = int(parts[id].split(':')[1].strip())
            except:
                data_num = float(parts[id].split(':')[1].strip())
                
            value = [x.split(':')[1].strip() for x in parts]
                
            profit = float(value[4])
            cost = float(value[6])
            avgTime = float(value[7])
            apu = float(value[8])
            
            data_name = value[0]
            model = value[1]
            assign = value[2]
            
            data_num_set.add(data_num)

            data[assign].append((data_num, profit))
            data_cost[assign].append((data_num, cost))
            data_time[assign].append((data_num, avgTime))
            data_apu[assign].append((data_num, apu))


    return data, data_cost, data_time, data_apu, sorted(data_num_set)


# Plotting function
def plot_data(data:dict, data_nums, temp_data_nums, data_name, datatype, miny, maxy, x_lable, Y_lable, cat):
    # 设置折线图
    fig, ax = plt.subplots(figsize=(4, 3.7))  # 控制整个图片大小
    color_list = ['#FF5733', '#33C1FF', '#28A745', '#FFC300', '#8A2BE2']
    type_list = ['^-', 'o-', 's-', 'd-', 'x-']

    # 绘制折线图
    for i, (k, v) in enumerate(data.items()):
        ax.plot(
            temp_data_nums, 
            [r for dn, r in sorted(v)], 
            type_list[i], 
            label=k, 
            color=color_list[i], 
            markersize=7, 
            markerfacecolor='none'
        )

    # 调整折线图区域大小
    ax.set_position([0.17, 0.15, 0.8, 0.75])  # [左边距, 底边距, 宽度, 高度]

    # 添加标签和标题
    ax.set_xlabel(x_lable, fontsize=16, fontdict={'family': 'Times New Roman'})
    ax.set_ylabel(Y_lable, fontsize=16, fontdict={'family': 'Times New Roman'})
    ax.set_xticks(temp_data_nums)
    ax.set_xticklabels(data_nums)
    ax.tick_params(axis='both', which='major', labelsize=12)
    y_ticks = np.linspace(miny, maxy, 6)  # 均匀生成 6 个刻度
    ax.set_yticks(y_ticks)  # 强制 y 轴显示这 6 个刻度
    

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)

    # 设置 y 轴范围和刻度
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None, integer=True))
    # 添加图例（可以根据需要放置在外部）
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=True)

    # 调整整体布局，避免重叠
    # fig.tight_layout(rect=[0, 0, 1, 1])

    # 保存折线图
    output_dir = f'./output/{data_name}/plt/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{datatype}_{data_name}_{cat}.pdf')
    plt.close()
    
