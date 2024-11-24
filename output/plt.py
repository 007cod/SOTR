import matplotlib.pyplot as plt
import os
from plt_utils import parse_data, plot_data

if __name__ == '__main__':
    
    args04 = {
        "w":{
            "name":"Number of Workers",
            "id":3,
            "R":[500, 3000],
            "C":[0.2, 0.6],
            "T":[0,25],
            "apu":[0.5, 1],
        },
        "s":{
            "name":"Number of Tasks",
            "id":3,
            "R":[0, 4000],
            "C":[0.3, 0.55],
            "T":[0,10],
            "apu":[0.5, 1],
        },
        "r":{
            "name":"Reachable Distance of Workers",
            "id":3,
            "R":[0, 3000],
            "C":[0, 1],
            "T":[0, 15],
            "apu":[0.5, 1],
        },
        "wdead":{
            "name":"Available Time of Workers (h)",
            "id":3,
            "R":[0, 3000],
            "C":[0, 0.5],
            "T":[0, 10],
            "apu":[0.5, 1],
        },
        "sdead":{
            "name":"Vaild Time of Tasks",
            "id":3,
            "R":[0, 3000],
            "C":[0, 0.5],
            "T":[0, 10],
            "apu":[0.5, 1],
        },
        "k":{
            "name":"k",
            "id":3,
            "R":[0, 3000],
            "C":[0.3, 0.5],
            "T":[0, 10],
            "apu":[0.5, 1],
        }
    }
    
    args08 = {
        "w":{
            "name":"Number of Workers",
            "id":3,
            "R":[500, 3000],
            "C":[0.3, 0.8],
            "T":[0, 100],
            "apu":[0.5, 1],
        },
        "s":{
            "name":"Number of Tasks",
            "id":3,
            "R":[0, 4000],
            "C":[0.3, 0.8],
            "T":[0, 30],
            "apu":[0.5, 1],
        },
        "r":{
            "name":"Reachable Distance of Workers",
            "id":3,
            "R":[0, 3000],
            "C":[0, 1],
            "T":[0, 50],
            "apu":[0.5, 1],
        },
        "wdead":{
            "name":"Available Time of Workers (h)",
            "id":3,
            "R":[0, 3000],
            "C":[0, 0.8],
            "T":[0, 30],
            "apu":[0.5, 1],
        },
        "sdead":{
            "name":"Vaild Time of Tasks",
            "id":3,
            "R":[0, 3000],
            "C":[0, 0.8],
            "T":[0, 30],
            "apu":[0.5, 1],
        },
        "k":{
            "name":"k",
            "id":3,
            "R":[0, 3000],
            "C":[0.3, 0.8],
            "T":[0, 30],
            "apu":[0.5, 1],
        },
    }
    
    for k, v in args04.items():
        if not os.path.exists(f'./output/PEMS04/{k}.txt'):
            continue
        if k != "k":
            continue
        data, data_cost, data_time, data_apu, data_nums = parse_data(f'./output/PEMS04/{k}.txt', v["id"])
        plot_data(data, data_nums, list(range(len(data_nums))), "PEMS04", k, v["R"][0], v["R"][1], v["name"], 'Completed Task Number Expectation', 'R')
        plot_data(data_cost, data_nums, list(range(len(data_nums))), "PEMS04", k, v["C"][0], v["C"][1],v["name"], 'Average Emission Expectation', 'C')
        plot_data(data_time, data_nums, list(range(len(data_nums))), "PEMS04", k, v["T"][0], v["T"][1] ,v["name"], 'CPU Time (s)', 'T')
        plot_data(data_apu, data_nums, list(range(len(data_nums))), "PEMS04", k, v["apu"][0], v["apu"][1], v["name"], 'Average Preference-based Utility', 'apu')
        
    # for k, v in args08.items():
    #     if not os.path.exists(f'./output/PEMS08/{k}.txt'):
    #         continue
    #     # if k != "r":
    #     #     continue
    #     data, data_cost, data_time, data_apu, data_nums = parse_data(f'./output/PEMS08/{k}.txt', v["id"])
    #     plot_data(data, data_nums, list(range(len(data_nums))), "PEMS08", k, v["R"][0], v["R"][1], v["name"], 'Completed Task Number Expectation', 'R')
    #     plot_data(data_cost, data_nums, list(range(len(data_nums))), "PEMS08", k, v["C"][0], v["C"][1], v["name"], 'Average Emission Expectation', 'C')
    #     plot_data(data_time, data_nums, list(range(len(data_nums))), "PEMS08", k, v["T"][0], v["T"][1], v["name"], 'CPU Time (s)', 'T')
    #     plot_data(data_apu, data_nums, list(range(len(data_nums))), "PEMS08", k, v["apu"][0], v["apu"][1], v["name"], 'Average Preference-based Utility', 'apu')


    