import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools


"""
input = np.array([0, 1, 2, 3, 4])
np.save("model_x_data", input)

x_save_load = np.load("model_x_data.npy")
print(x_save_load)
"""

item_list = ["item_1", "item_2", "item_3", "item_4", "item_5"]
value_list = [1, 6, 18, 22, 28]
weight_list = [1, 2, 5 ,6, 7]
actions = list(range(0, len(item_list), 1))

items = pd.DataFrame({"VALUE":value_list, "WEIGHT":weight_list}, index=actions)

print(items)
print(actions)
print(type(actions[0]))
