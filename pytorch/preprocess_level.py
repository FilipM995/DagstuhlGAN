

import json
import os
import numpy as np

data_folder = "pytorch\\zelda\\processed"
data_folder = os.path.join(os.getcwd(), data_folder)

save_folder = os.getcwd()

os.makedirs(save_folder, exist_ok=True)

window_size = (14, 9)

with_walls = True

levels_data = []

for file in os.listdir(data_folder):
    if file.endswith(".txt") and file != "README.txt":
        file = os.path.join(data_folder, file)
        with open(file, 'r') as file:
            level = file.readlines()
            level = [line.strip() for line in level]
            levels_data.append(level)

'''
W = WALL
- = VOID
F = FLOOR
B = BLOCK
M = MONSTER
P = ELEMENT (LAVA, WATER)
O = ELEMENT + FLOOR (LAVA/BLOCK, WATER/BLOCK)
I = ELEMENT + BLOCK
D = DOOR
S = STAIR
'''

mapping = {'W': 0, '-': 1, 'F': 2, 'B': 3, 'M': 4,
           'P': 5, 'O': 6, 'I': 7, 'D': 8, 'S': 9}
reverse_mapping = {v: k for k, v in mapping.items()}


def map_level(level):
    mapped_level = [[mapping[char] for char in line] for line in level]

    return np.array(mapped_level)[1:-1, 1:-1]


def print_from_mapping(mapped_level):
    orig_level = [[reverse_mapping[num] for num in row]
                  for row in mapped_level]

    for row in orig_level:
        print(''.join(row))


def slide_window(level, window_size, with_walls=False):
    window = []
    wall_mapping = mapping['W']
    door_mapping = mapping['D']
    empty_mapping = mapping['-']
    for i in range(0, level.shape[0] - window_size[0] + 1):
        for j in range(0, level.shape[1] - window_size[1] + 1):
            if level[i, j] == wall_mapping and level[i+1, j] == wall_mapping and level[i, j+1] == wall_mapping and level[i+1, j+1] not in [wall_mapping, door_mapping, empty_mapping]:
                window_without_walls = level[i+1:i +
                                             window_size[0]-1, j+1:j+window_size[1]-1]
                window_with_walls = level[i:i +
                                          window_size[0], j:j+window_size[1]]
                if not np.all(window_without_walls == window_without_walls[0, 0]):
                    window.append(
                        window_with_walls if with_walls else window_without_walls)
    return np.array(window)


mapped_levels = [map_level(level) for level in levels_data]

slided_levels = [slide_window(level, window_size, with_walls)
                 for level in mapped_levels]

complete_list = np.array(
    [window for level in slided_levels for window in level])

# print("SHAPE=", np.array(complete_list).shape)

if with_walls:
    save_name = "zelda_with_walls.json"
else:
    save_name = "zelda.json"

save_path = os.path.join(save_folder, save_name)

save_json = json.dumps(complete_list.tolist())

with open(save_path, 'w') as file:
    file.write(save_json)
