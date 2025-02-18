import json

import numpy as np
import torch.version
from preprocess_level import print_from_mapping
import torch

json_file="zelda_with_walls.json"

with open(json_file) as f:
    data = json.load(f)

data = np.array(data)

print(data.shape)

print_from_mapping(data[0])

print(np.unique(data))

print(torch.cuda.is_available())