import os 
import json

ar_root_dir = "../action_effect_image_rs"
output = []

for ae in os.listdir(ar_root_dir):
    a, e = ae.split("+")
    uniq_id = sum([ord(c) for c in ae])
    output.append({uniq_id: [a, e]})

with open("ae_list.json", 'w') as j_ptr:
    json.dump(output, j_ptr)

