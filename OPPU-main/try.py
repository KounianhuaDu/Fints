from safetensors import safe_open
import os
from safetensors.torch import load_file, save_file

ori_path = "../OPPU-main/ckpt/news_headline"
lora_path = "ckpt/news_headline"
for dir in os.listdir(ori_path):
    if 'Meta' in dir:
        lora_file = os.path.join(ori_path, dir, 'adapter_model.safetensors')
        state_dict = load_file(lora_file)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace('self_attn', "block.self_attn.attn")
            print(k, new_k)
            new_state_dict[new_k] = v
        # break
        save_file(new_state_dict, os.path.join(lora_path, dir, 'adapter_model.safetensors'))