# 将两个pkl文件和两个json文件 知识库合并，使用cat的方式，两个知识库ID之前会有重复，合并后使其无重复
import pickle
import os
import json
import numpy as np
from utils.utils import parse_id
import logging
from utils.biomedclip import keywords2vector

def merge_knowledge_bases(pkl_file_1, pkl_file_2, json_file_1, json_file_2, output_pkl, output_json):
    # 读取第一个PKL文件
    with open(pkl_file_1, 'rb') as f:
        knowledge_base_1 = pickle.load(f)

    # 读取第二个PKL文件
    with open(pkl_file_2, 'rb') as f:
        knowledge_base_2 = pickle.load(f)

    # 合并知识库，确保ID唯一
    existing_ids = set(item['id'] for item in knowledge_base_1)
    next_id = max(existing_ids) + 1 if existing_ids else 1

    for item in knowledge_base_2:
        item['id'] = next_id
        next_id += 1
        knowledge_base_1.append(item)

    # 保存合并后的PKL文件
    with open(output_pkl, 'wb') as f:
        pickle.dump(knowledge_base_1, f)
    logging.info(f"Merged PKL saved to {output_pkl}")

    # 读取第一个JSON文件
    with open(json_file_1, 'r', encoding='utf-8') as f:
        json_data_1 = json.load(f)

    # 读取第二个JSON file
    with open(json_file_2, 'r', encoding='utf-8') as f:
        json_data_2 = json.load(f)

    # 合并JSON数据，同时更新第二个JSON数据的ID以匹配PKL文件的更改
    # 注意：这里假设json_data_1和json_data_2的顺序与knowledge_base_1和knowledge_base_2完全对应
    # 重新计算起始ID，逻辑同PKL合并部分
    existing_json_ids = set(item['id'] for item in json_data_1)
    next_json_id = max(existing_json_ids) + 1 if existing_json_ids else 1

    for item in json_data_2:
        item['id'] = next_json_id
        next_json_id += 1
    
    merged_json_data = json_data_1 + json_data_2

    # 保存合并后的JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(merged_json_data, f, ensure_ascii=False, indent=4)
    logging.info(f"Merged JSON saved to {output_json}")

if __name__ == "__main__":
# 使用示例
    merge_knowledge_bases(
        pkl_file_1='knowledges/merged_slake_keyword_vectors.pkl',
        pkl_file_2='knowledges/vqa-rad_keyword_vectors copy.pkl',
        json_file_1='knowledges/merged_slake_knowledges.json',
        json_file_2='knowledges/vqa-rad_knowledges copy.json',
        output_pkl='knowledges/merged_slake_vqarad_keyword_vectors.pkl',
        output_json='knowledges/merged_slake_vqarad_knowledges.json'
    )