from datasets import load_dataset, Dataset, load_from_disk
import numpy as np
import os
import random
import logging
from PIL import Image
from utils.rewards import extract_hash_answer


# # GRPO training
# def get_gsm8k_questions(split = "train", grpo = True) -> Dataset:
#     GRPO_SYSTEM_PROMPT = r"""
# 1. Output the thinking process in <thinking> </thinking> and final answer in <answer> </answer> tags.
# 2. The output format should be as follows:
# <thinking>
# ...
# </thinking>
# <answer>
# ...
# </answer>
# 3. No extra information or text outside of these tags.
# """

#     SFT_SYSTEM_PROMPT = r"""
# 1. Output the final answer in <answer> </answer> tag, no extra information or text outside of the tag.
# 2. The output format should be as follows:
# <answer>
# ...
# </answer>
# """

#     data = load_dataset('/public_bme2/bme-wangqian2/lijm2024/VLM/checkpoints/datasets/openai/gsm8k', 'main')[split] # type: ignore
#     print(data[0])
#     dataset = []
#     for i, item in enumerate(data):
#         question = item["question"]
#         answer = extract_hash_answer(item["answer"])
#         dataset.append({
#             'prompt': [
#                 {'role': 'system', 'content': [{'type': 'text', 'text': GRPO_SYSTEM_PROMPT if grpo else SFT_SYSTEM_PROMPT}]},
#                 {'role': 'user', 'content': [{'type': 'text', 'text': question}]},
#             ],
#             'image': None,
#             'answer': answer
#         })
#     return dataset 



# def get_llava_med_zh_instruct_60k(split = 'train[0:2000]'):
#     SYSTEM_PROMPT = r'''
# # Below is an instruction that describes a task, paired with an input that provides further context.
# # Write a response that appropriately completes the request.
# # Before answering, think carefully about the question and create a step-by-step chain of 
# # thoughts to ensure a logical and accurate response.

# ### Instruction:
# The user asks a question about medicine, and you solve it. 
# You first think about the thinking process in the mind and then provide the user with the answer.
# Output the thinking process in <thinking> </thinking> and final answer in <answer> </answer> tags.
# The output format should be as follows:
# <thinking>
# ...
# </thinking>
# <answer>
# ...
# </answer>
# '''   # for a different language, please change the last few words.

#     data = load_dataset('checkpoints/datasets/BUAADreamer/llava-med-zh-instruct-60k', split=split, trust_remote_code=True) # type: ignore
#     print(data[0])
#     dataset = []
#     for i, item in enumerate(data):
#         dialogue_num = len(item['messages'])
#         n = 0
#         while n<dialogue_num:
#             assert item['messages'][n]['role']=='user' and item['messages'][n+1]['role']=='assistant'
#             question_sample = item['messages'][n]['content']
#             answer_sample = item['messages'][n+1]['content']
#             img_pil = item['images'][0].resize((128,128))  # reduce vRAM burden
#             dataset.append({
#                     'prompt': [
#                         {'role': 'system', 'content': [{"type": "text", "text": SYSTEM_PROMPT}]},
#                         {'role': 'user', 'content': [
#                             {"type": "image", },  
#                             {"type": "text", "text": question_sample},    
#                         ]}
#                     ],
#                     'image':img_pil,
#                     'answer':answer_sample,
#                 })
#             n+=2
#     return dataset

# def get_medo1(split='train'):
#     instruction = r'''Your task:
# 1. Output the reasoning process in <think>...</think> and final answer in <answer>...</answer> tags.
# 2. The output format should be as follows:
# <think>
# ...
# </think>
# <answer>
# ...
# </answer>
# 3. No extra information or text outside of these tags.
# '''

#     data = load_dataset('checkpoints/datasets/FreedomIntelligence/medical-o1-reasoning-SFT', 'en')[f'{split}']
#     print(data[0])

#     dataset = []
#     for i, item in enumerate(data):
#         conversation = [
#         { "role": 'system',
#           "content": [
#              {'type': 'text','text': instruction}
#          ]
#         },
#         { "role": "user",
#           "content" : [
#             {"type" : "text",  "text": item['Question']},
#             {"type" : "image"}]
#         },
#         { "role" : "assistant",
#           "content" : [
#             {"type" : "text",  "text": '<think>\n'+ item['Complex_CoT'] + '\n</think>\n'},
#             {"type" : "text",  "text": '<answer>\n'+ item['Response'] + '\n</answer>\n'}
#             ]
#         },
#     ]
#         dataset.append(conversation)
#     return dataset



# def get_medminist(root_path='datasets/MedMNIST', split='train', length=-1, special=None, data_limit=1000, thinktype='nothink'):
#     PROMPT_THINK = r'''Your task:
#     1. Think through the question step by step in no more than 3 sentences, enclose your reasoning process in <think>...</think> tags.
#     2. Then answer the correct single-letter choice (A, B, C, D,...) inside <answer>...</answer> tags.'''

#     PROMPT_NOTHINK = r'''
#     Answer directly with the option's letter from the given choices (A, B, C, D,...)'''

#     dataset = []
#     data_folder = os.path.join(root_path, split)
#     if special is None:
#         data_paths = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, file))]
#     else:
#         data_paths = [os.path.join(data_folder, f"{special}.npy")]

#     for path in data_paths:
#         data = np.load(path, allow_pickle=True)
#         count = 0
#         for row in data:
#             if count >= data_limit and data_limit > 0:
#                 break
#             img_pil = row["image"]
#             dataset.append({
#                 "prompt": [
#                     {"role": "user", "content": [
#                         {"type": "image", "image": img_pil},
#                         {"type": "text", "text": row["text"]},
#                         {"type": "text", "text": PROMPT_THINK if thinktype=='think' else PROMPT_NOTHINK}
#                     ]}
#                 ],
#                 "image": img_pil,
#                 "answer": row["answer"]
#             })
#             count += 1
#     # if split == 'train':
#     #     random.shuffle(dataset)
#     if length > 0 and length < len(dataset):
#         dataset = dataset[:length]

#     return dataset




# ====================== Dataset Processing ====================
def get_vqarad(truncate=None, split='train'):

    data = load_dataset(f'datasets/vqa-rad', split=split, trust_remote_code=True)

    logging.info(f"Dataset loaded from datasets/vqa-rad, split={split}")
    logging.info(f"Total samples in dataset: {len(data)}")
    logging.info(f"data[0]: {data[0]}")
    dataset = []
    for i, item in enumerate(data):
        question_sample = item['question']
        if question_sample[-1] != '?':
            question_sample += '?'

        answer_sample = item['answer']
        if answer_sample.lower() not in ['yes', 'no']:
            continue
        img_pil = item['image']# .resize((512, 512)) 
        img_path = item.get('img_path', f"image_{i}.png")

        dataset.append({
                'prompt': [
                        {'role': 'user', 'content': [
                            {"type": "image", "image": img_pil},
                            {"type": "text", "text": question_sample.strip()},
                        ]}
                    ],
                    'img_id': img_path,
                    'qid': item.get('qid', i),
                    'answer':answer_sample.strip(),
                })

    # if split == 'train':
    #     random.shuffle(dataset)

    logging.info(f"Filtered dataset size after processing: {len(dataset)}")
    if truncate is not None and truncate > 0 and truncate < len(dataset):
        dataset = dataset[:truncate]
        
    logging.info(f"Dataset truncated to first {len(dataset)} samples")
    return dataset



def get_slake(truncate=None, split='train', answer_type='closed', content_type='all', seed=42, preserve_distribution=False):
    random.seed(seed)

    data = load_from_disk("datasets/slake")[split]
    logging.info(f"Dataset loaded from datasets/slake, split={split}")
    logging.info(f"Total samples in dataset: {len(data)}")
    logging.info(f"data[0]: {data[0]}")

    img_root = 'datasets/slake/imgs'
    
    # 处理content_type参数
    if isinstance(content_type, str) and content_type != 'all':
        content_type_list = [ct.strip().lower() for ct in content_type.split(',')]
    else:  # 'all'
        content_type_list = None
    
    dataset = []   
    for i, item in enumerate(data):
        # 检查answer_type
        if answer_type != 'all' and item['answer_type'].lower() != answer_type:
            continue
        
        # 检查content_type
        if content_type_list is not None and item['content_type'].lower() not in content_type_list:
            continue
        
        # 检查语言
        if item['q_lang'].lower() != 'en':
            continue
        
        question_sample = item['question']
        if question_sample[-1] != '?':
            question_sample += '?'

        answer_sample = item['answer']
        img_path = os.path.join(img_root, item['img_name'])
        if not os.path.exists(img_path):
            logging.warning(f"Image not found: {img_path}, skipping sample.")
            continue
        
        try:
            with Image.open(img_path) as img:
                img_pil = img.copy()
        except Exception as e:
            print(f"Error opening or copying image {img_path}: {e}")

        dataset.append({
                'prompt': [
                    {'role': 'user', 'content': [
                        {"type": "image", "image": img_pil},
                        {"type": "text", "text": question_sample.strip()},
                    ]}
                ],
                'img_id': img_path,
                'qid': item.get('qid', i),
                'answer': answer_sample.strip(),
                'answer_type': item['answer_type'].lower(),
                'content_type': item['content_type'].lower(),
            })

    logging.info(f"Filtered dataset size after processing: {len(dataset)}")

   # 如果需要，截断数据集
    if truncate is not None and truncate > 0 and truncate < len(dataset):
        if preserve_distribution:
            logging.info(f"Truncating to {truncate} samples while preserving content_type distribution.")
            
            # 1. 按 content_type 分组
            content_groups = {}
            for item in dataset:
                ct = item['content_type']
                if ct not in content_groups:
                    content_groups[ct] = []
                content_groups[ct].append(item)
            
            # 2. 根据原始分布计算每个组应抽取的样本数
            final_dataset = []
            total_original_count = len(dataset)
            for ct, items in content_groups.items():
                group_proportion = len(items) / total_original_count
                num_to_sample = round(truncate * group_proportion)
                
                random.shuffle(items)
                sampled_items = items[:num_to_sample]
                final_dataset.extend(sampled_items)
                logging.info(f"Content type '{ct}': selected {len(sampled_items)} samples out of {len(items)} based on distribution.")
            
            dataset = final_dataset
            # 最终数据集可能由于四舍五入不完全等于 truncate 值，再次随机打乱
            random.shuffle(dataset)

        else:
            # 原始的简单随机截断
            logging.info(f"Truncating to {truncate} samples with simple random sampling.")
            random.shuffle(dataset)
            dataset = dataset[:truncate]

    # 统计最终数据集组成
    open_count = sum(1 for d in dataset if d['answer_type'] == 'open')
    closed_count = sum(1 for d in dataset if d['answer_type'] == 'closed')
    content_dist = {}
    for d in dataset:
        ct = d['content_type']
        content_dist[ct] = content_dist.get(ct, 0) + 1
    
    logging.info(f"Final dataset size: {len(dataset)} samples")
    logging.info(f"Answer type distribution: {open_count} open, {closed_count} closed")
    logging.info(f"Content type distribution: {content_dist}")
    if split == 'test':
        dataset = [dataset[2]]
    return dataset


def get_omni_rad_closed(truncate=None, split='train', seed=42, **kwargs):
    '''
    Data format:
    {
    "dataset": "Chest CT Scan",
    "question_id": "Chest CT Scan_0116",
    "question_type": "Disease Diagnosis",
    "question": "What is the specific type and stage of cancer seen in this image?",
    "gt_answer": "Adenocarcinoma of the left lower lobe, T2 N0 M0, Stage Ib",
    "image_path": "OmniMedVQA-Radiology-Closed/Images/000114.png",
    "option_A": "Neuroendocrine tumor of the left upper lobe, T3 N0 M1, Stage III",
    "option_B": "Small cell lung carcinoma of the right middle lobe, T1 N0 M0, Stage Ia",
    "option_C": "Adenocarcinoma of the left lower lobe, T2 N0 M0, Stage Ib",
    "option_D": "Non-Hodgkin lymphoma of the right lower lobe, T2 N1 M0, Stage II",
    "modality_type": "CT(Computed Tomography)",
    "_source_json": "Chest CT Scan.json",
    "closed_type": "MCQ"
  },
    '''
    random.seed(seed)
    
    import json
    if split == 'test':
        with open(r'datasets/OmniMedVQA-Radiology-Closed/radiology_closedset_test.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(r'datasets/OmniMedVQA-Radiology-Closed/radiology_closedset_train.json', 'r', encoding='utf-8') as f:
            data = json.load(f)

    print(data[0])
    dataset = []
    for i, item in enumerate(data):
        question_sample = item['question']
        if question_sample[-1] != '?':
            question_sample += '?'

        answer_sample = item['gt_answer']
        option_map = {
            item.get('option_A'): 'A' if item.get('option_A') else None,
            item.get('option_B'): 'B' if item.get('option_B') else None,
            item.get('option_C'): 'C' if item.get('option_C') else None,
            item.get('option_D'): 'D' if item.get('option_D') else None,
        }
        # 构建选项字符串，只包含存在的选项
        options_parts = []
        if item.get('option_A'):
            options_parts.append(f"A. {item['option_A']}\n")
        if item.get('option_B'):
            options_parts.append(f"B. {item['option_B']}\n")
        if item.get('option_C'):
            options_parts.append(f"C. {item['option_C']}\n")
        if item.get('option_D'):
            options_parts.append(f"D. {item['option_D']}\n")

        question_sample += f" Options:\n{''.join(options_parts)}Please answer with the correct single-letter choice."

        if answer_sample not in option_map:
            continue
        answer_letter = option_map[answer_sample]
        answer = answer_letter+'. '+answer_sample

        img_path = os.path.join('datasets', item['image_path'])
        try:
            with Image.open(img_path) as img:
                img_pil = img.copy()
        except Exception as e:
            print(f"Error opening or copying image {img_path}: {e}")

        dataset.append({
                'prompt': [
                        {'role': 'user', 'content': [
                            {"type": "image", "image": img_pil},
                            {"type": "text", "text": question_sample.strip()},
                        ]}
                    ],
                    'img_id': img_path,
                    'qid': item.get('question_id', i),
                    'answer':answer,
                    'content_type': item['question_type'].lower(),
                })
    logging.info(f"Filtered dataset size after processing: {len(dataset)}")
    random.shuffle(dataset)
    if truncate is not None and truncate > 0 and truncate < len(dataset):
        dataset = dataset[:truncate]
        logging.info(f"Dataset truncated to first {len(dataset)} samples")
    return dataset
    