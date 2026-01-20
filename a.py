# import pickle
# import os
# import numpy as np # 如果pkl里包含numpy数组，通常需要导入

# # 1. 设置文件路径 (请修改为你实际的文件名)
# # 例如: 'knowledges/vqa-rad_keyword_vectors.pkl' 或 'checkpoints/xxx.pkl'
# # file_path = r'knowledges\merged_slake_vqarad_keyword_vectors.pkl' 

# # if not os.path.exists(file_path):
# #     print(f"错误: 文件不存在 -> {file_path}")
# # else:
# #     # 2. 读取 PKL 文件 (注意使用 'rb' 二进制读取)
# #     print(f"正在读取: {file_path} ...")
# #     with open(file_path, 'rb') as f:
#         # data = pickle.load(f)
#     # 示例 B: 如果是知识库向量，想要修改某个值
#     # count = 0
#     # for item in data:
#     #     if ']. ' in item['knowledge']:
#     #         # 直接修改 item，这会反映到 data 列表中
#     #         original_text = item['knowledge']
#     #         new_text = original_text.split(']. ')[-1].strip()
            
#     #         item['knowledge'] = new_text
#     #         count += 1
#     #         # print(f"修改 ID {item['id']}: {original_text[:20]}... -> {new_text[:20]}...")

#     # print(f"共修改了 {count} 条数据。")


#     # for item in data:
#     #     print(item['id'], item['keywords'], item['knowledge'], item['vector'].shape)
#     # ==========================================

#     # 5. 保存回文件 (注意使用 'wb' 二进制写入)
#     # # 建议先保存为新文件进行测试，确认无误后再覆盖原文件
#     # save_path = file_path  # 或者 'modified_data.pkl'
    
#     # with open(save_path, 'wb') as f:
#     #     pickle.dump(data, f)
    
#     # print(f"修改完成，已保存至: {save_path}")


# # import re
# # def extract_answer(response_text):
# #     # 查找 <answer> 标签内的内容，忽略大小写，允许跨行
# #     pattern = r"<answer>(.*?)</answer>"
# #     match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
    
# #     if match:
# #         return match.group(1).strip()
# #     else:
# #         return None

# # res = '''errge.
# # <answer>
# # The patient is likely suffering from pneumonia.
# # </answer>
# # '''
# # answer = extract_answer(res)
# # print("Extracted Answer:", answer)

# # =============================== slake数据集探索 ===============================
# from datasets import load_from_disk
# from PIL import Image
# dataset = load_from_disk("datasets/slake")
# # 获取训练集
# if 'train' in dataset:
#     train_data = dataset['train']
# else:
#     train_data = dataset

# # 1. 查看数据集大小
# print(f"数据集大小: {len(train_data)}")

# # 2. 查看字段名称
# print(f"字段: {train_data.column_names}")
# # ['img_id', 'img_name', 'question', 'answer', 'q_lang', 'location', 'modality', 'answer_type', 'base_type', 'content_type', 'triple', 'qid']
# # img_name是图片路径

# # 3. 访问单条数据
# sample = train_data[0]
# print(f"\n单条数据的类型: {type(sample)}")
# print(f"单条数据的键: {sample.keys()}")

# # print("\n前3条数据:")
# # for i, example in enumerate(train_data):
# #     if i >= 3:
# #         break
# #     print(f"\n样本 {i}:")
# #     for key, value in example.items():
# #         if key == 'image':
# #             print(f"  {key}: <PIL.Image>")
# #         else:
# #             print(f"  {key}: {value}")

# # 8. 过滤数据
# filtered_dataset = train_data.filter(lambda x: x['content_type']== 'Abnormality' and x['q_lang']=='en')
# for i in range(len(filtered_dataset)):
#     print(filtered_dataset[i])

# from utils.datasets import get_slake as process_dataset
# data = process_dataset(truncate=200, split='test', answer_type='all', content_type='abnormality', seed=42)
# # 统计data中answer_type和content_type的分布
# from collections import Counter
# answer_type_counter = Counter()
# content_type_counter = Counter()
# for item in data:
#     answer_type_counter[item['answer_type']] += 1
#     content_type_counter[item['content_type']] += 1
# print("Answer Type Distribution:", answer_type_counter)
# print("Content Type Distribution:", content_type_counter)



from utils.utils import parse_tag

res = '''<Answer>
Yes, this image is abnormal '''
b = parse_tag(res, 'Answer')
print(b)



# python training_free_grpo_v2.py --en omni-rad-closed --at closed --ct all --model qwen2.5-vl-instruct --em qwen2.5-vl-instruct --cm qwen2.5-vl-instruct --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start --ak skd --dt 1000
# python test.py --en slake --at closed --dt 500 --ct all --model qwen2.5-vl-instruct --em qwen2.5-vl-instruct --cm qwen2.5-vl-instruct --bu https://genaiapi.shanghaitech.edu.cn/api/v1/start --ak skd --gk true