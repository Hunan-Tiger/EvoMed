
from utils.biomedclip import keywords2vector, BioMedCLIPEncoder
# 加载关键词向量库pkl和知识库json
import pickle
import os
import json

# 1. 设置文件路径 (请修改为你实际的文件名)
vectors_file_path = r'knowledges\merged_keyword_vectors.pkl'
knowledge_json_path = r'knowledges\merged_knowledges.json'

if not os.path.exists(vectors_file_path):
    print(f"错误: 向量文件不存在 -> {vectors_file_path}")
elif not os.path.exists(knowledge_json_path):
    print(f"错误: 知识文件不存在 -> {knowledge_json_path}")
else:
    # 1. 读取知识 JSON 文件
    print(f"正在读取知识文件: {knowledge_json_path} ...")
    with open(knowledge_json_path, 'r', encoding='utf-8') as f:
        knowledges = json.load(f)

    model_instance = BioMedCLIPEncoder()
    # 2. 根据知识数据生成关键词向量
    keywords_list = [k['keywords'] for k in knowledges]
    knowledge_list = [k['knowledge'] for k in knowledges]
    idx_list = [k['id'] for k in knowledges]
    vectors = keywords2vector(keywords_list, model_instance)

    # 3. 保存为字典，然后存储
    keywords_vectors = []
    for idx, knowledge, keywords, vector in zip(idx_list, knowledge_list, keywords_list, vectors):
        d = {
            'id': idx, # int
            'knowledge': knowledge, # str
            'keywords': keywords, # List[str]
            'vector': vector
        }
        keywords_vectors.append(d)
    # 4. 保存为新的 PKL 文件
    with open(vectors_file_path, 'wb') as f:
        pickle.dump(keywords_vectors, f)
    print(f"已保存关键词向量至: {vectors_file_path}")

