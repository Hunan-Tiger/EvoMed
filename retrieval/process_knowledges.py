# knowledges.json schema:
#[ 
#{
#     "id": int,                   # 唯一ID
#     "keywords": List[str],           # 关键词列表
#     "knowledge": str,                # 知识内容
#     "source": str = None,        # (可选) 来源
#     "timestamp": str = None,     # (可选) 时间
# },
# ...
# ]

# response:
# ```json
# [
#     {{  "option": "modify",
#         "knowledge": "the modified knowledge content",
#         "knowledge_keywords": "the key words of the modified knowledge",
#         "modified_from": "17" # specify the ID of knowledge that is modified
#     }},
#     {{  "option": "add",
#         "knowledge": "the added knowledge content",
#         "knowledge_keywords": "the key words of the added knowledge"
#     }},
#     {{  "option": "delete",
#         "delete_id": "the deleted knowledge ID" # must not overlap with 'modified_from' IDs above
#     }}, ...
# ]
# ```

from utils.utils import get_cur_time, parse_id, parse_keywords
import re
import logging
import json
from utils.biomedclip import BioMedCLIPEncoder, keywords2vector
import os
import pickle



def update_knowledges_and_to_dict(response, current_knowledges, source='qwen-vl-plus'):
    """
    将模型返回的知识更新结果解析为字典格式
    
    Args:
        response: 模型返回的字符串响应
        current_knowledges: 当前的知识列表，包含每条知识的ID等信息(字典)
        
    Returns:
        List[dict]: 解析后的知识更新结果列表
    """
    # 确保 current_knowledges 是列表
    if current_knowledges is None:
        current_knowledges = []

    try:
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            knowledge_updates = json.loads(json_match.group(1))
        else:
            knowledge_updates = json.loads(response)
        
        # 1. 确保解析出来的是列表
        if not isinstance(knowledge_updates, list):
            logging.warning("  ###  Parsed JSON is not a list, skipping update !!!")
            return current_knowledges

        # 2. 分类操作
        deletes = []
        modifies = []
        adds = []

        for u in knowledge_updates:
            opt = u.get('option', '').lower()
            if opt == 'delete':
                deletes.append(u)
            elif opt == 'modify':
                modifies.append(u)
            elif opt == 'add':
                adds.append(u)

        # 3. 处理删除 (Delete)
        # 这里的逻辑没问题，因为是根据 ID 查找对象并 remove，不依赖列表索引顺序
        for update in deletes:
            raw_id = update.get('delete_id')
            delete_id = parse_id(raw_id)
            
            if delete_id is not None:
                # 找到要删除的知识条目
                knowledge_to_delete = next((k for k in current_knowledges if k['id'] == delete_id), None)
                if knowledge_to_delete:
                    current_knowledges.remove(knowledge_to_delete)
                    logging.info(f"Deleted knowledge ID: {delete_id} - {knowledge_to_delete['knowledge']}")
                else:
                    logging.warning(f"Delete failed: ID {delete_id} not found.")


        # 4. 处理修改 (Modify)
        for update in modifies:
            raw_id = update.get('modified_from')
            modified_from_id = parse_id(raw_id)
            
            if modified_from_id is not None:
                knowledge_to_modify = next((k for k in current_knowledges if k['id'] == modified_from_id), None)
                if knowledge_to_modify:
                    old_knowledge = knowledge_to_modify['knowledge']
                    knowledge_to_modify['knowledge'] = update.get('knowledge', knowledge_to_modify['knowledge'])
                    # 使用增强的关键词解析
                    if 'knowledge_keywords' in update: # 兼容 prompt 中的 key 名字
                        knowledge_to_modify['keywords'] = parse_keywords(update['knowledge_keywords'])
                    elif 'knowledge_key_words' in update: # 兼容 prompt 修正后的 key 名字
                        knowledge_to_modify['keywords'] = parse_keywords(update['knowledge_key_words'])
                        
                    knowledge_to_modify['timestamp'] = get_cur_time()
                    knowledge_to_modify['source'] = source
                    logging.info(f"Modified knowledge ID: {modified_from_id} from: {old_knowledge} to: {knowledge_to_modify['knowledge'][:50]}...")
                else:
                    # 如果修改的ID不存在，则我们将其作为新知识加入知识库中
                    current_max_id = max((k['id'] for k in current_knowledges), default=0)
                    # 提取关键词，兼容两种 key 写法
                    keywords_raw = update.get('knowledge_keywords') or update.get('knowledge_key_words', "")
                    new_knowledge = {
                        'id': current_max_id + 1,
                        'knowledge': update.get('knowledge', ""),
                        'keywords': parse_keywords(keywords_raw),
                        'timestamp': get_cur_time(),
                        'source': source
                    }
                    current_knowledges.append(new_knowledge)
                    logging.info(f"Modify target ID {modified_from_id} not found. Added as new knowledge ID: {new_knowledge['id']} - {new_knowledge['knowledge']}")
            else:
                # 如果修改的ID不存在，则我们将其作为新知识加入知识库中
                current_max_id = max((k['id'] for k in current_knowledges), default=0)
                # 提取关键词，兼容两种 key 写法
                keywords_raw = update.get('knowledge_keywords') or update.get('knowledge_key_words', "")
                new_knowledge = {
                    'id': current_max_id + 1,
                    'knowledge': update.get('knowledge', ""),
                    'keywords': parse_keywords(keywords_raw),
                    'timestamp': get_cur_time(),
                    'source': source
                }
                current_knowledges.append(new_knowledge)
                logging.info(f"Modify target ID {modified_from_id} not found. Added as new knowledge ID: {new_knowledge['id']} - {new_knowledge['knowledge']}")


        # 5. 处理添加 (Add)
        for update in adds:
            # 计算新 ID：如果列表为空则为1，否则为最大ID+1
            # 注意：这里必须实时计算 max，因为前面的循环可能已经添加了元素
            current_max_id = max((k['id'] for k in current_knowledges), default=0)
            
            # 兼容不同的 key 名字
            keywords_raw = update.get('knowledge_keywords') or update.get('knowledge_key_words', "")
            
            new_knowledge = {
                'id': current_max_id + 1, # int
                'knowledge': update.get('knowledge', ""), # str
                'keywords': parse_keywords(keywords_raw), # List[str]
                'timestamp': get_cur_time(), # str
                'source': source # str
            }
            current_knowledges.append(new_knowledge)
            logging.info(f"Added new knowledge, assigned ID: {new_knowledge['id']} - {new_knowledge['knowledge']}")
        
        # 6. 重排 ID (Re-index)
        # 这一步是可选的。如果你希望 ID 永远连续（1,2,3...），保留此步。
        # 如果你希望 ID 永久固定（删除后 ID 空缺），则删除此步。
        # 通常在迭代优化的场景下，重排是比较干净的做法。
        for idx, knowledge in enumerate(current_knowledges):
            knowledge['id'] = idx + 1
        logging.info("Re-indexed knowledge IDs.")


    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logging.error(f"Failed to parse knowledge updates: {e}")
        logging.error(f"Response content: {response}")
    
    return current_knowledges


def save_keywords2vectors(knowledges, filepath = f'knowledges/expname_keyword_vectors.pkl', model_instance: BioMedCLIPEncoder = None):
    """
    计算并保存知识关键词的特征向量到文件
    
    Args:
        knowledges: 知识列表，包含每条知识的关键词
        filepath: 保存特征向量的文件路径
    """
    if knowledges is None or len(knowledges) == 0:
        logging.warning("No knowledges provided to save keywords vectors.")
        return

    keywords_list = [k['keywords'] for k in knowledges]
    knowledge_list = [k['knowledge'] for k in knowledges]
    idx_list = [k['id'] for k in knowledges]
    vectors = keywords2vector(keywords_list, model_instance)
    
    # 保存为字典，然后存储
    keywords_vectors = []
    for idx, knowledge, keywords, vector in zip(idx_list, knowledge_list, keywords_list, vectors):
        d = {
            'id': idx, # int
            'knowledge': knowledge, # str
            'keywords': keywords, # List[str]
            'vector': vector
        }
        keywords_vectors.append(d)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(keywords_vectors, f)

    logging.info(f"Saved knowledge keywords vectors to {filepath}")


    