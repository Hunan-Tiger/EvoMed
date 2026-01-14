import os
import json
import pickle
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from utils.utils import get_cur_time


# ==================== 社区标签定义 (标签名 -> 数字ID) ====================

QUESTION_TYPE_TAGS = {
    # 异常检测类
    "normality": 0,
    # "lesion": 1,
    # "tumor": 2,
    # "inflammation": 3,
    # "fracture": 4,
    
    # # 形态学特征类
    "size": 1,
    "shape": 2,
    # "density": 7,
    # "texture": 8,
    # "margin": 9,
    
    # # 空间位置类
    "position": 3,
    # "laterality": 11,
    # "orientation": 12,
    # "distribution": 13,
    
    # # 图像属性类
    "modality": 4,
    "plane": 5,
    # "contrast": 16,
    "quality": 6,
    "color": 7,
    
    # # 诊断推理类
    "diagnosis": 8,
    # "differential": 19,
    # "prognosis": 20,
    # "staging": 21,
    
    # # 计数和存在性
    "quantity": 9,
    "presence": 10,
    # "comparison": 24,
    
    # # 常识和知识类
    "anatomy": 11,
    "organ":12,
    # "physiology": 26,
    "common_sense": 13,
    # "procedure": 28,
}

IMAGE_MODALITY_TAGS = {
    "xray": 0,
    "ct": 1,
    "mri": 2,
    "ultrasound": 3,
    "pet": 4,
    "endoscopy": 5,
    "fundoscopy": 6,
    "pathology": 7,
}

ORGAN_TISSUE_TAGS = {
    # 头颈
    "brain": 0,
    "head_neck": 1,
    "eye_ear": 2,
    "oral_nasal": 3,
    "thyroid": 4,

    # 胸部
    "lung": 5,
    "heart": 6,
    "mediastinum": 7,
    "breast": 8,

    # 腹部
    "liver": 9,
    "gallbladder": 10,
    "pancreas": 11,
    "spleen": 12,
    "kidney": 13,
    "stomach": 14,
    "intestine": 15,
    "abdomen": 16,
}


# 反向映射：ID -> 标签名
ID_TO_QUESTION_TYPE = {v: k for k, v in QUESTION_TYPE_TAGS.items()}
ID_TO_MODALITY = {v: k for k, v in IMAGE_MODALITY_TAGS.items()}
ID_TO_ORGAN = {v: k for k, v in ORGAN_TISSUE_TAGS.items()}


def make_triple_id(question_type_ids: List[int], modality_ids: List[int], organ_ids: List[int]) -> str:
    """
    生成三元ID字符串
    格式: "qt1,qt2-mod1-org1,org2"
    """
    qt_str = ",".join(str(i) for i in sorted(question_type_ids))
    mod_str = ",".join(str(i) for i in sorted(modality_ids))
    org_str = ",".join(str(i) for i in sorted(organ_ids))
    return f"{qt_str}-{mod_str}-{org_str}"


def parse_triple_id(triple_id: str) -> Tuple[List[int], List[int], List[int]]:
    """
    解析三元ID字符串
    返回: (question_type_ids, modality_ids, organ_ids)
    """
    parts = triple_id.split("-")
    if len(parts) != 3:
        return [], [], []
    
    qt_ids = [int(x) for x in parts[0].split(",") if x.strip().isdigit()]
    mod_ids = [int(x) for x in parts[1].split(",") if x.strip().isdigit()]
    org_ids = [int(x) for x in parts[2].split(",") if x.strip().isdigit()]
    
    return qt_ids, mod_ids, org_ids


def tags_to_ids(tags: List[str], tag_dict: Dict[str, int]) -> List[int]:
    """将标签名列表转换为ID列表"""
    ids = []
    for tag in tags:
        tag_lower = tag.lower().strip().replace(" ", "_")
        if tag_lower in tag_dict:
            ids.append(tag_dict[tag_lower])
    return ids


def ids_to_tags(ids: List[int], id_dict: Dict[int, str]) -> List[str]:
    """将ID列表转换为标签名列表"""
    return [id_dict[i] for i in ids if i in id_dict]


class TagKnowledgeBase:
    """
    基于三元ID的分层知识库
    
    结构:
    三元ID (如 "18-4-14") -> 知识条目列表
    
    数据结构:
    {
        "knowledge_items": [
            {
                "id": 1,
                "knowledge": "...",
                "triple_id": "18-4-14",
                "question_type_ids": [18],
                "modality_ids": [4],
                "organ_ids": [14],
                "timestamp": "...",
                "source": "..."
            }
        ],
        "triple_index": {
            "18-4-14": [1, 2, 3],
            "18,19-4-14,15": [4, 5],
            ...
        }
    }
    """
    
    def __init__(self, kb_path: str = "knowledges/tag_knowledge_base.pkl"):
        self.kb_path = kb_path
        self.knowledge_items: List[Dict] = []
        
        # 三元ID索引
        self.triple_index: Dict[str, List[int]] = defaultdict(list)
        
        # 单级索引（用于部分匹配）
        self.question_type_index: Dict[int, List[int]] = defaultdict(list)
        self.modality_index: Dict[int, List[int]] = defaultdict(list)
        self.organ_index: Dict[int, List[int]] = defaultdict(list)
        
        # 标签定义
        self.question_type_tags = QUESTION_TYPE_TAGS
        self.modality_tags = IMAGE_MODALITY_TAGS
        self.organ_tags = ORGAN_TISSUE_TAGS
        
        self._load()
    
    def _load(self):
        """加载知识库"""
        if os.path.exists(self.kb_path):
            try:
                with open(self.kb_path, 'rb') as f:
                    data = pickle.load(f)
                self.knowledge_items = data.get('knowledge_items', [])
                self.triple_index = defaultdict(list, data.get('triple_index', {}))
                self.question_type_index = defaultdict(list, data.get('question_type_index', {}))
                self.modality_index = defaultdict(list, data.get('modality_index', {}))
                self.organ_index = defaultdict(list, data.get('organ_index', {}))
                logging.info(f"Loaded Tag KB from {self.kb_path}")
                logging.info(f"  - Knowledge Items: {len(self.knowledge_items)}")
                logging.info(f"  - Triple IDs: {len(self.triple_index)}")
            except Exception as e:
                logging.error(f"Failed to load KB: {e}")
                # 尝试备份文件
                backup_path = self.kb_path + '.backup'
                try:
                    import shutil
                    shutil.copy2(self.kb_path, backup_path)
                    logging.info(f"Corrupted KB backed up to {backup_path}")
                except Exception as backup_error:
                    logging.error(f"Failed to backup corrupted KB: {backup_error}")
                
                # 尝试从JSON恢复
                json_path = self.kb_path.replace('.pkl', '.json')
                if os.path.exists(json_path):
                    try:
                        logging.info(f"Attempting to recover from JSON: {json_path}")
                        self._load_from_json(json_path)
                        # 重新保存为pkl
                        self.save()
                        logging.info("Successfully recovered from JSON backup")
                        return
                    except Exception as json_error:
                        logging.error(f"Failed to recover from JSON: {json_error}")
                
                # 初始化空知识库
                logging.warning("Initializing empty KB")
                self._init_empty()
        else:
            logging.info(f"KB file not found, initializing empty KB.")
            self._init_empty()

    def _load_from_json(self, json_path: str):
        """从JSON文件加载知识库"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建知识库
        self._init_empty()
        
        for item in data.get('knowledge_items', []):
            # 从可读格式转换回内部格式
            question_type_ids = tags_to_ids(item.get('question_types', []), self.question_type_tags)
            modality_ids = tags_to_ids(item.get('modalities', []), self.modality_tags)
            organ_ids = tags_to_ids(item.get('organs', []), self.organ_tags)
            
            self.add_knowledge(
                knowledge_text=item['knowledge'],
                question_type_ids=question_type_ids,
                modality_ids=modality_ids,
                organ_ids=organ_ids,
                source=item.get('source', 'recovered_from_json')
            )
        
        logging.info(f"Loaded {len(self.knowledge_items)} items from JSON")
    
    def _init_empty(self):
        """初始化空知识库"""
        self.knowledge_items = []
        self.triple_index = defaultdict(list)
        self.question_type_index = defaultdict(list)
        self.modality_index = defaultdict(list)
        self.organ_index = defaultdict(list)
    
    def save(self, kb_path: str = None):
        if kb_path is None:
            kb_path = self.kb_path
        """保存知识库"""
        os.makedirs(os.path.dirname(kb_path) if os.path.dirname(kb_path) else '.', exist_ok=True)
        data = {
            'knowledge_items': self.knowledge_items,
            'triple_index': dict(self.triple_index),
            'question_type_index': dict(self.question_type_index),
            'modality_index': dict(self.modality_index),
            'organ_index': dict(self.organ_index)
        }
        with open(kb_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Saved Tag KB to {kb_path}")
    
    def save_to_json(self, json_path: str = None):
        """保存知识库到JSON（用于可读性检查）"""
        if json_path is None:
            json_path = self.kb_path.replace('.pkl', '.json')
        
        # 转换为可读格式
        readable_items = []
        for item in self.knowledge_items:
            readable_item = {
                'id': item['id'],
                'knowledge': item['knowledge'],
                'triple_id': item['triple_id'],
                'question_types': ids_to_tags(item['question_type_ids'], ID_TO_QUESTION_TYPE),
                'modalities': ids_to_tags(item['modality_ids'], ID_TO_MODALITY),
                'organs': ids_to_tags(item['organ_ids'], ID_TO_ORGAN),
                'timestamp': item.get('timestamp', ''),
                'source': item.get('source', '')
            }
            readable_items.append(readable_item)
        
        data = {
            'statistics': {
                'total_knowledge': len(self.knowledge_items),
                'unique_triple_ids': len(self.triple_index),
                'question_types_used': len(self.question_type_index),
                'modalities_used': len(self.modality_index),
                'organs_used': len(self.organ_index)
            },
            'tag_definitions': {
                'question_types': {k: v for k, v in self.question_type_tags.items()},
                'modalities': {k: v for k, v in self.modality_tags.items()},
                'organs': {k: v for k, v in self.organ_tags.items()}
            },
            'knowledge_items': readable_items,
            'triple_index': dict(self.triple_index)
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved KB to JSON: {json_path}")
    
    # ==================== 知识检索 ====================
    
    def retrieve_knowledge(
        self,
        question_type_ids: List[int],
        modality_ids: List[int],
        organ_ids: List[int],
    ) -> List[Dict]:
        """
        根据三元ID检索知识
        
        检索策略：
        1. 精确匹配三元ID
        2. 如果不足，尝试部分匹配（优先级：三个都匹配 > 两个匹配 > 一个匹配）
        3. 如果仍不足，返回当前结果（可能为空）
        
        Args:
            question_type_ids: 问题类型ID列表
            modality_ids: 图像模态ID列表
            organ_ids: 器官组织ID列表
            top_n: 返回的知识数量
            
        Returns:
            List[Dict]: 知识条目列表，如果没有匹配则返回空列表
        """
        if not question_type_ids and not modality_ids and not organ_ids:
            return []
        
        # 1. 精确匹配三元ID
        triple_id = make_triple_id(question_type_ids, modality_ids, organ_ids)
        exact_match_ids = self.triple_index.get(triple_id, [])
        
        if len(exact_match_ids)>0:
            return self._get_knowledge_by_ids(exact_match_ids[:])
        
        # 2. 获取各索引的知识ID集合
        qt_ids_set = set()
        for qt_id in question_type_ids:
            qt_ids_set.update(self.question_type_index.get(qt_id, []))
        
        mod_ids_set = set()
        for mod_id in modality_ids:
            mod_ids_set.update(self.modality_index.get(mod_id, []))
        
        org_ids_set = set()
        for org_id in organ_ids:
            org_ids_set.update(self.organ_index.get(org_id, []))
        
        # 3. 三级全匹配
        three_match = set()
        if qt_ids_set and mod_ids_set and org_ids_set:
            three_match = qt_ids_set & mod_ids_set & org_ids_set
        
        combined = list(set(exact_match_ids) | three_match)
        return self._get_knowledge_by_ids(combined[:])
        
        
        # # 4. 两级匹配
        # two_match = set()
        # if qt_ids_set and mod_ids_set:
        #     two_match.update(qt_ids_set & mod_ids_set)
        # if qt_ids_set and org_ids_set:
        #     two_match.update(qt_ids_set & org_ids_set)
        # if mod_ids_set and org_ids_set:
        #     two_match.update(mod_ids_set & org_ids_set)
        # two_match -= set(combined)
        
        # combined.extend(list(two_match))
        # if len(combined) >= top_n:
        #     return self._get_knowledge_by_ids(combined[:top_n])
        
        # # 5. 单级匹配
        # one_match = qt_ids_set | mod_ids_set | org_ids_set
        # one_match -= set(combined)
        
        # combined.extend(list(one_match))
        
        # # 返回结果（可能少于 top_n，甚至为空）
        # return self._get_knowledge_by_ids(combined[:top_n])
        return []
    
    def retrieve_by_tags(
        self,
        question_types: List[str],
        modalities: List[str],
        organs: List[str],
        top_n: int = 5
    ) -> List[Dict]:
        """
        根据标签名检索知识（便捷方法）
        """
        qt_ids = tags_to_ids(question_types, self.question_type_tags)
        mod_ids = tags_to_ids(modalities, self.modality_tags)
        org_ids = tags_to_ids(organs, self.organ_tags)
        
        return self.retrieve_knowledge(qt_ids, mod_ids, org_ids, top_n)
    
    def _get_knowledge_by_ids(self, ids: List[int]) -> List[Dict]:
        """根据ID列表获取知识条目"""
        id_set = set(ids)
        return [k for k in self.knowledge_items if k['id'] in id_set]
    
    def get_knowledge_by_id(self, knowledge_id: int) -> Optional[Dict]:
        """根据ID获取单个知识条目"""
        return next((k for k in self.knowledge_items if k['id'] == knowledge_id), None)
    
    # ==================== 知识管理 ====================
    
    def add_knowledge(
        self,
        knowledge_text: str,
        question_type_ids: List[int],
        modality_ids: List[int],
        organ_ids: List[int],
        source: str = "unknown"
    ) -> int:
        """
        添加新知识条目
        
        Args:
            knowledge_text: 知识内容
            question_type_ids: 问题类型ID列表
            modality_ids: 图像模态ID列表
            organ_ids: 器官组织ID列表
            source: 来源
            
        Returns:
            int: 新知识的ID
        """
        # 去重并排序
        question_type_ids = sorted(set(question_type_ids))
        modality_ids = sorted(set(modality_ids))
        organ_ids = sorted(set(organ_ids))
        
        # 生成三元ID
        triple_id = make_triple_id(question_type_ids, modality_ids, organ_ids)
        
        # 生成新ID
        current_max_id = max((k['id'] for k in self.knowledge_items), default=0)
        new_id = current_max_id + 1
        
        new_knowledge = {
            'id': new_id,
            'knowledge': knowledge_text,
            'triple_id': triple_id,
            'question_type_ids': question_type_ids,
            'modality_ids': modality_ids,
            'organ_ids': organ_ids,
            'timestamp': get_cur_time(),
            'source': source
        }
        
        self.knowledge_items.append(new_knowledge)
        
        # 更新三元索引
        if new_id not in self.triple_index[triple_id]:
            self.triple_index[triple_id].append(new_id)
        
        # 更新单级索引
        for qt_id in question_type_ids:
            if new_id not in self.question_type_index[qt_id]:
                self.question_type_index[qt_id].append(new_id)
        for mod_id in modality_ids:
            if new_id not in self.modality_index[mod_id]:
                self.modality_index[mod_id].append(new_id)
        for org_id in organ_ids:
            if new_id not in self.organ_index[org_id]:
                self.organ_index[org_id].append(new_id)
        
        logging.info(f"Added knowledge ID {new_id} with triple_id {triple_id}: {knowledge_text[:50]}...")
        return new_id
    
    def add_knowledge_by_tags(
        self,
        knowledge_text: str,
        question_types: List[str],
        modalities: List[str],
        organs: List[str],
        source: str = "unknown"
    ) -> int:
        """
        根据标签名添加知识（便捷方法）
        """
        qt_ids = tags_to_ids(question_types, self.question_type_tags)
        mod_ids = tags_to_ids(modalities, self.modality_tags)
        org_ids = tags_to_ids(organs, self.organ_tags)
        
        return self.add_knowledge(knowledge_text, qt_ids, mod_ids, org_ids, source)
    
    def modify_knowledge(
        self,
        knowledge_id: int,
        new_knowledge_text: str = None,
        new_question_type_ids: List[int] = None,
        new_modality_ids: List[int] = None,
        new_organ_ids: List[int] = None,
        source: str = "unknown"
    ) -> bool:
        """
        修改知识条目
        
        Returns:
            bool: 是否成功修改
        """
        knowledge_item = self.get_knowledge_by_id(knowledge_id)
        
        if knowledge_item is None:
            logging.warning(f"Modify failed: Knowledge ID {knowledge_id} not found. Will add as new.")
            if new_knowledge_text and (new_question_type_ids or new_modality_ids or new_organ_ids):
                self.add_knowledge(
                    new_knowledge_text,
                    new_question_type_ids or [],
                    new_modality_ids or [],
                    new_organ_ids or [],
                    source
                )
            return False
        
        # 清除旧索引
        old_triple_id = knowledge_item.get('triple_id', '')
        if old_triple_id and knowledge_id in self.triple_index.get(old_triple_id, []):
            self.triple_index[old_triple_id].remove(knowledge_id)
            if not self.triple_index[old_triple_id]:
                del self.triple_index[old_triple_id]
        
        for qt_id in knowledge_item.get('question_type_ids', []):
            if qt_id in self.question_type_index and knowledge_id in self.question_type_index[qt_id]:
                self.question_type_index[qt_id].remove(knowledge_id)
        for mod_id in knowledge_item.get('modality_ids', []):
            if mod_id in self.modality_index and knowledge_id in self.modality_index[mod_id]:
                self.modality_index[mod_id].remove(knowledge_id)
        for org_id in knowledge_item.get('organ_ids', []):
            if org_id in self.organ_index and knowledge_id in self.organ_index[org_id]:
                self.organ_index[org_id].remove(knowledge_id)
        
        # 更新内容
        if new_knowledge_text:
            knowledge_item['knowledge'] = new_knowledge_text
        if new_question_type_ids is not None:
            knowledge_item['question_type_ids'] = sorted(set(new_question_type_ids))
        if new_modality_ids is not None:
            knowledge_item['modality_ids'] = sorted(set(new_modality_ids))
        if new_organ_ids is not None:
            knowledge_item['organ_ids'] = sorted(set(new_organ_ids))
        
        # 重新生成三元ID
        new_triple_id = make_triple_id(
            knowledge_item['question_type_ids'],
            knowledge_item['modality_ids'],
            knowledge_item['organ_ids']
        )
        knowledge_item['triple_id'] = new_triple_id
        knowledge_item['timestamp'] = get_cur_time()
        knowledge_item['source'] = source
        
        # 更新新索引
        if knowledge_id not in self.triple_index[new_triple_id]:
            self.triple_index[new_triple_id].append(knowledge_id)
        
        for qt_id in knowledge_item['question_type_ids']:
            if knowledge_id not in self.question_type_index[qt_id]:
                self.question_type_index[qt_id].append(knowledge_id)
        for mod_id in knowledge_item['modality_ids']:
            if knowledge_id not in self.modality_index[mod_id]:
                self.modality_index[mod_id].append(knowledge_id)
        for org_id in knowledge_item['organ_ids']:
            if knowledge_id not in self.organ_index[org_id]:
                self.organ_index[org_id].append(knowledge_id)
        
        logging.info(f"Modified knowledge ID {knowledge_id}, new triple_id: {new_triple_id}")
        return True
    
    def delete_knowledge(self, knowledge_id: int) -> bool:
        """
        删除知识条目
        
        Returns:
            bool: 是否成功删除
        """
        knowledge_item = self.get_knowledge_by_id(knowledge_id)
        
        if knowledge_item is None:
            logging.warning(f"Delete failed: Knowledge ID {knowledge_id} not found.")
            return False
        
        # 从三元索引中移除
        triple_id = knowledge_item.get('triple_id', '')
        if triple_id and knowledge_id in self.triple_index.get(triple_id, []):
            self.triple_index[triple_id].remove(knowledge_id)
            if not self.triple_index[triple_id]:
                del self.triple_index[triple_id]
        
        # 从单级索引中移除
        for qt_id in knowledge_item.get('question_type_ids', []):
            if qt_id in self.question_type_index and knowledge_id in self.question_type_index[qt_id]:
                self.question_type_index[qt_id].remove(knowledge_id)
                if not self.question_type_index[qt_id]:
                    del self.question_type_index[qt_id]
        
        for mod_id in knowledge_item.get('modality_ids', []):
            if mod_id in self.modality_index and knowledge_id in self.modality_index[mod_id]:
                self.modality_index[mod_id].remove(knowledge_id)
                if not self.modality_index[mod_id]:
                    del self.modality_index[mod_id]
        
        for org_id in knowledge_item.get('organ_ids', []):
            if org_id in self.organ_index and knowledge_id in self.organ_index[org_id]:
                self.organ_index[org_id].remove(knowledge_id)
                if not self.organ_index[org_id]:
                    del self.organ_index[org_id]
        
        # 从列表中移除
        self.knowledge_items.remove(knowledge_item)
        
        logging.info(f"Deleted knowledge ID {knowledge_id}")
        return True
    
    # ==================== 辅助方法 ====================
    
    def get_all_tags(self) -> Dict[str, Dict[str, int]]:
        """获取所有可用标签及其ID"""
        return {
            'question_types': self.question_type_tags,
            'modalities': self.modality_tags,
            'organs': self.organ_tags
        }
    
    def get_tag_list(self, category: str) -> List[str]:
        """获取某个类别的所有标签名"""
        if category == 'question_type':
            return list(self.question_type_tags.keys())
        elif category == 'modality':
            return list(self.modality_tags.keys())
        elif category == 'organ':
            return list(self.organ_tags.keys())
        return []
    
    def get_statistics(self) -> Dict:
        """获取知识库统计信息"""
        return {
            'total_knowledge': len(self.knowledge_items),
            'unique_triple_ids': len(self.triple_index),
            'question_types_used': len(self.question_type_index),
            'modalities_used': len(self.modality_index),
            'organs_used': len(self.organ_index),
            'question_type_distribution': {ID_TO_QUESTION_TYPE.get(k, str(k)): len(v) for k, v in self.question_type_index.items()},
            'modality_distribution': {ID_TO_MODALITY.get(k, str(k)): len(v) for k, v in self.modality_index.items()},
            'organ_distribution': {ID_TO_ORGAN.get(k, str(k)): len(v) for k, v in self.organ_index.items()}
        }