import logging
import json
import re
from typing import List, Dict, Tuple, Optional
from retrieval.knowledge_base import (
    QUESTION_TYPE_TAGS, 
    IMAGE_MODALITY_TAGS, 
    ORGAN_TISSUE_TAGS,
    tags_to_ids,
    ids_to_tags,
    ID_TO_QUESTION_TYPE,
    ID_TO_MODALITY,
    ID_TO_ORGAN
)
from utils.utils import inference_chat_api_compatible, inference_chat_local

# ==================== 分类 Prompt ====================

CLASSIFICATION_SYSTEM_PROMPT = """You are a medical image analysis expert. Your task is to:
1. First, provide detailed visual findings from the image.
2. Then, classify each (question, image) pair with appropriate tags.

**Decision Priority:**
- question_types and organs: Decide mainly from the QUESTION TEXT; use the image only to confirm/refine if the text is ambiguous.
- modality: Decide mainly from the IMAGE; use the question only if the image is unclear.
- If text and image conflict: modality follows the image; question_types/organs follow the question.

**Output Format:**
Output ONLY this valid JSON (no extra text). Use only predefined tags; do not invent tags:
{
  "visual_findings": "Detailed description of observable features in the image.",
  "question_types": ["tag1", "tag2", ...],
  "modality": "single_tag",
  "organs": ["tag1", "tag2", ...]
}

**Rules:**
1. visual_findings: Detailed description of observable features in the image (e.g., density, shape, location, texture, borders). If a feature is ambiguous, describe its visual properties (e.g., 'linear density') rather than naming a disease. Mention 2-3 differential possibilities if unsure. DO NOT make any diagnosis here, only describe what you see.
2. question_types: Select 1-2 most relevant tags based primarily on the medical question.
3. modality: Select exactly 1 tag describing the image modality.
4. organs: Select 1-5 most relevant organ or tissue tags referenced or implied in the question.
5. Use ONLY tags from the provided lists.
6. Output ONLY the JSON object. Do not include explanations or additional text.
"""


def get_classification_prompt(
    question: str,
    question_type_tags: List[str],
    modality_tags: List[str],
    organ_tags: List[str]
) -> str:
    """生成分类 prompt"""
    
    qt_list = ", ".join(question_type_tags)
    mod_list = ", ".join(modality_tags)
    org_list = ", ".join(organ_tags)
    
    prompt = f"""Analyze the following medical question and perform classification.

**Medical Question:**
{question}

**Available Question Type Tags:**
{qt_list}

**Available Image Modality Tags:**
{mod_list}

**Available Organ / Tissue Tags:**
{org_list}
"""
    return prompt



# ==================== 答案评估 Prompt ====================
EVALUATION_SYSTEM_PROMPT = """You are a Text-Based Medical Logic Auditor. You evaluate the diagnostic reasoning of a Vision-Language Model (VLM) based on text descriptions alone.

**Evaluation Protocol:**

**Step 1: Strict Answer Matching**
- **CORRECT**: the VLM's full response EXACTLY matches the ground truth answer.
- **WRONG**: any deviation from the ground truth answer, including partial correctness or ambiguity.

**Step 2: Check for Logic/Knowledge Errors (Only if Answer is Wrong)**
- Only if the status is "wrong", analyze why.
- Identify if the error is due to:
    a) Reasoning Logic Error: The VLM's reasoning process is flawed, even if the visual description is accurate.
    b) Knowledge Deficiency: The VLM lacks necessary medical knowledge to arrive at the correct answer, despite accurate perception and reasoning.

**Output Format:**
Return ONLY a valid JSON object:
{
  "status": "correct" | "wrong",
  "error_type": "none" | "reasoning_logic_error" | "knowledge_deficiency",
  "critique": "Concise explanation. Do not hallucinate ground truth details not provided in the input text."
}
"""



def get_evaluation_prompt(question: str, response: str, ground_truth: str) -> str:
    """生成答案评估 prompt"""
    prompt = f"""**Question:** {question}

**Ground Truth:** {ground_truth}

**VLM Response:** {response}
"""
    return prompt



class RerietrievalClassifier:
    """
    使用 VLM 进行知识分类，返回三元ID
    """
    
    def __init__(self, inference_func, args):
        """
        Args:
            inference_func: 推理函数，接受 chat 列表，返回响应字符串
            args: 参数配置
        """
        self.inference = inference_func
        self.args = args
        
        # 标签定义
        self.question_type_tags = QUESTION_TYPE_TAGS
        self.modality_tags = IMAGE_MODALITY_TAGS
        self.organ_tags = ORGAN_TISSUE_TAGS
    
    def classify(
        self, 
        question: str, 
        image
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        对问题和图像进行分类
        
        Args:
            question: 医学问题文本
            image: 图像（路径或 PIL Image）
            
        Returns:
            Tuple[List[int], List[int], List[int]]: 
                (question_type_ids, modality_ids, organ_ids)
        """
        prompt = get_classification_prompt(
            question,
            list(self.question_type_tags.keys()),
            list(self.modality_tags.keys()),
            list(self.organ_tags.keys())
        )
        
        chat = [
            {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        response = self.inference(chat)
        if isinstance(response, tuple) and len(response) == 2:
            content, usage = response
            response = content
        
        # 解析响应
        return self._parse_classification(response), usage.total_tokens if usage and hasattr(usage, 'total_tokens') else None

    def _parse_classification(
        self, 
        response: str
    ) -> Tuple[List[int], List[int], List[int]]:
        """解析分类响应，返回ID列表"""
        try:
            # 尝试提取 JSON
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = json.loads(response)
            
            question_types = result.get('question_types', [])
            modality = result.get('modality', '')
            organs = result.get('organs', [])
            visual_findings = result.get('visual_findings', '')
            
            # 确保 modality 是列表
            if isinstance(modality, str):
                modalities = [modality] if modality else []
            else:
                modalities = modality if modality else []
            
            # 转换为ID
            qt_ids = tags_to_ids(question_types, self.question_type_tags)
            mod_ids = tags_to_ids(modalities, self.modality_tags)
            org_ids = tags_to_ids(organs, self.organ_tags)

            # 如果解析失败，返回默认值
            if not qt_ids:
                qt_ids = None
            if not mod_ids:
                mod_ids = None
            if not org_ids:
                org_ids = None
            
            return qt_ids, mod_ids, org_ids, visual_findings

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.warning(f"Failed to parse classification response: {e}")
            logging.warning(f"Response: {response}")
            # 返回默认值
            return None, None, None, None

    def get_tag_names(
        self, 
        qt_ids: List[int], 
        mod_ids: List[int], 
        org_ids: List[int]
    ) -> Tuple[List[str], List[str], List[str]]:
        """将ID转换为标签名（用于日志显示）"""
        qt_names = ids_to_tags(qt_ids, ID_TO_QUESTION_TYPE)
        mod_names = ids_to_tags(mod_ids, ID_TO_MODALITY)
        org_names = ids_to_tags(org_ids, ID_TO_ORGAN)
        return qt_names, mod_names, org_names



class AnswerEvaluator:
    """
    使用小型 LLM 评估答案正确性
    """
    def __init__(self, inference_func):
        """
        Args:
            inference_func: 推理函数，接受 chat 列表，返回响应字符串
        """
        self.inference = inference_func
    
    def evaluate(
        self, 
        question: str, 
        response: str, 
        ground_truth: str
    ) -> Dict[str, str]:
        """
        评估响应的正确性
        
        Args:
            question: 原始问题
            response: 模型响应
            ground_truth: 标准答案
        
        Returns:
            Dict: {'result': 'correct'/'wrong', 'reason': '...'}
        """
        # 使用 LLM 评估
        prompt = get_evaluation_prompt(question, response, ground_truth)
        
        chat = [{"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
        
        response = self.inference(chat)
        if isinstance(response, tuple) and len(response) == 2:
            content, usage = response
            response = content
        
        return self._parse_evaluation(response), usage.total_tokens if usage and hasattr(usage, 'total_tokens') else None
    
    
    def _parse_evaluation(self, response: str) -> Dict[str, str]:
        """解析评估响应"""
        import re
        import json
        
        try:
            # 尝试提取 JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                # 尝试直接解析
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    result = json.loads(response)
            
            eval_result = result.get('status', 'wrong').lower().strip()
            error_type = result.get('error_type', 'No error_type provided')
            critique = result.get('critique', 'No critique provided')
            
            # 标准化结果
            if 'correct' in eval_result and 'wrong' not in eval_result:
                eval_result = 'correct'
            else:
                eval_result = 'wrong'
            
            return {'result': eval_result, 'reason': f"{error_type}: {critique}"}
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.warning(f"Failed to parse evaluation JSON: {e}")
            logging.warning(f"Response: {response}")
            # 降级处理：从文本中提取
            response_lower = response.strip().lower()
            
            if 'correct' in response_lower and 'wrong' not in response_lower:
                return {'result': 'correct', 'reason': 'Parsed from text (fallback)'}
            else:
                return {'result': 'wrong', 'reason': 'Parse failed, default to wrong'}


def create_evaluator_inference_func(global_args=None):
    """
    创建评估器的推理函数
    可以使用本地小模型或 API
    """
    class EvalArgs:
        def __init__(self):
            self.model_name_or_path = global_args.evaluator_model
            # 将温度设为0以消除随机性，使输出尽可能固定
            self.rollout_temperature = 0.0
            self.rollout_max_tokens = 2048
            # 设置较小的 top_p 也有助于减少随机性
            self.rollout_top_p = 1.0
            self.rollout_top_k=1
            self.api_key = global_args.api_key if global_args is not None else None
            self.base_url = global_args.base_url if global_args is not None else None
            # 如果API支持seed，可以固定它
            self.seed = 42 
    
    args = EvalArgs()
    
    def inference_func(chat):
        if getattr(global_args, "infer_mode", "api") == "api":
            # Use the configured API model as the test judger or critic model.
            return inference_chat_api_compatible(chat, args)
        else:
            # Use the configured local model as the critic model.
            return inference_chat_local(chat, args)
    
    return inference_func


def create_classifier_inference_func(global_args=None):
    """
    创建分类器的推理函数
    
    可以使用本地小模型或 API
    """
    class ClassifyArgs:
        def __init__(self):
            self.model_name_or_path = global_args.classifier_model
            # 将温度设为0以消除随机性，使输出尽可能固定
            self.rollout_temperature = 0.0
            self.rollout_max_tokens = 2048
            # 设置较小的 top_p 也有助于减少随机性
            self.rollout_top_p = 1.0
            self.rollout_top_k=1
            self.api_key = global_args.api_key if global_args is not None else None
            self.base_url = global_args.base_url if global_args is not None else None
            # 如果API支持seed，可以固定它
            self.seed = 42 
    

    args = ClassifyArgs()
    
    def inference_func(chat):
        if getattr(global_args, "infer_mode", "api") == "api":
            # Use the configured API model as the retrieval classifier.
            return inference_chat_api_compatible(chat, args)
        else:
            # Use the configured local model as the retrieval classifier.
            return inference_chat_local(chat, args)
    
    return inference_func
