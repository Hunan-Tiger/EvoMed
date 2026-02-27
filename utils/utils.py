import requests
import re
import copy
import time
import logging
import json
import random
import base64
from PIL import Image
from io import BytesIO
from openai import OpenAI, APIConnectionError, RateLimitError, APIError



def image_to_base64(image):
    """将 PIL Image 转换为 base64 字符串"""
    if isinstance(image, str):
        # 如果是文件路径
        with open(image, "rb") as f:
            return base64.standard_b64encode(f.read()).decode("utf-8")
    elif isinstance(image, Image.Image):
        # 如果是 PIL Image
        # 根据是否有透明通道选择转换模式
        if image.mode == 'RGBA' or (image.mode == 'P' and 'transparency' in image.info):
            # 保留透明度
            image = image.convert('RGBA')
        elif image.mode not in ('RGB', 'L'):
            # 转换为 RGB（无透明度）
            image = image.convert('RGB')
        
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.standard_b64encode(buffered.getvalue()).decode("utf-8")
    elif isinstance(image, list):
        # 如果是图像列表，递归处理每个图像并返回列表
        img_list = [image_to_base64(img) for img in image]
        if len(img_list) == 1:
            return img_list[0]
        return img_list
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def inference_chat_local(chat, args, api_url="http://localhost:8000/v1", max_retries=10):
    """
    """
    
    # 1. 预处理 Messages，适配 vLLM/OpenAI 格式
    # Qwen2.5-VL 在 vLLM 中接收的标准格式是 OpenAI Vision 格式
    formatted_messages = []
    for msg in chat:
        if isinstance(msg.get("content"), list):
            formatted_content = []
            for item in msg["content"]:
                if item.get("type") == "image":
                    # 处理图像
                    image = item.get("image")
                    if image is not None:
                        base64_image = image_to_base64(image)
                        formatted_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        })
                elif item.get("type") == "image_url":
                    formatted_content.append(item)
                elif item.get("type") == "text":
                    formatted_content.append({
                        "type": "text",
                        "text": item["text"]
                    })
            formatted_messages.append({
                "role": msg["role"],
                "content": formatted_content
            })
        else:
            formatted_messages.append(msg)

    # 2. 构造 Payload
    payload = {
        "model": "Qwen/Qwen2.5-VL-7B-Instruct", # 必须与启动参数一致
        "messages": formatted_messages,
        "max_tokens": args.rollout_max_tokens,
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "seed":  getattr(args, 'seed', None),
        # vLLM 支持的额外参数
        "top_k": getattr(args, 'rollout_top_k', -1),  # -1 表示禁用
        "repetition_penalty": getattr(args, 'rollout_repetition_penalty', 1.0),
        "no_repeat_ngram_size": getattr(args, 'rollout_no_repeat_ngram_size', 0),
        "stop": getattr(args, 'stop_tokens', None),
    }

    logging.info(f"--- Calling vLLM API with parameters ---")
    logging.info(f"  url: {api_url}")
    logging.info(f"  model: {payload.get('model')}")
    logging.info(f"  max_tokens: {payload.get('max_tokens')}")
    logging.info(f"  temperature: {payload.get('temperature')}")
    logging.info(f"  top_p: {payload.get('top_p')}")
    logging.info(f"  top_k: {payload.get('top_k')}")
    logging.info(f"  repetition_penalty: {payload.get('repetition_penalty')}")
    logging.info(f"  no_repeat_ngram_size: {payload.get('no_repeat_ngram_size')}")
    logging.info(f"  seed: {payload.get('seed')}")
    logging.info(f"----------------------------------------")

    headers = {
        "Content-Type": "application/json"
    }

    # 3. 发送请求
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{api_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=600.0
            )
            response.raise_for_status()
            
            result = response.json()
            if result is None or "choices" not in result or len(result["choices"]) == 0:
                logging.error(f"API response is invalid: {response.text}")
                return "response_invalid", None
            content = result["choices"][0]["message"]["content"]
            
            # 构建 usage 对象
            usage_data = result.get("usage", {})
            
            class Usage:
                def __init__(self, data):
                    self.prompt_tokens = data.get("prompt_tokens", 0)
                    self.completion_tokens = data.get("completion_tokens", 0)
                    self.total_tokens = data.get("total_tokens", 0)
            
            usage = Usage(usage_data)
            return content, usage
        
        except requests.exceptions.Timeout as e:
            logging.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limit
                logging.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
            else:
                logging.error(f"HTTP error: {e}, Response: {response.text}")
                raise
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

    return "", None



def inference_chat_api_vllm(chat, args, max_retries=10):
    """
    使用 requests 直接调用 vLLM 兼容的 API 端点，
    支持更多生成参数控制，适用于 Qwen2.5-VL 模型
    """
    url = args.base_url
    
    headers = {
        "Content-Type": "application/json",
    }
    if hasattr(args, 'api_key') and args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    
    # 格式化消息
    formatted_messages = []
    for msg in chat:
        if isinstance(msg.get("content"), list):
            formatted_content = []
            for item in msg["content"]:
                if item.get("type") == "image":
                    # 处理图像
                    image = item.get("image")
                    if image is not None:
                        base64_image = image_to_base64(image)
                        formatted_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        })
                elif item.get("type") == "image_url":
                    formatted_content.append(item)
                elif item.get("type") == "text":
                    formatted_content.append({
                        "type": "text",
                        "text": item["text"]
                    })
            formatted_messages.append({
                "role": msg["role"],
                "content": formatted_content
            })
        else:
            formatted_messages.append(msg)

    # 构建请求体，包含所有生成参数
    payload = {
        "model": args.model_name_or_path,
        "messages": formatted_messages,
        "max_tokens": args.rollout_max_tokens,
        "temperature": args.rollout_temperature,
        "top_p": args.rollout_top_p,
        "seed":  getattr(args, 'seed', None),
        # vLLM 支持的额外参数
        "top_k": getattr(args, 'rollout_top_k', -1),  # -1 表示禁用
        "repetition_penalty": getattr(args, 'rollout_repetition_penalty', 1.0),
        "no_repeat_ngram_size": getattr(args, 'rollout_no_repeat_ngram_size', 0),
        "stop": getattr(args, 'stop_tokens', None),
        "presence_penalty": getattr(args, 'presence_penalty', 0.0),
        "frequency_penalty": getattr(args, 'frequency_penalty', 0.0),
    }
    
    # 移除 None 值
    # payload = {k: v for k, v in payload.items() if v is not None}
    
    logging.info(f"--- Calling vLLM API with parameters ---")
    logging.info(f"  url: {url}")
    logging.info(f"  model: {payload.get('model')}")
    logging.info(f"  max_tokens: {payload.get('max_tokens')}")
    logging.info(f"  temperature: {payload.get('temperature')}")
    logging.info(f"  top_p: {payload.get('top_p')}")
    logging.info(f"  top_k: {payload.get('top_k')}")
    logging.info(f"  repetition_penalty: {payload.get('repetition_penalty')}")
    logging.info(f"  no_repeat_ngram_size: {payload.get('no_repeat_ngram_size')}")
    logging.info(f"  seed: {payload.get('seed')}")
    logging.info(f"----------------------------------------")
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=600.0
            )
            response.raise_for_status()
            
            result = response.json()
            if result is None or "choices" not in result or len(result["choices"]) == 0:
                logging.error(f"API response is invalid: {response.text}")
                return "response_invalid", None
            content = result["choices"][0]["message"]["content"]
            
            # 构建 usage 对象
            usage_data = result.get("usage", {})
            
            class Usage:
                def __init__(self, data):
                    self.prompt_tokens = data.get("prompt_tokens", 0)
                    self.completion_tokens = data.get("completion_tokens", 0)
                    self.total_tokens = data.get("total_tokens", 0)
            
            usage = Usage(usage_data)
            return content, usage
        
        except requests.exceptions.Timeout as e:
            logging.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limit
                logging.warning(f"Rate limit hit (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
            else:
                logging.error(f"HTTP error: {e}, Response: {response.text}")
                raise
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

    return "", None


def inference_chat_api(messages, args, max_retries=5):
    """
    使用 OpenAI 兼容 API 进行推理（适用于 Qwen2.5-VL）
    
    Args:
        messages: 对话消息列表
        args: 参数对象，包含 rollout_max_tokens, rollout_temperature, rollout_top_p
    
    Returns:
        str: 模型生成的文本
    """

    if not args.api_key:
        raise ValueError("API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    
    # 转换消息格式（处理图像）
    formatted_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        if isinstance(content, str):
            # 纯文本消息
            formatted_messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            # 多模态消息（包含图像）
            formatted_content = []
            for item in content:
                if item.get("type") == "text":
                    formatted_content.append({
                        "type": "text",
                        "text": item.get("text", "")
                    })
                elif item.get("type") == "image":
                    # 处理图像
                    image = item.get("image")
                    if image is not None:
                        base64_image = image_to_base64(image)
                        formatted_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        })
                elif item.get("type") == "image_url":
                    # 已经是 URL 格式
                    formatted_content.append(item)
            
            formatted_messages.append({"role": role, "content": formatted_content})
        else:
            formatted_messages.append({"role": role, "content": str(content)})
    
    # 创建一个用于打印的消息副本，避免打印完整的 base64 字符串
    # printable_messages = copy.deepcopy(formatted_messages)
    # for msg in printable_messages:
    #     if isinstance(msg.get("content"), list):
    #         for item in msg["content"]:
    #             if item.get("type") == "image_url" and item.get("image_url", {}).get("url", "").startswith("data:image"):
    #                 # 找到原始 PIL Image 对象以获取尺寸
    #                 original_image = None
    #                 for original_msg in messages:
    #                     if isinstance(original_msg.get("content"), list):
    #                         for original_item in original_msg["content"]:
    #                             if original_item.get("type") == "image" and isinstance(original_item.get("image"), Image.Image):
    #                                 original_image = original_item.get("image")
    #                                 break
    #                     if original_image:
    #                         break
                    
    #                 if original_image:
    #                     item["image_url"]["url"] = f"<Image size: {original_image.size}>"
    #                 else:
    #                     item["image_url"]["url"] = "<Base64 Image Data>"
    # print(f" ######### Calling API with messages: {printable_messages} ######### ")


    for attempt in range(max_retries):
        try:
            # 将不支持的标准参数移入 extra_body
            extra_body = {
                "top_k": getattr(args, 'rollout_top_k', -1),
                "repetition_penalty": getattr(args, 'rollout_repetition_penalty', 1.0),
                "no_repeat_ngram_size": getattr(args, 'rollout_no_repeat_ngram_size', 0),
            }
            logging.info(f"--- Calling API with parameters ---")
            logging.info(f"  model: {args.model_name_or_path}")
            logging.info(f"  max_tokens: {args.rollout_max_tokens}")
            logging.info(f"  temperature: {args.rollout_temperature}")
            logging.info(f"  top_p: {args.rollout_top_p}")
            logging.info(f"  extra_body: {extra_body}")
            logging.info(f"  timeout: 120.0")
            logging.info(f"  seed: {args.seed}")
            logging.info(f"------------------------------------")

            response = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=formatted_messages,
                max_tokens=args.rollout_max_tokens,
                temperature=args.rollout_temperature,
                top_p=args.rollout_top_p,
                extra_body=extra_body,
                timeout=120.0,
                seed=args.seed,
            )
            if response and response.choices:
                content = response.choices[0].message.content
                usage = response.usage
                return content, usage # 返回内容和 usage 对象
            else:
                logging.error(f"API response is invalid: {response}")
                return "", None # 返回空值
        
        except APIConnectionError as e:
            wait_time = min(2 ** attempt, 60)  # 指数退避,最大60秒
            logging.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logging.error("Max retries reached. Failed to connect.")
                raise
        
        except RateLimitError as e:
            wait_time = 60  # 速率限制,等待更久
            logging.warning(f"Rate limit error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                logging.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
        
        except APIError as e:
            logging.error(f"API error: {e}")
            if e.status_code >= 500:  # 服务器错误,可以重试
                wait_time = min(2 ** attempt, 60)
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise
            else:  # 客户端错误,不重试
                raise

        except Exception as e:
            print(f"API Error: {e}")
            raise

    return None


def get_cur_time():
    import time
    cur_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    return cur_time


def extract_answer(response_text):
    # 查找 <answer> 标签内的内容，忽略大小写，允许跨行
    pattern = r"<answer>(.*?)</answer>|<Answer>(.*?)</Answer>"
    match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        return "PARSE_ERROR"
    

def parse_tag(text, tag):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text.split(f"<{tag}>")[-1].strip() if f"<{tag}>" in text else text.strip().splitlines()[-1]



def process_dataset(experiment_name_or_path, truncate=None, split='train', img_size=512):
    data = load_dataset(f'{experiment_name_or_path}', split=split, trust_remote_code=True) # type: ignore
    print(data[0])
    dataset = []
    for i, item in enumerate(data):
        question_sample = item['question']
        answer_sample = item['answer']

        if answer_sample.lower() not in ['yes', 'no']: # filter single-choice
            continue
        
        img_pil = item['image'].resize((img_size, img_size))  # reduce vRAM burden
        
        dataset.append({
                'prompt': [
                    {'role': 'user', 'content': [
                        {"type": "image", "image": img_pil},
                        {"type": "text", "text": question_sample},
                    ]}
                ],
                'image':img_pil,
                'answer':answer_sample,
            })

    if split == 'train':
        random.shuffle(dataset)

    if truncate is not None:
        dataset = dataset[:truncate]

    return dataset



def parse_id(id_val):
    """辅助函数：从各种格式中提取数字ID (e.g., "G17" -> 17, 17 -> 17)"""
    if isinstance(id_val, int):
        return id_val
    if isinstance(id_val, str):
        # 提取字符串中的第一个连续数字序列
        match = re.search(r'\d+', id_val)
        if match:
            return int(match.group())
    return None

def parse_keywords(keywords_val):
    """辅助函数：处理关键词，兼容字符串和列表格式"""
    if isinstance(keywords_val, list):
        return [str(k).strip() for k in keywords_val]
    if isinstance(keywords_val, str):
        # 处理中文或英文逗号
        keywords_val = keywords_val.replace('，', ',')
        return [kw.strip() for kw in keywords_val.split(',') if kw.strip()]
    return []



def calculate_word_recall(ground_truth, prediction):
    """计算词级别的Recall: |GT_words ∩ Pred_words|/|GT_words|"""
    # 转换为小写并分词
    gt_words = set(ground_truth.lower().split())
    pred_words = set(prediction.lower().split())
    
    if len(gt_words) == 0:
        return 0.0
    
    intersection = gt_words & pred_words
    recall = len(intersection) / len(gt_words)
    return recall



# 用于test上的评估
class Test_Judger:
    def __init__(self, inference_func):

        self.inference = inference_func

        self.JUDGER_SYSTEM_PROMPT = r"""You are a Medical AI Exam Grader. Grade the model's answer against the Ground Truth (GT).

**GRADING RULES:**

1. **Single Choice:** Match by letter (A/B/C/D) or option text. single-select requires exact match, don't accept multi-select.
2. **Multiple Choice:** All and only correct options must be selected, don't accept partial credit and any incorrect options. Order does not matter.
3. **Yes/No:** "Yes"="True"="Positive"="Present"; "No"="False"="Negative"="Absent".
4. **Robustness:** Ignore case, punctuation, filler text (e.g., "The answer is A" → "A. Pneumonia", "A" -> "A. Pneumonia").

**OUTPUT (JSON only):**
{"status": "correct" | "wrong", "reason": "Brief explanation."}
"""
    def _extract_answer(self, vlm_response: str) -> str:
        """
        从 VLM 输出中提取 <Answer> 标签内容。
        如果找不到标签，返回最后一段文本（兜底）。
        """
        # 尝试匹配 <Answer>...</Answer> (支持换行)
        # match = re.search(r"<Answer>(.*?)</Answer>", vlm_response, re.DOTALL | re.IGNORECASE)
        # if match:
        #     return match.group(1).strip()
        
        # match_open = re.search(r"<Answer>(.*)", vlm_response, re.DOTALL | re.IGNORECASE)
        # if match_open:
        #     return match_open.group(1).strip()
        
        # 如果格式不对，选择最后一行
        final_answer = vlm_response.strip().splitlines()[-1]
        return final_answer
    
    def evaluate(
        self, 
        question: str,
        vlm_response: str, 
        ground_truth: str
    ):
        """
        评估响应的正确性
        
        Args:
            response: 模型响应
            ground_truth: 标准答案
        
        Returns:
            correct/wrong
        """
        # 使用 LLM 评估
        extracted_pred = self._extract_answer(vlm_response)
        logging.info(f"Extracted Model Answer for Judging: {extracted_pred}")

        prompt = f"**Question:** {question}\n\n**Ground Truth:** {ground_truth}\n\n**Model Answer:** {extracted_pred}"
        
        chat = [{"role": "system", "content": self.JUDGER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]
        
        response = self.inference(chat)
        if isinstance(response, tuple) and len(response) == 2:
            content, usage = response
            response = content 
        
        return self._parse_evaluation(response), usage.total_tokens if usage and hasattr(usage, 'total_tokens') else None
    
    
    def _parse_evaluation(self, response: str):
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
            
            eval_result = result.get('status', 'No status').lower().strip()
            critique = result.get('reason', 'No reason provided')
            
            # 标准化结果
            if 'correct' in eval_result and 'wrong' not in eval_result:
                eval_result = 'correct'
            else:
                eval_result = 'wrong'
            return {'result': eval_result, 'reason': f"{critique}"}
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.warning(f"Failed to parse evaluation JSON: {e}")
            logging.warning(f"Response: {response}")
            # 降级处理：从文本中提取
            response_lower = response.strip().lower()
            
            if 'correct' in response_lower and 'wrong' not in response_lower:
                return {'result': 'correct', 'reason': 'Parsed from text (fallback)'}
            else:
                return {'result': 'wrong', 'reason': 'Parse failed, default to wrong'}

    
