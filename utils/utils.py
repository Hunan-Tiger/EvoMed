import torch
from transformers import (
AutoModelForCausalLM, 
AutoTokenizer, 
Qwen2_5_VLForConditionalGeneration, 
AutoProcessor,
BitsAndBytesConfig,
GenerationConfig
)
from qwen_vl_utils import process_vision_info

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
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.standard_b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")



def model_loader(name_or_path, compute_dtype=torch.bfloat16, args=None):

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        name_or_path,
        device_map="auto", 
        trust_remote_code=True,
        dtype=compute_dtype,
        attn_implementation="flash_attention_2",
    ).eval()

    tokenizer = AutoProcessor.from_pretrained(
        name_or_path,
        use_fast=False,
        trust_remote_code=True,
    )

    return model, tokenizer



def inference_chat_local(model, tokenizer, messages, args):

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 处理视觉信息
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 分词
    inputs = tokenizer(
        text=[input_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)
    
    # 生成
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=args.rollout_max_tokens,
        temperature=args.rollout_temperature,
        top_p=args.rollout_top_p,
        repetition_penalty=1.1,          # 增加重复惩罚
        # no_repeat_ngram_size=3,          # 禁止重复 3-gram
        do_sample=True,
        pad_token_id=tokenizer.tokenizer.pad_token_id,
        eos_token_id=tokenizer.tokenizer.eos_token_id,
    )
    
    # 解码
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text


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
            response = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=formatted_messages,
                max_tokens=args.rollout_max_tokens,
                temperature=args.rollout_temperature,
                top_p=args.rollout_top_p,
                # 将不支持的标准参数移入 extra_body
                extra_body={
                    "repetition_penalty": getattr(args, 'rollout_repetition_penalty', 1.0)
                },
                timeout=120.0
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
        return text.split(f"<{tag}>")[-1].strip() if f"<{tag}>" in text else "PARSE_ERROR"



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


API_KEY_list = {
    'lijm': "sk-0e4ecba324df49bf80a831dd9bc67ba8",
    'wujl': "sk-d41c52973bf549ab8e0011a58efe6ef9",
    'songq': "sk-36d2e2eea6c84cc299b3e023f32cefcc",
    'xiongda': "sk-6bba59784ecb4d21b408c76dabb8b60e",
    'skd': '0f02bb4443b94ed0a68012e44f6f7efd'
}