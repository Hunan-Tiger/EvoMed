import re


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return text.strip()
    return text.split("####")[1].strip()

def extract_class_answer(text: str) -> str | None:
    # m = re.search(r'["\'](.*?)["\']', text)
    # if m:
    text = re.sub(r'[^A-Za-z]', '', text)
    return text.strip()


XML_COT_FORMAT = """\
<think>
{think}
</think>
<answer>
{answer}
</answer>
"""

think_json_path = "./think.json"

# Reward functions
def directly_ans_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    print('-'*20, f"Question:\n{q}", f"\nResponse:\n{responses[0]}", f"\nAnswer:\n{answer[0]}")
    return [1.0 if r.strip().lower().replace('.','') == a.strip().lower() else 0.0 for r, a in zip(responses, answer)]



def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nResponse:\n{responses[0]}", f"\nExtrated_ans:\n{extracted_responses[0]}", f"\nAnswer:\n{answer[0]}")
    return [1.0 if r.lower() == a.lower() else 0.0 for r, a in zip(extracted_responses, answer)]


def long_correctness_reward_func(completions, answer, **kwargs):
    """
    奖励函数：检查答案的所有词是否都出现在响应中
    
    Args:
        completions: 模型生成的完成列表
        answer: 标准答案列表
    
    Returns:
        奖励列表，每个元素为 1.0（所有词都匹配）或 0.0（有词未匹配）
    """
    extracted_responses = []
    for completion in completions:
        completion = completion[0]['content']
        if '<answer>' in completion and '</answer>' in completion:
            extracted_responses.append(
                completion.split('<answer>')[-1].split('</answer>')[0].strip()
            )
        else:
            extracted_responses.append('')
    
    rewards = []
    for response, ans in zip(extracted_responses, answer):
        # 转为小写并分词
        response_lower = response.lower()
        answer_words = ans.lower().split()
        
        # 检查答案的每个词是否都在响应中
        all_words_present = all(word in response_lower for word in answer_words)
        
        # 如果所有词都存在，给予奖励
        rewards.append(1.0 if all_words_present else 0.0)
    
    return rewards


def format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern1 = r"^<think>.*?</think>.*?<answer>.*?</answer>$"
    pattern2 = r"^<answer>.*?</answer>$"
    # pattern = f"({pattern1})|({pattern2})"
    pattern = pattern1
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]



def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count



def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]



from Levenshtein import ratio as levenshtein_ratio
def levenshtein_reward_func(prompts, completions, answer, **kwargs):
    """Reward function that checks if the completion get solutions correctly."""
    res = []
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    print('-'*20, f"Question:\n{q}", f"\nResponse:\n{responses[0]}", f"\nAnswer:\n{answer[0]}")

    for completion, sol in zip(completions, answer):
        completion = completion[0]['content']
        if '<answer>' in completion:
            t = completion.split('<answer>')[-1].split('</answer>')[0].strip()
            res.append(levenshtein_ratio(t, sol)*0.5)
        else:
            res.append(0.0)
    return res



def completion_length_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that encourages longer completions."""
    responses = [completion[0]["content"] for completion in completions]
    return [min(len(r) / 1000.0, 1.0) for r in responses]


def reward_function(prediction, groundtruth):
    gt_content = groundtruth.strip().lower()
    pred_content = prediction.strip().lower()
    
    # 情况1: 完全相等
    if pred_content == gt_content:
        return 'correct'
    
    # 情况2: 预测包含在答案内 (例如 GT="viral pneumonia", Pred="pneumonia")
    # 或者 答案包含在预测内 (例如 GT="yes", Pred="yes, it is")
    # 这里使用单词边界匹配来避免部分单词匹配 (如 "no" 匹配 "not")
    gt_pattern = r"\b" + re.escape(gt_content) + r"\b"
    pred_pattern = r"\b" + re.escape(pred_content) + r"\b"

    # if re.search(gt_pattern, pred_content):
    #     return 'correct' # 宽松模式：只要预测包含了答案就算对
    
    # 如果需要更细粒度的奖励，可以返回不同的分数，例如：
    if re.search(pred_pattern, gt_content):
        return 'partial correct' 
        
    return 'wrong'
