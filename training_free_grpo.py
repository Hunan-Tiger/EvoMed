import random
import time
import json
import re
import os
import argparse
from typing import List, Dict
from utils.prompt import query_prompt, summarize_prompt, query_prompt_noKnowledge, update_knowledge_prompt
from utils.utils import inference_chat_local, get_cur_time, extract_answer, model_loader, inference_chat_api, API_KEY_list
import logging
import sys
import pickle
from retrieval.knowledge_base import TagKnowledgeBase, make_triple_id
from retrieval.utils import RerietrievalClassifier, AnswerEvaluator, create_evaluator_inference_func,create_classifier_inference_func


query_system_prompt = r'''You are an intelligent Medical Diagnostic Agent.
Your goal is to provide accurate, evidence-based solutions to medical imaging problems.
You must always explicitly write out your step-by-step reasoning process before drawing a conclusion. Do not guess.

Provide a concise final answer according to the question type:
- For yes/no questions, answer "Yes" or "No".
- For identification tasks, answer with the disease name or organ name.
- For multiple-choice questions, answer with the selected option only (A/B/C/D).

The last part of your final response must follow the format below:
<answer>
[the final concise answer here]
</answer>
'''




class LLM(): 
    def __init__(self, args): 
        self.model_name_or_path = args.model_name_or_path 
        self.args = args
        
        # 日志配置
        os.makedirs(f'logs/{self.args.exp_name}', exist_ok=True)
        safe_time = get_cur_time().replace(":", "-").replace(" ", "_")
        self.log_file = f'logs/{self.args.exp_name}/train_{safe_time}.log'

        self.rollouts_per_problem = []

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # 记录初始化参数
        logging.info("=" * 80)
        logging.info("TRAINING INITIALIZATION")
        logging.info("=" * 80)
        logging.info(f"Arguments: {self.args}")
        logging.info(f"Log file: {self.log_file}")

        # 检查点配置
        os.makedirs('checkpoints', exist_ok=True)
        self.checkpoint_file = f'checkpoints/{self.args.exp_name}_train_{self.model_name_or_path}_checkpoint.pkl'
        logging.info(f"Checkpoint file: {self.checkpoint_file}")

        # 初始化基于标签的知识库
        os.makedirs('knowledges', exist_ok=True)
        self.kb_path = f'knowledges/{self.args.exp_name}_tag_kb_{self.model_name_or_path}.pkl'
        self.kb_version_dir = f'knowledges/{self.args.exp_name}_versions_{self.model_name_or_path}'
        os.makedirs(self.kb_version_dir, exist_ok=True)
        
        self.knowledge_base = TagKnowledgeBase(kb_path=self.kb_path)
        logging.info(f"Knowledge base path: {self.kb_path}")
        logging.info(f"KB version directory: {self.kb_version_dir}")
        logging.info(f"Initial knowledge items: {len(self.knowledge_base.knowledge_items)}")
        
        # 当前检索到的知识和三元ID
        self.current_retrieved_knowledge = []
        self.current_triple_ids = {
            'question_type_ids': [],
            'modality_ids': [],
            'organ_ids': []
        }
        self.current_triple_id_str = ""
        self.formatted_special_knowledges = "None"

        # 加载推理模型
        if self.args.infer_mode != "api":
            logging.info(f"Loading local model: {self.model_name_or_path}")
            self.model, self.tokenizer = model_loader(name_or_path=self.model_name_or_path, args=self.args)
            logging.info("Local model loaded successfully.")
        else:
            logging.info(f"Using API mode with model: {self.model_name_or_path}")

        # 初始化分类器
        classifier_inference_func = create_classifier_inference_func(self.model_name_or_path, self.args)
        self.classifier = RerietrievalClassifier(classifier_inference_func, self.args)
        logging.info("Knowledge Classifier initialized.")
        
        # 初始化答案评估器
        eval_inference_func = create_evaluator_inference_func(args.evaluator_model, self.args)
        self.evaluator = AnswerEvaluator(eval_inference_func)
        logging.info(f"Answer Evaluator initialized with model: {args.evaluator_model}")
        
        logging.info("=" * 80)
        logging.info("INITIALIZATION COMPLETE")
        logging.info("=" * 80 + "\n")

    def inference(self, chat):
        if self.args.infer_mode == "api":
            return inference_chat_api(chat, self.args)
        else:
            return inference_chat_local(self.model, self.tokenizer, chat, self.args)

    def save_checkpoint(self, idx):
        """保存训练检查点"""
        checkpoint = {
            'question_idx': idx,
            'kb_path': self.kb_path,
            'timestamp': get_cur_time()
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # 同时保存知识库
        self.knowledge_base.save()
        self.knowledge_base.save_to_json()
        
        logging.info(f"[CHECKPOINT] Saved at index {idx}, KB items: {len(self.knowledge_base.knowledge_items)}")

    def save_kb_version(self, global_idx):
        """保存知识库版本快照"""
        safe_time = get_cur_time().replace(":", "-").replace(" ", "_")
        version_path_pkl = os.path.join(self.kb_version_dir, f"kb_step_{global_idx}_{safe_time}.pkl")
        version_path_json = version_path_pkl.replace('.pkl', '.json')
        
        # 保存 pickle 和 json 版本
        self.knowledge_base.save(version_path_pkl)
        self.knowledge_base.save_to_json(version_path_json)
        
        logging.info(f"[KB VERSION] Saved at step {global_idx}")
        logging.info(f"  PKL: {version_path_pkl}")
        logging.info(f"  JSON: {version_path_json}")
        logging.info(f"  KB size: {len(self.knowledge_base.knowledge_items)} items")

    def load_checkpoint(self):
        """加载训练检查点"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                logging.info(f"[CHECKPOINT] Loaded from index {checkpoint['question_idx']}")
                logging.info(f"[CHECKPOINT] Saved at: {checkpoint.get('timestamp', 'unknown')}")
                return checkpoint
            except Exception as e:
                logging.error(f"[CHECKPOINT] Failed to load: {e}")
                return None
        return None

    def query_process(self, data_item):
        """处理单个问题"""
        grpo_n = self.args.grpo_n

        rollouts = {
            'image': data_item['prompt'][0]['content'][-2]['image'],
            'question': data_item['prompt'][0]['content'][-1]['text'],
            'answer': data_item['answer'],
            'qid': data_item['qid'],
            'img_id': data_item['img_id']
        }
        
        logging.info("-" * 60)
        logging.info("[QUERY PROCESS] Starting")
        logging.info(f"  QID: {rollouts['qid']}, IMG_ID: {rollouts['img_id']}")
        logging.info(f"  Question: {rollouts['question']}")
        logging.info(f"  Ground Truth: {rollouts['answer']}")

        # 1. 使用 VLM 进行分类，获取三元ID
        logging.info("[STEP 1] Classification")
        classification_start = time.time()
        
        qt_ids, mod_ids, org_ids = self.classifier.classify(
            rollouts['question'],
            rollouts['image']
        )
        if qt_ids is None or mod_ids is None or org_ids is None:
            logging.warning("  Classification failed, skipping knowledge retrieval.")
            rollouts['retrieved_knowledge_count'] = 0
            self.rollouts_per_problem.append(rollouts)
            return rollouts
        self.current_triple_ids = {
            'question_type_ids': qt_ids,
            'modality_ids': mod_ids,
            'organ_ids': org_ids
        }
        self.current_triple_id_str = make_triple_id(qt_ids, mod_ids, org_ids)
        
        # 获取标签名用于日志显示
        qt_names, mod_names, org_names = self.classifier.get_tag_names(qt_ids, mod_ids, org_ids)
        
        classification_time = time.time() - classification_start
        logging.info(f"  Triple ID: {self.current_triple_id_str}")
        logging.info(f"  Question Types: {qt_names} (IDs: {qt_ids})")
        logging.info(f"  Modalities: {mod_names} (IDs: {mod_ids})")
        logging.info(f"  Organs: {org_names} (IDs: {org_ids})")
        logging.info(f"  Classification time: {classification_time:.2f}s")

        # 2. 基于三元ID检索知识
        logging.info("[STEP 2] Knowledge Retrieval")
        retrieval_start = time.time()
        
        self.current_retrieved_knowledge = []
        self.formatted_special_knowledges = "None"
        
        kb_size = len(self.knowledge_base.knowledge_items)
        logging.info(f"  Current KB size: {kb_size} items")
        
        if kb_size > 0:
            self.current_retrieved_knowledge = self.knowledge_base.retrieve_knowledge(
                question_type_ids=qt_ids,
                modality_ids=mod_ids,
                organ_ids=org_ids,
                top_n=self.args.topk_knowledge
            )

        retrieval_time = time.time() - retrieval_start
        
        # 格式化知识
        if self.current_retrieved_knowledge:
            formatted_items = []
            logging.info(f"  Retrieved {len(self.current_retrieved_knowledge)} knowledge items:")
            for idx, item in enumerate(self.current_retrieved_knowledge):
                formatted_items.append(f"[{item['id']}]. {item['knowledge']}")
                
                logging.info(f"    [{idx+1}] ID={item['id']}, Triple={item['triple_id']}")
                logging.info(f"        Knowledge: {item['knowledge'][:100]}...")
                
            self.formatted_special_knowledges = "\n\n".join(formatted_items)
        else:
            logging.info("  No relevant knowledge retrieved (KB empty or no match).")
        
        logging.info(f"  Retrieval time: {retrieval_time:.4f}s")

        # 构建 chat
        chat = []
        chat.append({"role": "system", "content": query_system_prompt})

        if self.formatted_special_knowledges != "None":
            prompt = query_prompt(rollouts['question'], self.formatted_special_knowledges)
        else:
            prompt = query_prompt_noKnowledge(rollouts['question'])

        chat.append({
            'role': 'user',
            'content': [
                {"type": "image", "image": rollouts['image']},
                {"type": "text", "text": prompt},
            ]
        })

        # 3. 执行 rollouts
        logging.info(f"[STEP 3] Executing {grpo_n} Rollouts")
        
        for i in range(grpo_n):
            rollout_start = time.time()
            
            response = self.inference(chat)
            rollouts[f'trajectory_{i+1}'] = response

            predict = extract_answer(response)
            rollouts[f'predicted_answer_{i+1}'] = predict
            
            # 使用评估器评估答案
            eval_result = self.evaluator.evaluate(
                rollouts['question'],
                predict,
                rollouts['answer']
            )
            rollouts[f'reward_{i+1}'] = eval_result
            
            rollout_time = time.time() - rollout_start
            
            logging.info(f"  Rollout {i+1}/{grpo_n}:")
            logging.info(f"    Predicted: {predict}")
            logging.info(f"    Ground Truth: {rollouts['answer']}")
            logging.info(f"    Evaluation: {eval_result['result']} - {eval_result['reason']}")
            logging.info(f"    Response length: {len(response)} chars")
            logging.info(f"    Rollout time: {rollout_time:.2f}s")

        self.rollouts_per_problem.append(rollouts)
        
        # 统计本次query的结果
        results_summary = [rollouts[f'reward_{i+1}']['result'] for i in range(grpo_n)]
        logging.info(f"[QUERY PROCESS] Complete - Results: {results_summary}")
        logging.info("-" * 60)
        
        return rollouts

    def summarize_process(self, rollouts):
        """总结每个 trajectory"""
        logging.info("[SUMMARIZE PROCESS] Starting")
        
        summaries = []
        for i in range(self.args.grpo_n):
            summary_start = time.time()
            
            problem = rollouts['question']
            trajectory = rollouts[f'trajectory_{i+1}']
            eval_result = rollouts[f'reward_{i+1}']
            evaluation = f"This trajectory delivers **{eval_result['result']}** answer."
            groundtruth = rollouts['answer']
            pred_answer = rollouts[f'predicted_answer_{i+1}']

            chat = []
            prompt = summarize_prompt(problem, trajectory, evaluation, groundtruth)
            chat.append({
                "role": 'user',
                'content': [
                            {"type": "text", "text": prompt}
                            ]
            })
            
            response = self.inference(chat)
            summaries.append(response)

            rollouts[f'trajectory_summary_{i+1}'] = response
            
            summary_time = time.time() - summary_start
            logging.info(f"  Summary {i+1}/{self.args.grpo_n}:")
            logging.info(f"    Original result: {eval_result['result']}")
            logging.info(f"    Summary: {response[:200]}...")
            logging.info(f"    Summary time: {summary_time:.2f}s")

        logging.info("[SUMMARIZE PROCESS] Complete")
        return summaries
    
    def update_knowledge(self, rollouts):
        """更新知识库（使用当前分类的三元ID）"""
        logging.info("[KNOWLEDGE UPDATE] Starting")
        logging.info(f"  Current Triple ID: {self.current_triple_id_str}")
        
        update_start = time.time()
        # 1. 准备 trajectory summaries
        trajectories_summary_text = ""
        for i in range(self.args.grpo_n):
            eval_result = rollouts[f'reward_{i+1}']
            trajectories_summary_text += f"\n### Trajectory {i+1} (Answer {eval_result['result']}):\n"
            trajectories_summary_text += f"{rollouts[f'trajectory_summary_{i+1}']}\n"
        # 2. 构建更新 prompt
        prompt = update_knowledge_prompt(
            rollouts['question'],
            trajectories_summary_text,
            rollouts['answer'],
            self.formatted_special_knowledges
        )

        chat = []
        chat.append({
                "role": 'user',
                'content': [
                            {"type": "text", "text": prompt}
                            ]
            })
        response = self.inference(chat)
        
        logging.info(f"  LLM Update Response: {response[:300]}...")

        # 3. 解析并执行知识更新
        kb_size_before = len(self.knowledge_base.knowledge_items)
        self._process_knowledge_updates(response, rollouts)
        kb_size_after = len(self.knowledge_base.knowledge_items)
        
        update_time = time.time() - update_start
        
        logging.info(f"  KB size before: {kb_size_before}, after: {kb_size_after}")
        logging.info(f"  Net change: {kb_size_after - kb_size_before}")
        logging.info(f"  Update time: {update_time:.2f}s")
        logging.info("[KNOWLEDGE UPDATE] Complete")

        return response

    def _process_knowledge_updates(self, response: str, rollouts: Dict):
        """解析 LLM 的知识更新指令并执行"""
        try:
            # 提取 JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                updates = json.loads(json_match.group(1))
            else:
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    updates = json.loads(json_match.group())
                else:
                    updates = json.loads(response)
            
            if not isinstance(updates, list):
                logging.warning("  [UPDATE] Parsed JSON is not a list, skipping.")
                return
            
            if len(updates) == 0:
                logging.info("  [UPDATE] No updates requested (empty list).")
                return

            logging.info(f"  [UPDATE] Processing {len(updates)} update operations:")
            
            for idx, update in enumerate(updates):
                option = update.get('option', '').lower()
                
                if option == 'add':
                    knowledge_text = update.get('knowledge', '')
                    if knowledge_text:
                        new_id = self.knowledge_base.add_knowledge(
                            knowledge_text=knowledge_text,
                            question_type_ids=self.current_triple_ids['question_type_ids'],
                            modality_ids=self.current_triple_ids['modality_ids'],
                            organ_ids=self.current_triple_ids['organ_ids'],
                            source=self.args.model_name_or_path
                        )
                        logging.info(f"    [{idx+1}] ADD: ID={new_id}, Triple={self.current_triple_id_str}")
                        logging.info(f"         Content: {knowledge_text[:100]}...")
                
                elif option == 'modify':
                    knowledge_id = self._parse_id(update.get('modified_from'))
                    knowledge_text = update.get('knowledge', '')
                    
                    if knowledge_id is not None:
                        success = self.knowledge_base.modify_knowledge(
                            knowledge_id=knowledge_id,
                            new_knowledge_text=knowledge_text,
                            new_question_type_ids=self.current_triple_ids['question_type_ids'],
                            new_modality_ids=self.current_triple_ids['modality_ids'],
                            new_organ_ids=self.current_triple_ids['organ_ids'],
                            source=self.args.model_name_or_path
                        )
                        logging.info(f"    [{idx+1}] MODIFY: ID={knowledge_id}, Success={success}")
                        logging.info(f"         New content: {knowledge_text[:100]}...")
                
                elif option == 'delete':
                    knowledge_id = self._parse_id(update.get('delete_id'))
                    if knowledge_id is not None:
                        success = self.knowledge_base.delete_knowledge(knowledge_id)
                        logging.info(f"    [{idx+1}] DELETE: ID={knowledge_id}, Success={success}")
                
                else:
                    logging.warning(f"    [{idx+1}] UNKNOWN option: {option}")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.error(f"  [UPDATE] Failed to parse: {e}")
            logging.error(f"  [UPDATE] Response content: {response[:300]}")

    def _parse_id(self, id_val) -> int:
        """解析 ID"""
        if isinstance(id_val, int):
            return id_val
        if isinstance(id_val, str):
            match = re.search(r'\d+', id_val)
            if match:
                return int(match.group())
        return None


def main(args):
    random.seed(args.seed)
    Agent = LLM(args=args)

    if args.exp_name == "slake":
        from utils.datasets import get_slake as process_dataset
    elif args.exp_name == "vqa-rad":
        from utils.datasets import get_vqarad as process_dataset
    else:
        raise ValueError(f"Unsupported experiment name: {args.exp_name}")

    data = process_dataset(truncate=args.dataset_truncate, split=args.data_split, answer_type=args.answer_type, content_type=args.content_type, seed=args.seed)
    
    logging.info("=" * 80)
    logging.info("TRAINING START")
    logging.info("=" * 80)
    logging.info(f"Dataset: {args.exp_name}")
    logging.info(f"Data split: {args.data_split}")
    logging.info(f"Answer type: {args.answer_type}")
    logging.info(f"Content type: {args.content_type}")
    logging.info(f"Dataset size: {len(data)}")
    logging.info(f"Epochs: {args.epochs}")
    logging.info(f"GRPO N: {args.grpo_n}")
    logging.info(f"Top-K Knowledge: {args.topk_knowledge}")
    logging.info(f"KB save steps: {args.kb_save_steps}")
    logging.info("=" * 80 + "\n")

    start_idx = 0
    checkpoint = Agent.load_checkpoint()
    if checkpoint:
        start_idx = checkpoint.get('question_idx', -1) + 1
        logging.info(f"Resuming from checkpoint. Starting at index {start_idx}.")

    # 统计变量
    total_correct = 0
    total_partial = 0
    total_wrong = 0
    total_samples = 0
    
    start_time = time.time()
    len_data = len(data)

    for epoch in range(args.epochs):
        logging.info("\n" + "=" * 80)
        logging.info(f"EPOCH {epoch+1}/{args.epochs}")
        logging.info("=" * 80)
        
        epoch_correct = 0
        epoch_partial = 0
        epoch_wrong = 0
        epoch_start_time = time.time()
        
        for idx, item in enumerate(data):
            global_idx = idx + epoch * len_data
            if global_idx < start_idx:
                continue

            logging.info(f"\n{'#' * 80}")
            logging.info(f"SAMPLE {idx+1}/{len(data)} (Global Index: {global_idx})")
            logging.info(f"{'#' * 80}")
            
            sample_start_time = time.time()

            try:
                # 主流程
                rollouts = Agent.query_process(item)
                summaries = Agent.summarize_process(rollouts)
                update_response = Agent.update_knowledge(rollouts)

                # 统计结果
                for i in range(args.grpo_n):
                    result = rollouts[f'reward_{i+1}']['result']
                    if result == 'correct':
                        epoch_correct += 1
                        total_correct += 1
                    elif result == 'partial_correct':
                        epoch_partial += 1
                        total_partial += 1
                    else:
                        epoch_wrong += 1
                        total_wrong += 1
                total_samples += args.grpo_n

                # 保存检查点
                Agent.save_checkpoint(global_idx)
                
                # 按步数保存知识库版本
                if args.kb_save_steps > 0 and (global_idx + 1) % args.kb_save_steps == 0:
                    Agent.save_kb_version(global_idx)

                sample_time = time.time() - sample_start_time
                
                # 样本完成统计
                logging.info(f"\n[SAMPLE COMPLETE]")
                logging.info(f"  Sample time: {sample_time:.2f}s")
                logging.info(f"  Epoch progress: {idx+1}/{len(data)} ({(idx+1)/len(data)*100:.1f}%)")
                logging.info(f"  Epoch stats - Correct: {epoch_correct}, Partial: {epoch_partial}, Wrong: {epoch_wrong}")
                
                # 计算当前准确率
                epoch_total = epoch_correct + epoch_partial + epoch_wrong
                if epoch_total > 0:
                    epoch_acc = epoch_correct / epoch_total * 100
                    epoch_partial_acc = (epoch_correct + epoch_partial) / epoch_total * 100
                    logging.info(f"  Epoch accuracy: {epoch_acc:.2f}% (strict), {epoch_partial_acc:.2f}% (with partial)")
                
                logging.info(f"  KB size: {len(Agent.knowledge_base.knowledge_items)}")

            except Exception as e:
                logging.error(f"[ERROR] Processing sample {idx}: {e}", exc_info=True)
                Agent.save_checkpoint(global_idx - 1 if global_idx > 0 else -1)
                logging.info("Emergency checkpoint saved.")
                raise

        # Epoch 完成统计
        epoch_time = time.time() - epoch_start_time
        logging.info("\n" + "=" * 80)
        logging.info(f"EPOCH {epoch+1} COMPLETE")
        logging.info("=" * 80)
        logging.info(f"Epoch time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)")
        logging.info(f"Epoch results - Correct: {epoch_correct}, Partial: {epoch_partial}, Wrong: {epoch_wrong}")
        
        epoch_total = epoch_correct + epoch_partial + epoch_wrong
        if epoch_total > 0:
            logging.info(f"Epoch accuracy: {epoch_correct/epoch_total*100:.2f}% (strict)")
            logging.info(f"Epoch accuracy: {(epoch_correct+epoch_partial)/epoch_total*100:.2f}% (with partial)")
        
        logging.info(f"KB size: {len(Agent.knowledge_base.knowledge_items)}")
        logging.info("=" * 80)
        
        start_idx = 0

    # 最终保存
    Agent.knowledge_base.save()
    Agent.knowledge_base.save_to_json()
    # 保存最终版本
    if args.kb_save_steps > 0:
        Agent.save_kb_version(global_idx)

    # 最终统计
    total_time = time.time() - start_time
    
    logging.info("\n" + "=" * 80)
    logging.info("TRAINING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    logging.info(f"Total samples processed: {total_samples}")
    logging.info(f"Total results - Correct: {total_correct}, Partial: {total_partial}, Wrong: {total_wrong}")
    
    if total_samples > 0:
        logging.info(f"Overall accuracy: {total_correct/total_samples*100:.2f}% (strict)")
        logging.info(f"Overall accuracy: {(total_correct+total_partial)/total_samples*100:.2f}% (with partial)")
    
    # 知识库统计
    stats = Agent.knowledge_base.get_statistics()
    logging.info("\n" + "-" * 40)
    logging.info("KNOWLEDGE BASE STATISTICS")
    logging.info("-" * 40)
    logging.info(f"Total Knowledge Items: {stats['total_knowledge']}")
    logging.info(f"Unique Triple IDs: {stats['unique_triple_ids']}")
    logging.info(f"Question Types Used: {stats['question_types_used']}")
    logging.info(f"Modalities Used: {stats['modalities_used']}")
    logging.info(f"Organs Used: {stats['organs_used']}")
    
    if stats['question_type_distribution']:
        logging.info("\nQuestion Type Distribution:")
        for qt, count in sorted(stats['question_type_distribution'].items(), key=lambda x: -x[1]):
            logging.info(f"  {qt}: {count}")
    
    if stats['modality_distribution']:
        logging.info("\nModality Distribution:")
        for mod, count in sorted(stats['modality_distribution'].items(), key=lambda x: -x[1]):
            logging.info(f"  {mod}: {count}")
    
    if stats['organ_distribution']:
        logging.info("\nOrgan Distribution:")
        for org, count in sorted(stats['organ_distribution'].items(), key=lambda x: -x[1]):
            logging.info(f"  {org}: {count}")
    
    logging.info("=" * 80)
    logging.info(f"Log saved to: {Agent.log_file}")
    logging.info(f"Knowledge base saved to: {Agent.kb_path}")
    logging.info(f"KB versions saved to: {Agent.kb_version_dir}")
    logging.info("=" * 80)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-free MedLLM - Tag-based Knowledge Base")
    parser.add_argument("--model_name_or_path", "--model", type=str, default="qwen3-vl-flash")
    parser.add_argument("--evaluator_model", type=str, default="qwen-plus", help="Model for answer evaluation")
    parser.add_argument("--infer_mode", type=str, default="api", choices=["local", "api"], help="Inference mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--api_key", "--ak", type=str, default=API_KEY_list['lijm'], help="API key for inference if using API mode")
    
    parser.add_argument("--data_split", type=str, default="train")
    parser.add_argument("--exp_name", "--en", type=str, default="slake", choices=["vqa-rad", "slake"])
    parser.add_argument("--dataset_truncate", "--dt", type=int, default=300)
    parser.add_argument("--answer_type", "--at", type=str, default="all", choices=["closed", "open", "all"])
    parser.add_argument("--content_type", "--ct", type=str, default="all") #choices=["all", "Modality", "Organ", "Abnormality"...]

    # 知识检索参数
    parser.add_argument("--topk_knowledge", type=int, default=10, help="Top-N knowledge items to retrieve")
    parser.add_argument("--kb_save_steps", type=int, default=50, help="Save KB version every N steps (0 to disable)")

    parser.add_argument("--epochs", "--e", type=int, default=1)
    parser.add_argument("--grpo_n", type=int, default=3)
    parser.add_argument("--rollout_top_p", type=float, default=0.95)
    parser.add_argument("--rollout_temperature", type=float, default=0.7)
    parser.add_argument("--rollout_max_tokens", type=int, default=5120)

    args = parser.parse_args()
    main(args)