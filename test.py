import random
import time
import json
import re
import os
import argparse
import pickle
import logging
import sys
import numpy as np
from utils.prompt import query_prompt, query_system_prompt
from retrieval.utils import RerietrievalClassifier, AnswerEvaluator, create_evaluator_inference_func, create_classifier_inference_func
from utils.utils import (
    inference_chat_local,
    inference_chat_api,
    get_cur_time,
    extract_answer, 
    API_KEY_list,
    parse_tag,
    inference_chat_api_vllm
)
from retrieval.knowledge_base import (
    TagKnowledgeBase, 
    make_triple_id, 
    ids_to_tags,
    ID_TO_QUESTION_TYPE, 
    ID_TO_MODALITY, 
    ID_TO_ORGAN
)
from retrieval.Reranker import MedicalVisualHyDE


#
# query_system_prompt = rf'''You are an expert Medical AI Assistant. Your task is to analyze medical images and answer clinical questions.
# You have access to an external Knowledge Base (if any) to assist in your diagnosis.

# Output format:
# <Reasoning> Your step-by-step diagnostic reasoning process. </Reasoning>
# <Answer> yes/no </Answer>

# Note: Use the knowledge as a reference, but prioritize the visual evidence in the medical image.
# '''


class LLM():
    def __init__(self, args):
        args.api_key = API_KEY_list.get(args.api_key)
        self.model_name_or_path = args.model_name_or_path
        self.args = args
        
        # 1. 日志配置
        os.makedirs(f'logs/{self.args.exp_name}', exist_ok=True)
        safe_time = get_cur_time().replace(":", "-").replace(" ", "_")
        self.log_file = f'logs/{self.args.exp_name}/test_Know({self.args.give_knowledges})_{safe_time}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logging.info("=" * 80)
        logging.info("TEST INITIALIZATION")
        logging.info("=" * 80)
        logging.info(f"Arguments: {self.args}")
        logging.info(f"Log file: {self.log_file}")
        
        # 2. 检查点配置
        os.makedirs('checkpoints', exist_ok=True)
        self.checkpoint_file = f'checkpoints/{self.args.exp_name}_test_Know({self.args.give_knowledges})_{self.model_name_or_path}.pkl'
        logging.info(f"Checkpoint file: {self.checkpoint_file}")
        
        self.rollouts_per_problem = []
        
        # 3. 初始化知识库相关变量
        self.knowledge_base = None
        self.classifier = None
        self.current_retrieved_knowledge = []
        self.current_triple_ids = {
            'question_type_ids': [],
            'modality_ids': [],
            'organ_ids': []
        }
        self.current_triple_id_str = ""
        self.formatted_special_knowledges = ""
        self.total_tokens_consumed = 0

        self.reranker = MedicalVisualHyDE(device='cuda')
        
        # 4. 加载推理模型 (仅在 local 模式下)
        if self.args.infer_mode != "api":
            logging.info(f"Loading local model: {self.model_name_or_path}")
        else:
            logging.info(f"Using API mode with model: {self.model_name_or_path}")
        
        # 5. 仅在 give_knowledges=true 时初始化知识库
        if self.args.give_knowledges.lower() == "true":
            # 如果你要指定特定路径，可以在这里修改
            self.kb_path = self.args.kb_path
            if os.path.exists(self.kb_path):
                self.knowledge_base = TagKnowledgeBase(kb_path=self.kb_path)
                logging.info(f"Tag Knowledge Base loaded from: {self.kb_path}")
                
                # 统计信息
                stats = self.knowledge_base.get_statistics()
                logging.info(f"  Knowledge Items: {stats['total_knowledge']}")
                logging.info(f"  Unique Triple IDs: {stats['unique_triple_ids']}")
            else:
                logging.warning(f"Knowledge base not found at: {self.kb_path}")
        else:
            logging.info("Knowledge retrieval disabled (give_knowledges=false).")
        
        # 初始化分类器
        classifier_inference_func = create_classifier_inference_func(self.args)
        self.classifier = RerietrievalClassifier(classifier_inference_func, self.args)
        logging.info(f"Rerietrieval Classifier initialized with model: {self.args.classifier_model}")
        # 初始化Judger
        eval_inference_func = create_evaluator_inference_func(self.args)
        from utils.utils import Test_Judger
        self.evaluator = Test_Judger(eval_inference_func)
        logging.info(f"Answer Evaluator initialized with model: {self.args.evaluator_model}")
        
        logging.info("=" * 80)
        logging.info("INITIALIZATION COMPLETE")
        logging.info("=" * 80 + "\n")
    

    def inference(self, chat):
        if self.args.infer_mode == "api":
            return inference_chat_api_vllm(chat, self.args)
        else:
            return inference_chat_local(chat, self.args)
    

    def save_checkpoint(self, idx, all_metrics, failed_indices):
        """保存检查点"""
        checkpoint = {
            'idx': idx,
            'all_metrics': all_metrics,
            'failed_indices': failed_indices,
            'rollouts': self.rollouts_per_problem,
            'timestamp': get_cur_time(),
            'total_tokens_consumed': self.total_tokens_consumed
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        logging.info(f"[CHECKPOINT] Saved at index {idx}")
    
    def load_checkpoint(self):
        """加载检查点"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint = pickle.load(f)
                logging.info(f"[CHECKPOINT] Loaded from index {checkpoint['idx']}")
                logging.info(f"[CHECKPOINT] Saved at: {checkpoint.get('timestamp', 'unknown')}")
                return checkpoint
            except Exception as e:
                logging.error(f"[CHECKPOINT] Failed to load: {e}")
        return None
    
    def query_process(self, data_item):
        """处理单个问题，生成一个rollout"""
        
        question = data_item['prompt'][0]['content'][-1]['text']
        query_img = data_item['prompt'][0]['content'][-2]['image']
        answer = data_item['answer']
        qid = data_item.get('qid', 'unknown')
        img_id = data_item.get('img_id', 'unknown')
        
        rollouts = {
            'question': question, 
            'answer': answer,
            'qid': qid,
            'img_id': img_id,
            'image': query_img
        }
        
        logging.info("-" * 60)
        logging.info("[QUERY PROCESS] Starting")
        logging.info(f"  QID: {qid}, IMG_ID: {img_id}")
        logging.info(f"  Question: {question}")
        logging.info(f"  Ground Truth: {answer}")
        
        # 重置知识检索状态
        self.current_retrieved_knowledge = []
        self.formatted_special_knowledges = ""
        self.current_triple_id_str = ""
        
        # 1. 使用 VLM 进行分类，获取三元ID
        logging.info("[STEP 1] Classification")
        classification_start = time.time()
        # 仅在 give_knowledges=true 时进行知识检索
        if self.args.give_knowledges.lower() == "true":
            (qt_ids, mod_ids, org_ids, rollouts['visual_findings']), classify_tokens = self.classifier.classify(question, query_img)
            if (self.knowledge_base is not None 
                and len(self.knowledge_base.knowledge_items) > 0
                and self.classifier is not None):
                
                if classify_tokens is not None:
                    self.total_tokens_consumed += classify_tokens
                if qt_ids is None or mod_ids is None or org_ids is None:
                    logging.info(f"  Received invalid classification IDs: QT={qt_ids}, MOD={mod_ids}, ORG={org_ids}")
                    logging.info(f"  <Visual Findings>: {rollouts['visual_findings']}")
                    logging.warning("  Classification failed, skipping knowledge retrieval.")
                    rollouts['retrieved_knowledge_count'] = 0
                    self.rollouts_per_problem.append(rollouts)
                else:
                    self.current_triple_ids = {
                        'question_type_ids': qt_ids,
                        'modality_ids': mod_ids,
                        'organ_ids': org_ids
                    }
                    self.current_triple_id_str = make_triple_id(qt_ids, mod_ids, org_ids)
                    
                    # 获取标签名用于日志显示
                    qt_names = ids_to_tags(qt_ids, ID_TO_QUESTION_TYPE)
                    mod_names = ids_to_tags(mod_ids, ID_TO_MODALITY)
                    org_names = ids_to_tags(org_ids, ID_TO_ORGAN)
                    
                    classification_time = time.time() - classification_start
                    logging.info(f"  Triple ID: {self.current_triple_id_str}")
                    logging.info(f"  Question Types: {qt_names} (IDs: {qt_ids})")
                    logging.info(f"  Modalities: {mod_names} (IDs: {mod_ids})")
                    logging.info(f"  Organs: {org_names} (IDs: {org_ids})")
                    logging.info(f"  <Visual Findings>: {rollouts['visual_findings']}")
                    logging.info(f"  Classification time: {classification_time:.2f}s")
                    
                    rollouts['triple_id'] = self.current_triple_id_str
                    rollouts['question_types'] = qt_names
                    rollouts['modalities'] = mod_names
                    rollouts['organs'] = org_names
                    
                    # 2. 基于三元ID检索知识
                    logging.info("[STEP 2] Knowledge Retrieval")
                    retrieval_start = time.time()
                    
                    self.current_retrieved_knowledge = self.knowledge_base.retrieve_knowledge(
                        question_type_ids=qt_ids,
                        modality_ids=mod_ids,
                        organ_ids=org_ids,
                    )
                    
                    # 格式化知识
                    self.current_rerank_knowledge = self.reranker.rerank(
                        question,
                        rollouts['visual_findings'],
                        self.current_retrieved_knowledge,
                        threshold=self.args.reranker_threshold,
                        top_k=self.args.top_k_knowledges
                    )
                    if self.current_rerank_knowledge:
                        formatted_items = []
                        logging.info(f"  Retrieved {len(self.current_rerank_knowledge)} knowledge items:")
                        for idx, item in enumerate(self.current_rerank_knowledge):
                            formatted_items.append(f"[{item['id']}]. {item['knowledge']}")
                            
                            logging.info(f"  [{idx+1}] ID={item['id']}  Knowledge: {item['knowledge']}")
                            
                        self.formatted_special_knowledges = "\n".join(formatted_items)
                    else:
                        logging.info("  No relevant knowledge retrieved.")

                    retrieval_time = time.time() - retrieval_start
                    logging.info(f"  Retrieval time: {retrieval_time:.4f}s")
                    rollouts['retrieved_knowledge_count'] = len(self.current_rerank_knowledge)
            else:
                logging.info("  Knowledge base empty or classifier not initialized.")
                rollouts['retrieved_knowledge_count'] = 0
        else:
            # give_knowledges=false，跳过知识检索
            # logging.info(f"  <Visual Findings>: {rollouts['visual_findings']}")
            logging.info("  Knowledge retrieval skipped (give_knowledges=false).")
            rollouts['retrieved_knowledge_count'] = 0

       # 构建 chat
        prompt = query_prompt(rollouts['question'], self.formatted_special_knowledges)
        chat = []
        chat.append({"role": "system", "content": query_system_prompt})
        chat.append({
            'role': 'user',
            'content': [
                {"type": "image", "image": rollouts['image']},
                {"type": "text", "text": prompt},
            ]
        })

        if self.args.model_name_or_path == 'Qwen2.5-VL-7B-Instruct':
            instruction_text = rf'''You are an expert Medical AI Assistant. Your task is to analyze medical images and answer clinical questions.
Analyze the medical image and the provided knowledge (if any) to answer the question.

Output format:
<Reasoning> Your step-by-step diagnostic reasoning process. </Reasoning>
<Answer> Yes. or No. or option (e.g., A, B, C, D) or organ name. etc. </Answer>

Note: Use the knowledge as a reference, but prioritize the visual evidence in the medical image.

'''
            chat = []
            chat.append({
            'role': 'user',
            'content': [
                {"type": "text", "text": instruction_text},
                {"type": "image", "image": rollouts['image']},
                {"type": "text", "text": prompt},
            ]
        })
        
        # 执行 rollouts
        logging.info(f"[STEP {'3' if self.args.give_knowledges.lower() == 'true' else '1'}] Executing Rollout")
        
        rollout_start = time.time()
        
        response = self.inference(chat)
        if isinstance(response, tuple) and len(response) == 2:
            content, usage = response
            response = content
        if usage and hasattr(usage, 'total_tokens') and usage.total_tokens is not None:
            self.total_tokens_consumed += usage.total_tokens

        # 解析 XML 标签内容
        rollouts['reasoning'] = parse_tag(response, "Reasoning")
        rollouts['predicted_answer'] = parse_tag(response, "Answer")
        rollouts['trajectory'] = response
        logging.info(f"  Full Response: {rollouts['trajectory']}")

        # 使用Test_Judger评估答案
        eval_result, eval_tokens = self.evaluator.evaluate(
            rollouts['question'],
            response,
            rollouts['answer']
        )
        if eval_tokens is not None:
            self.total_tokens_consumed += eval_tokens
        rollouts[f'reward'] = eval_result
        
        rollout_time = time.time() - rollout_start
        
        logging.info(f"    Predicted: {rollouts[f'predicted_answer']}")
        logging.info(f"    Ground Truth: {rollouts['answer']}")
        logging.info(f"    Evaluation: {eval_result['result']} - {eval_result['reason']}")
        logging.info(f"    Response length: {len(response)} chars")
        logging.info(f"    Rollout time: {rollout_time:.2f}s")
        
        self.rollouts_per_problem.append(rollouts)
        
        # 统计本次query的结果
        results_summary = rollouts[f'reward']['result']
        logging.info(f"[QUERY PROCESS] Complete - Results: {results_summary}")
        logging.info("-" * 60)
        
        return rollouts


def main(args):
    random.seed(args.seed)
    Agent = LLM(args=args)
    
    if args.exp_name == "slake":
        from utils.dataset import get_slake as process_dataset
    elif args.exp_name == "vqa-rad":
        from utils.dataset import get_vqarad as process_dataset
    elif args.exp_name == "omni-rad-closed":
        from utils.dataset import get_omni_rad_closed as process_dataset
    elif args.exp_name == "pathvqa":
        from utils.dataset import get_pathvqa as process_dataset
    else:
        raise ValueError(f"Unsupported experiment name: {args.exp_name}")

    data = process_dataset(truncate=args.dataset_truncate, split=args.data_split, answer_type=args.answer_type, content_type=args.content_type, seed=args.seed)

    logging.info("=" * 80)
    logging.info("TESTING START")
    logging.info("=" * 80)
    logging.info(f"Model: {args.model_name_or_path}")
    logging.info(f"Dataset: {args.exp_name}")
    logging.info(f"Data split: {args.data_split}")
    logging.info(f"Answer type: {args.answer_type}")
    logging.info(f"Content type: {args.content_type}")
    logging.info(f"Dataset size: {len(data)}")
    logging.info(f"hasKnowledge: {args.give_knowledges}")
    logging.info(f"Knowledge Base Path: {args.kb_path if args.give_knowledges.lower() == 'true' else 'N/A'}")
    logging.info(f"GRPO N: {args.grpo_n}")
    logging.info(f"Temperature: {args.rollout_temperature}")
    logging.info(f"Reranker Theshold: {args.reranker_threshold}")
    logging.info(f"Top-K Knowledges: {args.top_k_knowledges}")
    logging.info(f"Evaluator Model: {args.evaluator_model}")
    logging.info("=" * 80 + "\n")
    
    start_time = time.time()
    all_predictions = []
    all_ground_truths = []
    all_metrics = []

    failed_indices = []

    start_idx = args.start_idx
    
    # 尝试加载检查点
    checkpoint = Agent.load_checkpoint()
    if checkpoint:
        start_idx = checkpoint['idx'] + 1
        all_metrics = checkpoint['all_metrics']
        failed_indices = checkpoint['failed_indices']
        Agent.rollouts_per_problem = checkpoint['rollouts']
        Agent.total_tokens_consumed = checkpoint.get('total_tokens_consumed', 0)

        logging.info(f"Resuming from checkpoint at index {start_idx}")
        if args.answer_type != 'open':
            total_correct = sum(1 for m in all_metrics if m == 'correct')
            total_wrong = sum(1 for m in all_metrics if m == 'wrong')
            logging.info(f"  Loaded stats - Correct: {total_correct}, Wrong: {total_wrong}, ACC: {total_correct/(total_correct+total_wrong+0.00001)*100:.2f}%")
        else:
            avg_recall = sum(all_metrics) / len(all_metrics) if all_metrics else 0
            logging.info(f"  Loaded stats - Average Recall: {avg_recall:.4%} over {len(all_metrics)} samples")



    for idx, item in enumerate(data):
        if idx < start_idx:
            continue
        try:
            logging.info(f"\n{'#' * 80}")
            logging.info(f"SAMPLE {idx+1}/{len(data)}")
            logging.info(f"{'#' * 80}")
            
            sample_start_time = time.time()
            
            rollouts = Agent.query_process(item)
            
            predict = rollouts[f'predicted_answer']
            reward = rollouts[f'reward']['result']
            
            all_predictions.append(predict)
            all_ground_truths.append(item['answer'])

            # 统计评估结果
            if args.answer_type == 'open':
                from utils.utils import calculate_word_recall
                # 开放式回答，计算词汇召回率
                pred = rollouts[f'predicted_answer']
                gt = rollouts['answer']
                recall_score = calculate_word_recall(gt, pred)
                all_metrics.append(recall_score)
                # 开放式回答，设定召回率低于0.6为失败
                if recall_score < 0.6:
                    failed_indices.append(idx)
            else:
                if reward == 'correct':
                    all_metrics.append('correct')
                else:
                    failed_indices.append(idx)
                    all_metrics.append('wrong')
            
            
            sample_time = time.time() - sample_start_time
            
            logging.info(f"\n[SAMPLE COMPLETE]")
            logging.info(f"  Sample time: {sample_time:.2f}s")
            logging.info(f"  Current Sample Metrics: {all_metrics[-1]}")
            logging.info(f"  Progress: {idx+1}/{len(data)} ({(idx+1)/len(data)*100:.1f}%)")
            logging.info(f"  Failed Samples So Far: {[i+1 for i in failed_indices]}")
            
            # 计算累计准确率
            if args.answer_type != 'open':
                total_correct = sum(1 for m in all_metrics if m == 'correct')
                total_wrong = sum(1 for m in all_metrics if m == 'wrong')
                total_samples = total_correct + total_wrong
                strict_acc = total_correct / total_samples * 100 if total_samples > 0 else 0
                logging.info(f"  Cumulative Accuracy: {strict_acc:.2f}% ({total_correct}/{total_samples})")
            else:
                avg_recall = sum(all_metrics) / len(all_metrics) if all_metrics else 0
                logging.info(f"  Cumulative Average Recall: {avg_recall:.4%} ({sum(all_metrics):.4f} total recall over {len(all_metrics)} samples)")

            # 定期保存
            if (idx + 1) % args.save_interval == 0:
                Agent.save_checkpoint(idx, all_metrics, failed_indices)

        except Exception as e:
            logging.error(f"[ERROR] Processing sample {idx}: {e}", exc_info=True)
            Agent.save_checkpoint(idx, all_metrics, failed_indices)
            logging.info(f"Error occurred, checkpoint saved. Resume from index {idx}")
            raise
    
    # 最终保存
    # Agent.save_checkpoint(idx, all_metrics, failed_indices)
    
    total_time = time.time() - start_time
    
    logging.info("\n" + "=" * 80)
    logging.info("TESTING COMPLETE")
    logging.info("=" * 80)
    logging.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    logging.info(f"Total samples: {len(data)}")
    logging.info(f"Total tokens consumed: {Agent.total_tokens_consumed}")
    
    logging.info("\n" + "-" * 40)
    if args.answer_type != 'open':
        logging.info("ACCURACY METRICS")
        logging.info("-" * 40)
        logging.info(f"  Correct: {total_correct}")
        logging.info(f"  Wrong: {total_wrong}")
        acc = total_correct / (total_correct+total_wrong) * 100
        logging.info(f"  Accuracy: {acc:.2f}%")
    else:
        logging.info("RECALL METRICS")
        logging.info("-" * 40)
        avg_recall = sum(all_metrics) / len(all_metrics) if all_metrics else 0
        logging.info(f"  Average Recall: {avg_recall:.4%} over {len(all_metrics)} samples")
    
    logging.info("\n" + "-" * 40)
    logging.info("FAILED SAMPLES")
    logging.info("-" * 40)
    logging.info(f"  Total Failed (not correct): {len(failed_indices)}")
    logging.info(f"  Failed Sample IDs: {[i+1 for i in failed_indices]}")
    
    logging.info("=" * 80)
    
    # 保存结果 JSON
    if args.answer_type != 'open':
        eval_={
            'correct': total_correct,
            'wrong': total_wrong,
            'accuracy': total_correct / (total_correct+total_wrong+0.00001) * 100
        }
    else:
        eval_={
            'average_recall': sum(all_metrics) / len(all_metrics) if all_metrics else 0,
        }
    results = {
        'random_seed': args.seed,
        'model': args.model_name_or_path,
        'dataset': args.exp_name,
        'data_split': args.data_split,
        'data_truncate': args.dataset_truncate,
        'answer_type': args.answer_type,
        'content_type': args.content_type,
        'temperature': args.rollout_temperature,
        'knowledge_base': args.kb_path if args.give_knowledges.lower() == "true" else None,
        'grpo_n': args.grpo_n,
        'give_knowledges': args.give_knowledges,
        'evaluator_model': args.evaluator_model,
        'evaluation': eval_,
        'failed_idx': failed_indices,
        'failed_count': len(failed_indices),
        'total_time_seconds': total_time,
        'total_tokens_consumed': Agent.total_tokens_consumed
    }
    
    os.makedirs('results', exist_ok=True)
    safe_time = get_cur_time().replace(":", "-").replace(" ", "_")
    results_file = f'results/{args.exp_name}_test_results_Know({args.give_knowledges})_{safe_time}.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logging.info(f"Results saved to: {results_file}")
    

    # 保存详细rollouts（可选）
    if args.save_rollouts:
        rollouts_file = f'results/{args.exp_name}_test_rollouts_Know({args.give_knowledges})_{safe_time}.json'
        # 转换为可序列化格式（移除图像数据）
        serializable_rollouts = []
        for r in Agent.rollouts_per_problem:
            r_copy = {k: v for k, v in r.items() if k != 'image'}
            serializable_rollouts.append(r_copy)
        with open(rollouts_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_rollouts, f, indent=2, ensure_ascii=False)
        logging.info(f"Rollouts saved to: {rollouts_file}")
    
    logging.info(f"Log saved to: {Agent.log_file}")


    # 删除检查点文件
    if os.path.exists(Agent.checkpoint_file):
        os.remove(Agent.checkpoint_file)
        logging.info(f"Checkpoint file deleted: {Agent.checkpoint_file}")
    
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training-free MedLLM - Tag-based Knowledge Base")
    parser.add_argument("--model_name_or_path", "--model", type=str, default="qwen2.5-vl-instruct")
    parser.add_argument("--evaluator_model", "--em", type=str, default="qwen2.5-vl-instruct", help="Model for answer evaluation")
    parser.add_argument("--classifier_model", "--cm", type=str, default="qwen2.5-vl-instruct", help="Model for retrieval classification")
    parser.add_argument("--infer_mode", type=str, default="api", choices=["local", "api"], help="Inference mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--api_key", "--ak", type=str, default='skd', help="API key for inference if using API mode")
    parser.add_argument("--base_url", "--bu", type=str, default="https://genaiapi.shanghaitech.edu.cn/api/v1/start", help="API base URL for inference if using API mode")

    parser.add_argument("--exp_name", "--en", type=str, default="slake", help="Experiment name")
    parser.add_argument("--data_split", type=str, default="test", help="Dataset split")
    parser.add_argument("--dataset_truncate", "--dt", type=int, default=50, help="Truncate dataset")
    parser.add_argument("--answer_type", "--at", type=str, default="all", choices=["closed", "open", "all"])
    parser.add_argument("--content_type", "--ct", type=str, default="all") 

    # 标签知识库参数
    parser.add_argument("--kb_path", "--kb", type=str, default="knowledges/slake_tag_kb.pkl", help="Path to tag knowledge base")

    parser.add_argument("--grpo_n", type=int, default=1, help="Number of rollouts per sample")
    parser.add_argument("--rollout_top_p", type=float, default=0.01, help="Top-p sampling")
    parser.add_argument("--rollout_repetition_penalty", '--rp', type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--rollout_no_repeat_ngram_size", '--nrg', type=int, default=0, help="No repeat ngram size")
    parser.add_argument("--rollout_temperature", '--temp', type=float, default=0, help="Temperature")
    parser.add_argument("--rollout_max_tokens", type=int, default=5120, help="Max tokens")
 
    parser.add_argument("--reranker_threshold", '--ts', type=float, default=0.8, help="Reranker relevance threshold")
    parser.add_argument("--top_k_knowledges", '--topk', type=int, default=1000, help="Top K knowledges to retrieve")
    
    parser.add_argument("--give_knowledges", "--gk", type=str, default="true", help="Whether to provide knowledge")
    
    parser.add_argument("--start_idx", "--si", type=int, default=0, help="Starting index")
    parser.add_argument("--save_interval", type=int, default=5, help="Save checkpoint interval")
    parser.add_argument("--save_rollouts", action="store_true", help="Save detailed rollouts to JSON")
    
    args = parser.parse_args()
    
    main(args)