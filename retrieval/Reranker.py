import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging


class MedicalVisualHyDE:
    def __init__(self, rerank_model_path='BAAI/bge-reranker-v2-m3', device='cuda'):
        self.device = device
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_path).to(device).eval()
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)

    def rerank(self, question, visual_findings, current_retrieved_knowledge, threshold=0.8, top_k=10):
        """
        Input: Question + Visual Context + Candidate List
        current_retrieved_knowledge: {id: int, knowledge: str}
        Output: Top-K Knowledge Items
        """
        candicate_knowledges = [item['knowledge'] for item in current_retrieved_knowledge]
        logging.info(f"Reranker received {len(candicate_knowledges)} candidate knowledges:")
        logging.info(candicate_knowledges)

        knowledge_id_map = {item['knowledge']: item['id'] for item in current_retrieved_knowledge}
        if candicate_knowledges == []:
            return []
        rich_query = f"Question: {question}\nVisual Context: {visual_findings}"
        pairs = [[rich_query, doc] for doc in candicate_knowledges]

        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(self.device)
            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1,).float()

        # Sigmoid step (optional but easier to interpret)
        scores = torch.sigmoid(scores).cpu().numpy()
        
        # 1. 将分数和内容打包
        scored_candidates = list(zip(candicate_knowledges, scores))
        
        # 2. 根据分数降序排序 (让最相关的排在前面)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 过滤阈值并去重 (Filter and Deduplicate)
        final_knowledges = []
        seen_knowledge = set()
        scores = []

        for doc, score in scored_candidates:
            if score > threshold:
                if doc not in seen_knowledge:  # 关键：检查是否已存在
                    final_knowledges.append(doc)
                    seen_knowledge.add(doc)
                    scores.append(score)
            
            # 4. 达到 Top-K 数量即停止
            if len(final_knowledges) >= top_k:
                break

        logging.info(f"Reranker filtered {len(candicate_knowledges) - len(final_knowledges)} out of {len(candicate_knowledges)} candidate knowledges.")
        
        # 重构 ID
        final_ids = [knowledge_id_map[doc] for doc in final_knowledges]
        final_result = [{'id': id_, 'knowledge': doc, 'score': score.item()} for id_, doc, score in zip(final_ids, final_knowledges, scores)]
        logging.info(f"Top-{top_k} relevant knowledges: {final_result}")

        return final_result
    
if '__main__' == __name__:
    reranker = MedicalVisualHyDE(device='cuda')
    question = "What is the diagnosis for the patient?"
    visual_findings = "The chest X-ray shows bilateral infiltrates."
    candicate_knowledges = [
        "Pneumonia is an infection that inflames the air sacs in one or both lungs.",
        "Diabetes is a chronic condition that affects how your body turns food into energy.",
        "Hypertension is a condition in which the blood pressure in the arteries is persistently elevated."
    ]
    top_knowledges = reranker.rerank(question, visual_findings, candicate_knowledges, threshold=0.5, top_k=2)
    print("Top Relevant Knowledges:")
    for idx, knowledge in enumerate(top_knowledges):
        print(f"{idx+1}. {knowledge}")