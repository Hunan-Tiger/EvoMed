import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class MedicalVisualHyDE:
    def __init__(self, rerank_model_path='BAAI/bge-reranker-v2-m3', device='cuda'):
        self.device = device
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_path).to(device).eval()
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_path)

    def rerank(self, question, visual_findings, candidate_knowledges, threshold=0.8, top_k=10):
        """
        Input: Question + Visual Context + Candidate List
        Output: Top-K Knowledge Items
        """
        if not candidate_knowledges:
            return []
        rich_query = f"Question: {question}\nVisual Context: {visual_findings}"
        pairs = [[rich_query, doc] for doc in candidate_knowledges]

        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
            ).to(self.device)
            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1,).float()

        # Sigmoid step (optional but easier to interpret)
        scores = torch.sigmoid(scores).cpu().numpy()
        
        # Filter by threshold
        final_results = []
        for doc, score in zip(candidate_knowledges, scores):
            if score > threshold:  # Only keep relevant knowledge
                final_results.append(doc)
                
        return final_results[:top_k]
    
if '__main__' == __name__:
    reranker = MedicalVisualHyDE(device='cpu')
    # question = "What is the diagnosis for the patient?"
    # visual_findings = "The chest X-ray shows bilateral infiltrates."
    # candidate_knowledges = [
    #     "Pneumonia is an infection that inflames the air sacs in one or both lungs.",
    #     "Diabetes is a chronic condition that affects how your body turns food into energy.",
    #     "Hypertension is a condition in which the blood pressure in the arteries is persistently elevated."
    # ]
    # top_knowledges = reranker.rerank(question, visual_findings, candidate_knowledges, threshold=0.5, top_k=2)
    # print("Top Relevant Knowledges:")
    # for idx, knowledge in enumerate(top_knowledges):
    #     print(f"{idx+1}. {knowledge}")