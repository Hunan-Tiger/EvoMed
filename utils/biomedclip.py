import os
import torch
import open_clip
import logging
import numpy as np
from PIL import Image
from typing import Union, List, Dict


class BioMedCLIPEncoder:
    def __init__(self, model_path: str = "./biomedclip", device: str = None):
        """初始化 BioMedCLIP 编码器"""
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BioMedCLIP on {self.device}...")

        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            print("BioMedCLIP loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_image_features(self, images: Union[str, Image.Image, List[Union[str, Image.Image]]]) -> torch.Tensor:
        """提取图片特征向量 (经过投影头，用于与文本对齐) [B, 512]"""
        if not isinstance(images, list):
            images = [images]
            
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                img = img.convert('RGB')
            processed_images.append(self.preprocess(img))
            
        image_input = torch.stack(processed_images).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features

    def get_image_key_features(
        self, 
        images: Union[str, Image.Image, List[Union[str, Image.Image]]], 
        num_key_features: int = 10
    ) -> np.ndarray:
        """
        基于范数 + 多样性选取关键视觉向量
        使用贪心算法：每次选取与已选 token 最不相似且范数较大的 token
        
        Args:
            images: 图片路径或 PIL Image
            num_key_features: 要保留的关键向量数量 (默认10)
            
        Returns:
            np.ndarray: [num_key_features, 768] 的关键视觉向量
        """
        if not isinstance(images, list):
            images = [images]
        
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, Image.Image):
                img = img.convert('RGB')
            processed_images.append(self.preprocess(img))
        
        image_input = torch.stack(processed_images).to(self.device)
        
        with torch.no_grad():
            visual = self.model.visual
            
            # BioMedCLIP 使用 TimmModel，需要使用 timm 的接口
            # 获取中间特征（不经过最后的投影层）
            
            # 方法1：使用 forward_features 获取所有 patch tokens
            # TimmModel 的 trunk 是实际的 ViT 模型
            trunk = visual.trunk
            
            # timm ViT 的前向过程
            x = trunk.patch_embed(image_input)  # [B, num_patches, embed_dim]
            
            # 添加 cls token
            cls_token = trunk.cls_token.expand(x.shape[0], -1, -1)  # [B, 1, embed_dim]
            x = torch.cat([cls_token, x], dim=1)  # [B, num_patches + 1, embed_dim]
            
            # 添加位置编码
            x = x + trunk.pos_embed
            x = trunk.pos_drop(x)
            
            # 通过 transformer blocks
            x = trunk.blocks(x)
            x = trunk.norm(x)  # [B, num_patches + 1, embed_dim]
            
            # 提取 patch tokens（排除 CLS token）
            patch_tokens = x[:, 1:, :]  # [B, num_patches, 768]
            
            batch_size = patch_tokens.shape[0]
            key_features = []
            
            for b in range(batch_size):
                tokens = patch_tokens[b]  # [num_patches, 768]
                tokens_norm = tokens / (tokens.norm(dim=-1, keepdim=True) + 1e-8)
                
                # 计算范数作为初始分数
                norms = torch.norm(tokens, dim=-1)  # [num_patches]
                # 例如: 如果 tokens.shape = [196, 768], 则 norms.shape = [196]
                # norms[i] 表示第 i 个 patch token 的 L2 范数
                
                # 贪心选取
                num_patches = tokens.shape[0]
                k = min(num_key_features, num_patches)
                selected_indices = []
                
                # 第一个：选范数最大的
                first_idx = torch.argmax(norms).item()
                selected_indices.append(first_idx)
                
                # 后续：选与已选 token 最不相似且范数较大的
                for _ in range(k - 1):
                    # 计算候选 token 与已选 token 的最大相似度
                    selected_tokens = tokens_norm[selected_indices]  # [num_selected, 768]
                    sim_to_selected = torch.mm(tokens_norm, selected_tokens.T)  # [num_patches, num_selected]
                    max_sim = sim_to_selected.max(dim=1)[0]  # [num_patches]
                    
                    # 综合分数 = 范数（归一化） + 多样性
                    norm_score = norms / (norms.max() + 1e-8)
                    diversity_score = 1 - max_sim
                    combined_score = 0.3 * norm_score + 0.7 * diversity_score
                    
                    # 排除已选的
                    for idx in selected_indices:
                        combined_score[idx] = -float('inf')
                    
                    next_idx = torch.argmax(combined_score).item()
                    selected_indices.append(next_idx)
                
                selected = tokens_norm[selected_indices]
                key_features.append(selected.cpu().numpy())
            
            if len(key_features) == 1:
                return key_features[0]
            
            return np.stack(key_features)

    def get_text_features(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """提取文本特征向量 [B, 512]"""
        if not isinstance(texts, list):
            texts = [texts]
            
        text_input = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """计算余弦相似度 (numpy版本)"""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.dot(a_norm, b_norm.T)


def compute_image_similarity(features_a: np.ndarray, features_b: np.ndarray) -> float:
    """
    计算两组图像关键向量的相似度
    取对角线上的值，去掉最大和最小值后取平均
    
    Args:
        features_a: [N, 768] 查询图像的关键向量
        features_b: [N, 768] 候选图像的关键向量
    
    Returns:
        float: 相似度分数
    """
    sim_matrix = cosine_similarity_np(features_a, features_b)  # [N, N]
    
    # 取对角线元素（取较短边的长度）
    diag_len = min(sim_matrix.shape[0], sim_matrix.shape[1])
    diag_values = np.array([sim_matrix[i, i] for i in range(diag_len)])
    
    # 如果对角线元素少于3个，无法去掉最大最小值，直接返回平均
    if diag_len <= 2:
        return float(np.mean(diag_values))
    
    # 去掉最大值和最小值
    trimmed_values = np.sort(diag_values)[1:-1]  # 排序后去掉首尾
    
    return float(np.mean(trimmed_values))


def HierarchicalKnowledgeFilter(
    query_image: Union[str, Image.Image],
    query_question: str,
    knowledge_base,  # HierarchicalKnowledgeBase instance
    biomedclip_model: BioMedCLIPEncoder,
    top_k_questions: int = 5,
    top_m_images: int = 10,
    top_n_knowledge: int = 5,
    question_sim_threshold: float = 0.6,
    image_sim_threshold: float = 0.5
) -> List[Dict]:
    """
    分层知识检索
    
    检索流程：
    1. 问题 embedding 匹配 -> Top-K 相似问题
    2. 图像关键向量匹配 -> Top-M 相似图像
    3. 关键词投票 -> Top-N 知识
    
    Args:
        query_image: 查询图像
        query_question: 查询问题文本
        knowledge_base: HierarchicalKnowledgeBase 实例
        biomedclip_model: BioMedCLIPEncoder 实例
        top_k_questions: 检索的相似问题数量
        top_m_images: 检索的相似图像数量
        top_n_knowledge: 最终返回的知识数量
        question_sim_threshold: 问题相似度阈值
        image_sim_threshold: 图像相似度阈值
        
    Returns:
        List[Dict]: 包含 'id', 'knowledge', 'keywords' 的字典列表
    """
    from collections import Counter
    
    # 1. 获取查询的 embeddings
    query_q_emb = biomedclip_model.get_text_features(query_question).cpu().numpy()  # [1, 512]
    query_img_features = biomedclip_model.get_image_key_features(query_image, num_key_features=10)  # [10, 768]
    
    # 2. 获取知识库中所有问题的 embeddings
    all_qids, all_q_embeddings = knowledge_base.get_all_question_embeddings()
    
    if len(all_qids) == 0:
        logging.info("Knowledge base is empty, no retrieval performed.")
        return []
    
    # 3. 计算问题相似度
    q_similarities = cosine_similarity_np(query_q_emb, all_q_embeddings).flatten()  # [N]
    
    # 获取 Top-K 并过滤低于阈值的
    sorted_indices = np.argsort(q_similarities)[::-1]  # 降序
    top_k_qids = []
    top_k_sims = []
    
    for i in sorted_indices[:top_k_questions]:
        if q_similarities[i] >= question_sim_threshold:
            top_k_qids.append(all_qids[i])
            top_k_sims.append(q_similarities[i])
    
    if not top_k_qids:
        logging.info(f"No questions found above threshold {question_sim_threshold}")
        return []
    
    logging.info(f"Top-{len(top_k_qids)} similar questions: {list(zip(top_k_qids, [f'{s:.3f}' for s in top_k_sims]))}")
    
    # 4. 获取这些问题对应的所有图像
    candidate_images = []
    for q in knowledge_base.questions:
        if q['qid'] in top_k_qids:
            for img in q['images']:
                candidate_images.append({
                    'qid': q['qid'],
                    'img_id': img['img_id'],
                    'image_key_features': img['image_key_features'],
                    'keywords': img.get('keywords', [])
                })
    
    if not candidate_images:
        logging.info("No images found for top questions.")
        return []
    
    # 5. 计算图像相似度
    image_similarities = []
    for img_record in candidate_images:
        sim = compute_image_similarity(query_img_features, img_record['image_key_features'])
        image_similarities.append((img_record, sim))
    
    image_similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 过滤并取 Top-M
    top_m_images_list = [(img, sim) for img, sim in image_similarities[:top_m_images] if sim >= image_sim_threshold]
    
    if not top_m_images_list:
        logging.info(f"No images above threshold {image_sim_threshold}, returning empty.")
        return []
    
    logging.info(f"Top-{len(top_m_images_list)} similar images selected.")
    
    
    # 6. 关键词投票
    keyword_counter = Counter()
    for img_record, sim in top_m_images_list:
        for kw in img_record.get('keywords', []):
            keyword_counter[kw] += 1
        logging.info(f"  Image {img_record['img_id']} (sim={sim:.3f}): keywords={img_record.get('keywords', [])}")
    
    if not keyword_counter:
        logging.info("No keywords found in selected images.")
        return []
    
    logging.info(f"Top keywords by voting: {keyword_counter.most_common(10)}")
    
    # 7. 根据关键词获取知识
    top_keywords = keyword_counter.most_common(top_n_knowledge)
    knowledge_items = knowledge_base.get_knowledge_by_keywords(top_keywords)
    
    logging.info(f"Retrieved {len(knowledge_items)} knowledge items.")
    for item in knowledge_items:
        logging.info(f"  ID {item['id']}: keywords={item['keywords']}")
    
    return knowledge_items