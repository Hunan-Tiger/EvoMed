import numpy as np
import spacy
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings


# 首次运行需要下载 NLTK 的分词数据
# try:
#     nltk.data.find('metrics/tokenizers/punkt')
# except LookupError:
nltk.download('punkt_tab')


class VQAMetricEvaluator:
    def __init__(self, dataset):
        """
        初始化评估器
        :param dataset: 符合你描述的字典列表格式的数据集
        """
        self.dataset = dataset
        self.questions, self.answers = self._extract_text_data()
        
        # 加载医学 NLP 模型 (用于实体识别)
        # 如果没有安装 en_core_sci_md，可以临时换成 en_core_web_sm (但在医学上不准)
        try:
            print("正在加载医学 NLP 模型 (en_core_sci_md)...")
            self.nlp = spacy.load("en_core_sci_md")
        except OSError:
            print("警告: 未找到 'en_core_sci_md'，回退使用标准英语模型。请务必安装 scispacy 以获得准确的医学实体覆盖率。")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("错误: 请安装 spacy 模型: python -m spacy download en_core_web_sm")
                self.nlp = None

    def _extract_text_data(self):
        """解析数据结构，提取纯文本的问题和答案"""
        qs = []
        as_ = []
        
        for item in self.dataset:
            # 提取 Answer
            as_.append(item.get('answer', '').strip())
            
            # 提取 Question (遍历 content 寻找 type='text')
            q_text = ""
            try:
                # 假设结构固定: prompt -> list -> index 0 -> content -> list
                content_list = item['prompt'][0]['role_content']['content'] if 'role_content' in item['prompt'][0] else item['prompt'][0]['content']
                
                # 你的数据结构里 content 是个列表，包含 image 和 text
                for content_item in content_list:
                    if content_item.get('type') == 'text':
                        q_text = content_item.get('text', '').strip()
                        break
            except (KeyError, IndexError, TypeError) as e:
                # 容错处理，防止数据格式微小差异导致崩溃
                print(f"数据解析警告: {e}")
            
            qs.append(q_text)
            
        return qs, as_

    def compute_avg_sentence_length(self):
        """
        1. 平均句子长度 (Average Sentence Length)
        计算问题的平均单词数
        """
        lengths = [len(word_tokenize(q)) for q in self.questions if q]
        if not lengths: return 0.0
        return np.mean(lengths)

    def compute_medical_entity_coverage(self):
        """
        2. 实体覆盖率 (Medical Entity Coverage/Recall)
        计算：(问题中识别出的医学实体数量) / (问题总单词数)
        反映了问题的含“医”量。
        """
        if self.nlp is None: return 0.0
        
        densities = []
        # 使用 nlp.pipe 批量处理提高速度
        docs = list(self.nlp.pipe(self.questions))
        
        for doc in docs:
            if len(doc) == 0:
                densities.append(0)
                continue
            
            # scispacy 会将医学术语识别为实体
            entity_count = len(doc.ents)
            token_count = len(doc)
            
            # 计算密度：实体数 / 总词数
            density = entity_count / token_count if token_count > 0 else 0
            densities.append(density)
            
        return np.mean(densities)

    def compute_qa_correlation(self):
        """
        3. 问答相关性 (Q-A Correlation)
        这里使用 TF-IDF + 余弦相似度来衡量。
        如果分值过高，说明答案只是在重复问题中的词汇；
        如果分值适中，说明存在逻辑相关性；
        如果分值过低，可能说明答案与问题“驴唇不对马嘴”。
        """
        if not self.questions or not self.answers: return 0.0
        
        # 将问题和答案拼接在一起训练 TF-IDF，保证维度一致
        vectorizer = TfidfVectorizer().fit(self.questions + self.answers)
        
        q_vecs = vectorizer.transform(self.questions)
        a_vecs = vectorizer.transform(self.answers)
        
        # 计算每一对 Q 和 A 的余弦相似度
        # 对角线元素即为对应的 Q-A 相似度
        # 注意：对于海量数据，这里可以用循环避免内存溢出，或者用 batch
        correlations = []
        for i in range(q_vecs.shape[0]):
            # 计算单个向量的点积（因为是归一化的 TF-IDF，点积即余弦相似度）
            sim = cosine_similarity(q_vecs[i], a_vecs[i])[0][0]
            correlations.append(sim)
            
        return np.mean(correlations)

    def generate_report(self):
        print(f"--- 数据集定量分析报告 (样本数: {len(self.dataset)}) ---")
        
        asl = self.compute_avg_sentence_length()
        print(f"1. 平均句子长度 (Avg Length): \t{asl:.4f} words")
        
        mec = self.compute_medical_entity_coverage()
        print(f"2. 医学实体密度 (Entity Density): \t{mec:.4%}")
        
        qac = self.compute_qa_correlation()
        print(f"3. 问答文本相关性 (Q-A Correlation): \t{qac:.4f} (Cosine Sim)")
        
        return {"asl": asl, "mec": mec, "qac": qac}


if __name__ == '__main__':

    import sys
    import os
    # 将项目根目录添加到 Python 路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from utils.dataset import get_omni_rad_closed
    from utils.dataset import get_slake

    omni_data = get_omni_rad_closed(truncate=500, split='train')
    slake_data = get_slake(truncate=500, split='train')

    # 实例化并运行
    # evaluator = VQAMetricEvaluator(your_real_dataset_list) # 传入你的真实变量
    evaluator = VQAMetricEvaluator(omni_data)
    metrics = evaluator.generate_report()
    # metrics 变量中包含了计算得到的各项指标
    print(metrics)

    evaluator_slake = VQAMetricEvaluator(slake_data)
    metrics_slake = evaluator_slake.generate_report()
    print(metrics_slake)