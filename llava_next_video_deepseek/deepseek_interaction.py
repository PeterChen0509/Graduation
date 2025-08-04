import os
import time
import torch
import numpy as np
import pandas as pd
import faiss
import pickle
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from datetime import datetime
import copy
import argparse
import logging
import json
import signal
import traceback
from multiprocessing import Pool, set_start_method, Queue
import builtins
import threading
import transformers

# 禁用OpenAI客户端HTTP请求日志
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)

# 设置多进程启动方法为spawn，以避免CUDA在fork子进程中初始化的问题
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 在某些情况下可能已经设置过了

sys.path.append("/home/peterchen/M2/contriever")
from src.contriever import Contriever

# 配置
MAX_ROUNDS = 5  # 最大交互轮次
TOP_K = 10      # 检索结果数量/成功阈值（当排名<=此值时提前结束）
OPENAI_BASE_URL = "https://api.deepseek.com"
OPENAI_API_KEY = "sk-3e3bc603ac9647af91919f2cd57efbf5"
LLM_MODEL = "deepseek-chat"

# 全局输出队列
output_queue = Queue()

# 自定义打印函数，将进程ID添加到每行前面
def process_print(msg, process_id=None):
    """进程安全的打印函数，将输出送入队列"""
    if process_id is not None:
        prefix = f"[进程 {process_id}] "
    else:
        prefix = ""
    output_queue.put(prefix + msg)

def dummy_print(*args, **kwargs):
    """空打印函数，用于临时禁止打印"""
    pass

transformers.logging.set_verbosity_error()  # 只显示错误，忽略警告

class InteractiveVideoRetrievalSystem:
    def __init__(self, caption_file, faiss_index_path=None, id2video_path=None):
        # 初始化交互式视频检索系统
        self.caption_df = pd.read_excel(caption_file)
        
        # 检查是否存在必要的列
        if "video_name" not in self.caption_df.columns or "model_caption" not in self.caption_df.columns:
            raise ValueError("Excel 文件中必须包含 'video_name' 和 'model_caption' 列")
        
        #  过滤掉空的 caption
        self.caption_df = self.caption_df.dropna(subset=['model_caption'])
        
        # 创建 video_name 到 caption 的映射
        self.video_to_caption = dict(zip(self.caption_df['video_name'], self.caption_df['model_caption']))
        print(f"加载了 {len(self.video_to_caption)} 个视频的 caption")
        
        # 初始化 Contriever 模型和 tokenizer
        self.contriever, self.tokenizer = self.load_contriever()
        
        # 创建或加载 FAISS 索引
        self.faiss_index_path = faiss_index_path
        self.id2video_path = id2video_path
        
        if faiss_index_path and id2video_path and os.path.exists(faiss_index_path) and os.path.exists(id2video_path):
            print(f"加载 FAISS 索引: {faiss_index_path}")
            self.faiss_index = faiss.read_index(faiss_index_path)
            with open(id2video_path, 'rb') as f:
                self.id2video = pickle.load(f)
            
            # 反向创建 video 到 id 的映射
            self.video2id = {v:k for k, v in self.id2video.items()}
            print(f"成功加载 FAISS 索引，包含 {self.faiss_index.ntotal} 个向量")
        else:
            print("创建新的 FAISS 索引")
            self.id2video, self.video2id, self.faiss_index = self.build_faiss_index()
            
            # 保存索引和映射
            if faiss_index_path and id2video_path:
                os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
                os.makedirs(os.path.dirname(id2video_path), exist_ok=True)
                
                print(f"保存 FAISS 索引到: {faiss_index_path}")
                faiss.write_index(self.faiss_index, faiss_index_path)
                with open(id2video_path, "wb") as f:
                    pickle.dump(self.id2video, f)
                print("索引保存完成")
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
             api_key=OPENAI_API_KEY,
             base_url=OPENAI_BASE_URL,
        )
    
    def load_contriever(self):
        # 加载 Contriever 模型和 tokenizer
        print("加载 Contriever 模型和 tokenizer")
        model = Contriever.from_pretrained("facebook/contriever")
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        
        model.eval()
        # 转移到当前进程使用的GPU
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()  # 获取当前进程设置的GPU ID
            print(f"将Contriever模型移动到GPU {current_device}")
            model = model.to(current_device)  # 明确指定GPU设备
        return model, tokenizer
        
    def encode_text(self, texts):
        # 使用 Contriever 编码文本
        if isinstance(texts, str):
            texts = [texts]
        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return np.array([])

        # 使用 tokenizer 编码
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # 把输入转移到与模型相同的设备
        if torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            inputs = {k: v.to(current_device) for k, v in inputs.items()}
        
        # 使用模型生成嵌入
        with torch.no_grad():
            embeddings = self.contriever(**inputs)
            
            # 将张量移到CPU并转换为NumPy数组
            embeddings = embeddings.cpu().numpy()
            
            # 归一化向量（用于余弦相似度计算）
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def build_faiss_index(self):
        # 构建 FAISS 索引并返回 id2video 映射
        video_names = list(self.video_to_caption.keys())
        captions = list(self.video_to_caption.values())
        print(f"正在对{len(captions)}个视频进行编码...")
        
        # 批量编码
        batch_size = 128
        all_embeddings = []
        
        for i in tqdm(range(0, len(captions), batch_size)):
            batch_captions = captions[i:i+batch_size]
            batch_embeddings = self.encode_text(batch_captions)
            if len(batch_embeddings) > 0:
                all_embeddings.append(batch_embeddings)
        
        # 合并所有批次的嵌入
        if not all_embeddings:
            raise ValueError("无法编码任何 caption, 请检查数据")

        embeddings = np.vstack(all_embeddings)
        
        # 创建 ID 到视频的映射
        id2video = {i: video_names[i] for i in range(len(video_names))}
        video2id = {v: k for k,v in id2video.items()}
        
        # 创建 FAISS 索引
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        
        return id2video, video2id, index
    
    def retrieve_videos(self, query, k=TOP_K, query_embedding=None):
        # 检索与查询最相似的前 k 个视频
        if query_embedding is None:
            query_embedding = self.encode_text([query])
        
        if len(query_embedding) == 0:
            print(f"警告: 无法编码查询：{query}")
            return []

        # 检索最相似的视频
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # 转换为结果列表
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.id2video):
                video_id = self.id2video[idx]
                score = float(distances[0][i])
                caption = self.video_to_caption.get(video_id, "")
                results.append({
                    "video_id": video_id,
                    "score": score,
                    "caption": caption,
                })
        
        return results
    
    def find_target_rank(self, target_video_id, query_embedding, top_captions=None, top_k=TOP_K):
        """找出目标视频在检索结果中的排名"""
        if target_video_id is None:
            return None
        
        # 如果传入的不是查询嵌入，则返回None
        if not isinstance(query_embedding, np.ndarray):
            print(f"错误：需要传入查询嵌入向量进行全局搜索")
            return None
        
        try:
            # 获取索引中的总视频数量
            total_videos = self.faiss_index.ntotal
            
            # 搜索整个索引
            distances, indices = self.faiss_index.search(query_embedding, total_videos)
            all_video_ids = [self.id2video[idx] for idx in indices[0] if idx >= 0 and idx < len(self.id2video)]
            
            # 在完整结果中查找目标ID（完全匹配）
            if target_video_id in all_video_ids:
                global_rank = all_video_ids.index(target_video_id) + 1
                return global_rank
            
            return None
            
        except Exception as e:
            print(f"搜索全部索引时出错: {e}")
            return None
    
    def _clean_generated_text(self, text):
        """清理生成的文本，去除不必要的格式和说明"""
        if not text:
            return text
            
        # 去除多余的标记和前缀
        text = text.strip()
        text = text.replace("**", "")
        
        # 去除常见的前缀
        prefixes = ["Question:", "Answer:", "Updated Query:", "Query:", "Response:"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # 去除引号
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        
        # 去除解释部分
        explanation_markers = ["###", "Key Changes:", "Explanation:", "Note:"]
        for marker in explanation_markers:
            if marker in text:
                text = text.split(marker)[0].strip()
        
        return text
    
    def generate_question(self, query, top_k_videos, conversation_history=None, target_caption=None):
        # 生成关于视频的问题
        # Step 1: 判断是否是第一轮交互
        round_num = len(conversation_history) if conversation_history else 0
        is_first_round = (round_num == 0)
        
        # Step 2: 准备caption内容
        all_captions = [video.get("caption", "") for video in top_k_videos]
        if target_caption:
            all_captions.append(target_caption)
        
        # 合并所有caption用于分析
        combined_captions = "\n".join([f"Caption {i}: {caption}" for i, caption in enumerate(all_captions, 1)])
        
        # Step 3: 历史对话拼接
        history_text = ""
        if conversation_history:
            for i, exchange in enumerate(conversation_history):
                history_text += f"Q{i+1}: {exchange['question']}\nA{i+1}: {exchange['answer']}\n"
        
        # Step 4: 构建prompt
                prompt = f"""
        You are generating a question about a video based on its caption.

                    Original query: "{query}"
        
        Available captions:
        {combined_captions}
        
        Previous questions and answers:
        {history_text}
        
        Your task is to generate ONE simple question that:
        1. Can be directly answered using information from one or more of these captions
        2. Uses only words and concepts that appear in these captions
        3. Is simple and focuses on one specific detail
        4. Avoids asking about details not mentioned in any caption
        
        Examples of good questions (only if mentioned in captions):
        - "Is the person sitting or standing?"
        - "What is the person wearing?"
        - "Is the person indoors or outdoors?"
        - "Is the person alone or with others?"
        
        IMPORTANT: Return ONLY the question text itself. Do not add any formatting like asterisks, quotes, or prefixes like "Question:". Do not include explanations of your reasoning.
        """
        
        # Step 5: 调用LLM生成问题
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages = [{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,
                max_tokens=50,
            )
            # 使用通用清理函数处理结果
            return self._clean_generated_text(response.choices[0].message.content)
        except Exception as e:
            print(f"生成问题时发生错误: {e}")
            return "这个视频中有什么内容?"

    def generate_answer(self, query, question, video_description):
        prompt = f"""
        You are answering a question about a video based on its caption.
        
        Video caption: "{video_description}"
        Question: "{question}"
        
        Answer guidelines:
        1. Your goal is to provide a helpful answer based on the caption
        2. Try to find information in the caption that can answer the question
        3. Be lenient - if the caption has related information that could help, use it
        4. Only say "Cannot determine" if there is absolutely nothing relevant in the caption
        5. Keep your answer brief but informative
        6. It's okay to make reasonable inferences from the caption
        
        IMPORTANT: Return ONLY the answer text itself. Do not add any formatting like asterisks, quotes, or prefixes like "Answer:". Do not include explanations of your reasoning.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages = [{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,
                max_tokens=50,
            )
            # 使用通用清理函数处理结果
            return self._clean_generated_text(response.choices[0].message.content)
        except Exception as e:
            print(f"生成回答时发生错误: {e}")
            return "不确定"
    
    def generate_summary(self, query, conversation_history, top_k_captions, target_caption=None):
        # 创建描述所有视频的文本
        video_descriptions = ""
        for i, caption in enumerate(top_k_captions):
            video_descriptions += f"视频 {i+1}: {caption}\n"
        
        # 创建对话历史的文本
        history_text = ""
        for i, exchange in enumerate(conversation_history):
            history_text += f"问题 {i+1}: {exchange['question']}\n"
            history_text += f"回答 {i+1}: {exchange['answer']}\n"
        
        prompt = f"""
            You are a video retrieval assistant helping a user find a specific video.

            Original query: {query}

            Conversation history:
            {history_text}

            Current search results (ranked by relevance):
            {video_descriptions}

            Please generate a concise summary for the next round, including:
            - What the user is originally looking for
            - All known visual details about the target video so far (e.g., people, actions, objects, expressions, settings)
            - What key information is still missing and should be asked next

            Keep the summary clear and focused, no more than 5 sentences.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                temperature=0.3,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成总结时发生错误: {e}")
            return "无法生成总结。请重试或直接提问。"
    
    def update_query(self, current_query, question, answer, current_rank=None, best_rank=None):
        prompt = f"""
        You are updating a search query for a video retrieval system.

        Current query: "{current_query}"
        Question: "{question}"
        Answer: "{answer}"

        Guidelines:
        1. Keep the important parts of the original query
        2. Add new information from the answer if it's helpful
        3. Remove any parts contradicted by the answer
        4. If the answer was "Cannot determine" or similar, keep the query mostly the same
        5. Make the query read naturally as a description
        6. Be concise but descriptive
        
        IMPORTANT: Return ONLY the updated query text itself. Do not add any formatting like asterisks, quotes, or phrases like "Updated Query:". Do not include explanations of changes.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.3,
                max_tokens=50,
            )
            # 使用通用清理函数处理结果
            return self._clean_generated_text(response.choices[0].message.content)
        except Exception as e:
            print(f"更新查询时发生错误: {e}")
            return current_query
    
    def run_interactive_retrieval(self, query, video_captions, target_video_id=None, max_turns=MAX_ROUNDS, worker_id=None):
        # 创建前缀函数
        def prefix_print(msg):
            if worker_id is not None:
                print(f"[进程 {worker_id}] {msg}")
            else:
                print(msg)
        
        # 初始化状态变量
        best_rank = float('inf')
        best_turn = -1  # 记录最佳排名出现的轮次
        best_query = ""  # 记录最佳排名对应的查询
        best_top_videos = []  # 记录最佳排名时的top视频
        conversation_history = []
        expanded_query = query
        
        # 打印初始查询
        prefix_print(f"目标视频: {target_video_id}")
        prefix_print(f"初始查询: {expanded_query}")
        
        # 初始搜索
        query_embedding = self.encode_text([expanded_query])
        top_k_videos = self.retrieve_videos(expanded_query, TOP_K, query_embedding)
        
        # 如果有目标视频，计算初始排名 - 搜索全部视频以获取准确排名
        current_rank = None
        if target_video_id is not None:
            current_rank = self.find_target_rank(target_video_id, query_embedding)
            if current_rank is not None:
                prefix_print(f"初始排名: {current_rank}")
                best_rank = current_rank
            else:
                prefix_print(f"目标视频未在检索结果中找到")
                best_rank = -1  # 使用-1表示未找到，而不是inf
            best_query = expanded_query
            best_top_videos = top_k_videos.copy()
        
        search_results = {
            'initial_query': query,
            'turns': [],
            'final_query': expanded_query,
            'initial_rank': current_rank,
            'best_rank': best_rank,
            'final_rank': current_rank,
            'best_turn': -1,  # -1表示初始轮
            'target_video': target_video_id
        }
        
        # 记录每轮的top_k视频列表
        all_turns_top_videos = []
        all_turns_top_videos.append(top_k_videos)
        
        # 如果初始搜索已经达到成功阈值，可以提前结束
        if current_rank is not None and current_rank > 0 and current_rank <= TOP_K:
            prefix_print(f"初始搜索已达到成功阈值，排名为: {current_rank}")
            search_results['best_turn'] = -1
            return search_results, all_turns_top_videos
        
        # 交互循环
        for turn in range(max_turns):
            # 获取目标视频caption用于问题生成
            target_caption = None
            if target_video_id is not None:
                if isinstance(video_captions, dict):
                    # 直接查找
                    if target_video_id in video_captions:
                        target_caption = video_captions[target_video_id]
                        if turn == 0:  # 只在第一轮打印
                            prefix_print(f"\n目标视频Caption: {target_caption}")
                    # 检查字典中是否有带扩展名的版本
                    else:
                        for vid_id, caption in video_captions.items():
                            # 去除扩展名比较
                            if '.' in vid_id and os.path.splitext(vid_id)[0] == target_video_id:
                                target_caption = caption
                                if turn == 0:  # 只在第一轮打印
                                    prefix_print(f"\n目标视频Caption: {target_caption}")
                                break
                elif isinstance(video_captions, list):
                    # 列表处理
                    for item in video_captions:
                        if isinstance(item, tuple) and len(item) == 2:
                            vid_id, caption = item
                            # 去除扩展名比较
                            vid_base = os.path.splitext(vid_id)[0] if '.' in vid_id else vid_id
                            if vid_base == target_video_id:
                                target_caption = caption
                                break
                
            # 生成问题 - 使用全部对话历史，并传入目标caption
            question = self.generate_question(expanded_query, top_k_videos, conversation_history, target_caption=target_caption)
            prefix_print(f"\n问题 {turn+1}: {question}")
            
            # 如果有目标视频，使用模拟用户回答
            if target_video_id is not None:
                if target_caption:
                    answer = self.generate_answer(query, question, target_caption)
                    prefix_print(f"回答 {turn+1}: {answer}")
                    
                    # 简短提示，不要太多调试信息
                    if "cannot determine" in answer.lower():
                        prefix_print("* 提示: 尝试其他问题")
                else:
                    prefix_print(f"警告: 无法找到目标视频 {target_video_id} 的描述")
                    answer = "不确定"
                    prefix_print(f"回答 {turn+1}: {answer}")
            else:
                # 否则，让用户回答
                prefix_print(f"\n问题: {question}")
                answer = input("请输入回答: ")
                prefix_print(f"回答 {turn+1}: {answer}")
            
            # 记录对话
            conversation_history.append({
                'question': question,
                'answer': answer
            })
            
            # 更新查询
            expanded_query = self.update_query(
                expanded_query, question, answer, 
                current_rank=current_rank, best_rank=best_rank
            )
            prefix_print(f"更新查询 {turn+1}: {expanded_query}")
            
            # 使用更新后的查询重新检索
            query_embedding = self.encode_text([expanded_query])
            top_k_videos = self.retrieve_videos(expanded_query, TOP_K, query_embedding)
            
            # 保存本轮的top_k视频
            all_turns_top_videos.append(top_k_videos)
            
            # 如果有目标视频，更新排名
            if target_video_id is not None:
                current_rank = self.find_target_rank(target_video_id, query_embedding)
                if current_rank is not None:
                    rank_info = f"排名 {turn+1}: {current_rank}"
                    if current_rank <= TOP_K:
                        rank_info += " (in top-k)"
                    # 只有当best_rank为-1（未找到）或当前排名更好时才更新
                    if best_rank == -1 or current_rank < best_rank:
                        best_rank = current_rank
                        best_turn = turn
                        best_query = expanded_query
                        best_top_videos = top_k_videos.copy()
                    prefix_print(rank_info)
                else:
                    prefix_print(f"排名 {turn+1}: 目标视频未在检索结果中找到")
                    # 如果best_rank还是-1，保持-1；否则保持当前最佳排名
                    if best_rank == -1:
                        current_rank = -1
            
            # 记录本轮结果
            turn_result = {
                'turn': turn + 1,
                'question': question,
                'answer': answer,
                'query': expanded_query,
                'rank': current_rank,
            }
            search_results['turns'].append(turn_result)
            
            # 如果达到了成功阈值，提前结束
            if current_rank is not None and current_rank > 0 and current_rank <= TOP_K:
                prefix_print(f"达到成功阈值，终止交互，排名为: {current_rank}")
                break
        
        # 更新最终结果 - 使用最佳排名的信息
        search_results['final_query'] = best_query if best_turn >= 0 else expanded_query
        search_results['final_rank'] = current_rank
        search_results['best_rank'] = best_rank
        search_results['best_turn'] = best_turn
        
        # 打印最终结果总结
        if best_turn >= 0:
            prefix_print(f"\n最佳排名: {best_rank} (轮次 {best_turn+1})")
        else:
            prefix_print(f"\n最佳排名: {best_rank} (初始查询)")
        
        return search_results, all_turns_top_videos

def process_single_item(args):
    """处理单个样本的函数"""
    idx, video_name, eng_caption, captions_file, output_dir, results_df, video_path, faiss_index_path, id2video_path = args
    
    # 设置环境变量，分配GPU - 修改GPU分配方式
    if torch.cuda.is_available():
        gpu_id = idx % torch.cuda.device_count()
        # 使用torch直接设置设备而不是通过环境变量
        torch.cuda.set_device(gpu_id)
        print(f"[进程 {idx}] 使用GPU {gpu_id}")
    
    try:
        print(f"[进程 {idx}] 初始化处理样本 {idx}: {video_name}")
        start_time = time.time()
        
        # 初始化系统（使用预先构建的FAISS索引）
        system_start_time = time.time()
        system = InteractiveVideoRetrievalSystem(captions_file, faiss_index_path, id2video_path)
        all_captions = dict(zip(system.caption_df['video_name'], system.caption_df['model_caption']))
        system_init_time = time.time() - system_start_time
        print(f"[进程 {idx}] 系统初始化耗时: {system_init_time:.2f}秒")
        
        # 获取目标视频的model_caption
        model_caption = all_captions.get(video_name, "")
        
        # 运行交互式检索，传入worker_id以便在输出中添加前缀
        print(f"[进程 {idx}] 开始交互式检索: {video_name}")
        search_results, all_turns_top_videos = system.run_interactive_retrieval(
            query=eng_caption,
            video_captions=all_captions,
            target_video_id=video_name,
            max_turns=MAX_ROUNDS,
            worker_id=idx  # 传递idx作为worker_id用于输出标识
        )
        
        # 收集基本结果
        result = {
            'video_name': video_name,
            'video_path': video_path,
            'model_caption': model_caption,
            'initial_query': eng_caption,
            'initial_rank': search_results['initial_rank'],
            'final_rank': search_results['final_rank'],
            'best_rank': search_results['best_rank'],
            'best_turn': search_results['best_turn']+1 if search_results['best_turn'] >= 0 else 0,
            'turns': len(search_results['turns']),
        }
        
        # 添加初始查询的top10视频
        if all_turns_top_videos:
            videos = []
            scores = []
            for v in all_turns_top_videos[0][:10]:
                if isinstance(v, dict):
                    videos.append(v.get('video_id', ''))
                    scores.append(v.get('score', 0.0))
            result['top10_videos_0'] = str(videos)
            result['top10_scores_0'] = str(scores)
        
        # 添加每轮交互信息
        for turn_idx, turn_data in enumerate(search_results['turns']):
            turn_num = turn_idx + 1
            if turn_num <= MAX_ROUNDS:
                result[f'question_{turn_num}'] = turn_data.get('question', '')
                result[f'answer_{turn_num}'] = turn_data.get('answer', '')
                result[f'query_{turn_num}'] = turn_data.get('query', '')
                result[f'rank_{turn_num}'] = turn_data.get('rank', '')
                
                # 保存该轮的top10视频
                if turn_num < len(all_turns_top_videos):
                    videos = []
                    scores = []
                    for v in all_turns_top_videos[turn_num][:10]:
                        if isinstance(v, dict):
                            videos.append(v.get('video_id', ''))
                            scores.append(v.get('score', 0.0))
                    result[f'top10_videos_{turn_num}'] = str(videos)
                    result[f'top10_scores_{turn_num}'] = str(scores)
        
        processing_time = time.time() - start_time
        print(f"[进程 {idx}] 完成样本 {idx}: {video_name} - 初始排名: {result['initial_rank']} -> 最佳排名: {result['best_rank']} (耗时: {processing_time:.2f}秒)")
        
        return idx, result, processing_time
    
    except Exception as e:
        print(f"[进程 {idx}] 处理样本 {idx} 出错: {str(e)[:100]}...")
        traceback.print_exc()
        return idx, None, 0

def main():
    # ==================== 配置区域 ====================
    # 修改这些路径来适配你的新数据
    
    # 1. 输入文件路径 - 修改为你的新 xlsx 文件路径
    captions_file = "/home/peterchen/M2/MER2024/llava_next_video_caption.xlsx"  # 修改这里
    
    # 2. 处理范围 - 根据需要修改
    start_idx = 0  # 起始索引
    end_idx = 100  # 结束索引（包含）
    
    # 3. FAISS 索引路径 - 如果使用相同的数据集，可以复用现有索引
    # 如果使用新的数据集，系统会自动创建新的索引
    faiss_index_path = "/home/peterchen/M2/本番/llava_next_video_deepseek/mer2024/faiss_index.index"
    id2video_path = "/home/peterchen/M2/本番/llava_next_video_deepseek/mer2024/id2video_mapping.pkl"
    
    # 4. 输出目录 - 会自动创建带时间戳的目录
    output_dir = f"/home/peterchen/M2/本番/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 5. 其他参数
    save_interval = 5  # 每处理5个样本保存一次结果
    
    # ==================== 配置结束 ====================
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # FAISS索引文件路径
    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    
    # 检查GPU可用性
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count > 0:
        print(f"检测到 {gpu_count} 个GPU:")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("没有检测到GPU，将使用CPU运行")
    
    # 创建日志文件
    stats_file = os.path.join(output_dir, "stats.txt")
    with open(stats_file, "w") as f:
        f.write(f"处理开始时间: {datetime.now()}\n")
        f.write(f"处理范围: 从索引 {start_idx} 到 {end_idx}\n")
        if gpu_count > 0:
            f.write(f"检测到 {gpu_count} 个GPU:\n")
            for i in range(gpu_count):
                f.write(f"  GPU {i}: {torch.cuda.get_device_name(i)}\n")
    
    # 读取数据
    print(f"从文件加载数据: {captions_file}")
    df = pd.read_excel(captions_file)
    subset_df = df.iloc[start_idx:end_idx+1]
    print(f"需要处理 {len(subset_df)} 条记录，从索引 {start_idx} 到 {end_idx}")
    
    # 预先构建FAISS索引（如果不存在）
    # 这样所有进程可以直接加载同一个索引，而不是每个进程都重新构建
    if not (os.path.exists(faiss_index_path) and os.path.exists(id2video_path)):
        print("预先构建FAISS索引，这样所有进程可以共享使用...")
        system = InteractiveVideoRetrievalSystem(captions_file, faiss_index_path, id2video_path)
        print(f"FAISS索引已保存到: {faiss_index_path}")
    else:
        print(f"将使用现有FAISS索引: {faiss_index_path}")
    
    # 创建空的结果DataFrame
    results_df = pd.DataFrame()
    
    # 创建参数列表
    args_list = [
        (i, row['video_name'], row['eng_caption'], captions_file, output_dir, results_df, row.get('video_path', ''), faiss_index_path, id2video_path)
        for i, (_, row) in enumerate(subset_df.iterrows(), start=start_idx)
    ]
    
    # 确定进程数 - 使用GPU数量但最大不超过样本数和并调整到最优计算效率
    num_processes = min(max(gpu_count, 1), len(args_list))  # 至少一个进程，最多不超过样本数
    print(f"将使用 {num_processes} 个并行进程")
    
    # 更新日志
    with open(stats_file, "a") as f:
        f.write(f"样本数量: {len(subset_df)}\n")
        f.write(f"使用进程数: {num_processes}\n\n")
        f.write("=" * 50 + "\n\n")
    
    # 使用进程池并行处理
    start_time = time.time()
    processed_count = 0
    all_results = {}
    processing_times = []
    
    print(f"开始并行处理...\n")
    
    with Pool(processes=num_processes) as pool:
        for idx, result, proc_time in pool.imap_unordered(process_single_item, args_list):
            if result is not None:
                processed_count += 1
                all_results[idx] = result
                processing_times.append(proc_time)
                
                # 每处理save_interval个样本，或者处理完所有样本，保存一次结果
                if processed_count % save_interval == 0 or processed_count == len(subset_df):
                    # 保存当前所有结果
                    result_df = pd.DataFrame.from_dict(all_results, orient='index')
                    result_file = os.path.join(output_dir, "results.xlsx")
                    result_df.to_excel(result_file)
                    print(f"已处理 {processed_count}/{len(subset_df)} 个样本，保存结果到: {result_file}")
                    
                    # 更新统计信息
                    with open(stats_file, "a") as f:
                        f.write(f"[{datetime.now()}] 已处理 {processed_count}/{len(subset_df)} 个样本\n")
                        if processing_times:
                            avg_time = sum(processing_times) / len(processing_times)
                            f.write(f"平均处理时间: {avg_time:.2f}秒/样本\n")
                            # 计算预计剩余时间
                            remaining = len(subset_df) - processed_count
                            if remaining > 0:
                                est_time = remaining * avg_time / num_processes
                                hours = int(est_time // 3600)
                                minutes = int((est_time % 3600) // 60)
                                seconds = int(est_time % 60)
                                f.write(f"预计剩余时间: {hours}小时 {minutes}分钟 {seconds}秒\n")
                        f.write("-" * 30 + "\n")
    
    # 程序结束时无需再次保存，因为最后一次迭代已经保存过了
    print(f"已完成所有处理，结果保存在: {os.path.join(output_dir, 'results.xlsx')}")
    
    # 计算总耗时
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    print(f"总处理时间: {hours}小时 {minutes}分钟 {seconds}秒")
    
    # 计算每样本耗时
    avg_time = total_time / len(all_results) if all_results else 0
    print(f"平均每样本耗时: {avg_time:.2f}秒")
    
    # 统计交互轮次
    if all_results and 'turns' in all_results[next(iter(all_results))].keys():
        avg_turns = sum(r['turns'] for r in all_results.values()) / len(all_results)
        print(f"平均交互轮次: {avg_turns:.2f}")
    
    # 简单统计
    if all_results and 'initial_rank' in all_results[next(iter(all_results))].keys():
        # 过滤掉-1值（未找到的情况）
        valid_initial_ranks = [r['initial_rank'] for r in all_results.values() if r['initial_rank'] is not None and r['initial_rank'] > 0]
        if valid_initial_ranks:
            avg_initial = sum(valid_initial_ranks) / len(valid_initial_ranks)
            print(f"平均初始排名: {avg_initial:.2f}")
        else:
            print("没有有效的初始排名数据")
    
    if all_results and 'best_rank' in all_results[next(iter(all_results))].keys():
        # 过滤掉-1值（未找到的情况）
        valid_best_ranks = [r['best_rank'] for r in all_results.values() if r['best_rank'] is not None and r['best_rank'] > 0]
        if valid_best_ranks:
            avg_best = sum(valid_best_ranks) / len(valid_best_ranks)
            print(f"平均最佳排名: {avg_best:.2f}")
        else:
            print("没有有效的最佳排名数据")
        
        # 计算召回率 - 只考虑找到的视频（排名>0）
        ranks = [10, 50, 100, 200]
        print("\n召回率统计:")
        for k in ranks:
            r_at_k = sum(r['best_rank'] > 0 and r['best_rank'] <= k for r in all_results.values()) / len(all_results) * 100
            print(f"最佳 R@{k}: {r_at_k:.2f}%")
        
        # 统计未找到的视频数量
        not_found_count = sum(r['best_rank'] == -1 for r in all_results.values())
        not_found_rate = not_found_count / len(all_results) * 100
        print(f"未找到的视频: {not_found_count}/{len(all_results)} ({not_found_rate:.2f}%)")
    
    # 更新日志
    with open(stats_file, "a") as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"处理完成时间: {datetime.now()}\n")
        f.write(f"总处理时间: {hours}小时 {minutes}分钟 {seconds}秒\n")
        f.write(f"平均每样本耗时: {avg_time:.2f}秒\n")
        
        if all_results and 'turns' in all_results[next(iter(all_results))].keys():
            f.write(f"平均交互轮次: {avg_turns:.2f}\n")
            
        if all_results and 'best_rank' in all_results[next(iter(all_results))].keys():
            f.write(f"平均最佳排名: {avg_best:.2f}\n")
            
            for k in ranks:
                r_at_k = sum(r['best_rank'] > 0 and r['best_rank'] <= k for r in all_results.values()) / len(all_results) * 100
                f.write(f"最佳 R@{k}: {r_at_k:.2f}%\n")
            
            # 记录未找到的视频统计
            not_found_count = sum(r['best_rank'] == -1 for r in all_results.values())
            not_found_rate = not_found_count / len(all_results) * 100
            f.write(f"未找到的视频: {not_found_count}/{len(all_results)} ({not_found_rate:.2f}%)\n")
    
    return 0
    
if __name__ == "__main__":
    main()
                    
                                    
                                 
        
        

        
