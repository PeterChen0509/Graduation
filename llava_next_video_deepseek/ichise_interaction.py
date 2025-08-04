import os
import torch
import numpy as np
import pandas as pd
import faiss
import pickle
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
import sys
sys.path.append("/home/peterchen/M2/contriever")
from src.contriever import Contriever

# 配置
MAX_ROUNDS = 5  # 最大交互轮次
TOP_K = 10      # 检索结果数量/成功阈值（当排名<=此值时提前结束）
OPENAI_BASE_URL = "https://api.deepseek.com"
OPENAI_API_KEY = "sk-3e3bc603ac9647af91919f2cd57efbf5"
LLM_MODEL = "deepseek-chat"
MAX_SEARCH_LIMIT = None  # 大型视频库搜索上限，设为None则搜索全部

class InteractiveVideoRetrievalSystem:
    def __init__(self, caption_file, faiss_index_path=None, id2video_path=None):
        # 初始化交互式视频检索系统
        print(f"正在读取 Excel 文件: {caption_file}")
        self.caption_df = pd.read_excel(caption_file)
        
        # 检查是否存在必要的列
        if "name" not in self.caption_df.columns or "model_caption" not in self.caption_df.columns:
            raise ValueError("Excel 文件中必须包含 'name' 和 'model_caption' 列")
        
        #  过滤掉空的 caption
        self.caption_df = self.caption_df.dropna(subset=['model_caption'])
        
        # 创建 video_name 到 caption 的映射
        self.video_to_caption = dict(zip(self.caption_df['name'], self.caption_df['model_caption']))
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
            print(f"成功加载索引，包含{self.faiss_index.ntotal}个视频")
            
            # 添加调试信息
            print("\n=== 调试: FAISS索引加载 ===")
            print(f"索引类型: {type(self.faiss_index)}")
            print(f"向量维度: {self.faiss_index.d}")
            print(f"id2video类型: {type(self.id2video)}")
            print(f"id2video大小: {len(self.id2video)}")
            if len(self.id2video) > 0:
                id_samples = list(self.id2video.items())[:5]
                print(f"id2video样例(前5项): {id_samples}")
                # 检查ID格式
                has_extension = ['.mp4' in str(v) or '.avi' in str(v) or '.mov' in str(v) for k,v in id_samples]
                print(f"示例ID是否包含扩展名: {has_extension}")
            
            # 反向创建 video 到 id 的映射
            self.video2id = {v:k for k, v in self.id2video.items()}
            print(f"video2id大小: {len(self.video2id)}")
            # 检查扩展名与映射的关系
            key_with_ext = [k for k in list(self.video2id.keys())[:20] if '.' in str(k)]
            print(f"前20个key中带扩展名的数量: {len(key_with_ext)}")
            if len(key_with_ext) > 0:
                print(f"带扩展名的key样例: {key_with_ext[:3]}")
        else:
            print("创建新的 FAISS 索引")
            self.id2video, self.video2id, self.faiss_index = self.build_faiss_index()
            
            # 保存索引和映射
            if faiss_index_path and id2video_path:
                os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
                os.makedirs(os.path.dirname(id2video_path), exist_ok=True)
                
                faiss.write_index(self.faiss_index, faiss_index_path)
                with open(id2video_path, "wb") as f:
                    pickle.dump(self.id2video, f)
                print(f"FAISS 索引已保存至: {faiss_index_path}")
                print(f"ID 映射已保存至：{id2video_path}")
        
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
        # 转移到 GPU
        if torch.cuda.is_available():
            model = model.cuda()
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
        
        # 把输入转移到 GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # 使用模型生成嵌入
        with torch.no_grad():
            embeddings = self.contriever(**inputs)
            
            # 将张量移到CPU并转换为NumPy数组
            if torch.cuda.is_available():
                embeddings = embeddings.cpu()
            embeddings = embeddings.numpy()
            
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
        
        print(f"FAISS 索引已创建，包含 {index.ntotal} 个向量")
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
        print(f"FAISS检索到 {len(indices[0])} 个结果")
        
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
        
        print(f"成功转换 {len(results)} 个结果到字典")
        if len(results) > 0:
            print(f"前5个结果:")
            for i, res in enumerate(results[:5]):
                print(f"  {i+1}. ID: '{res['video_id']}' (得分: {res['score']:.4f})")
                print(f"     描述: {res['caption'][:100]}...")
        
        return results
    
    def find_target_rank(self, target_video_id, query_embedding, top_captions=None, top_k=TOP_K):
        """找出目标视频在检索结果中的排名"""
        if target_video_id is None:
            return None
        
        # 简单打印调试信息
        print(f"查找目标ID: '{target_video_id}'")
        
        try:
            # 获取索引中的总视频数量
            total_videos = self.faiss_index.ntotal
            
            # 搜索整个索引以获取完整排名
            distances, indices = self.faiss_index.search(query_embedding, total_videos)
            all_video_ids = [self.id2video[idx] for idx in indices[0] if idx >= 0 and idx < len(self.id2video)]
            
            print(f"搜索了整个索引，共{len(all_video_ids)}个视频")
            print(f"检索结果ID示例: {all_video_ids[:3]}")
        
        # 直接匹配：检查目标ID是否在检索结果中
            if target_video_id in all_video_ids:
                global_rank = all_video_ids.index(target_video_id) + 1
                print(f"目标视频精确匹配成功! 排名: {global_rank}/{len(all_video_ids)}")
                return global_rank
            
        # 如果没有直接匹配，尝试不区分扩展名的匹配
        # 将目标ID和检索结果ID都处理为无扩展名形式
        target_base_id = os.path.splitext(target_video_id)[0] if '.' in target_video_id else target_video_id
        
            for i, vid in enumerate(all_video_ids):
            vid_base = os.path.splitext(vid)[0] if '.' in vid else vid
            if vid_base == target_base_id:
                    global_rank = i + 1
                    print(f"目标视频基础名匹配成功! 排名: {global_rank}/{len(all_video_ids)}")
                    return global_rank
        
        print(f"目标视频不在检索结果中")
        return None
            
        except Exception as e:
            print(f"搜索全部索引时出错: {e}")
            return None
    
    def generate_question(self, query, top_k_videos, conversation_history=None):
        # 生成关于视频的问题
        similarity_scores = [video["score"] for video in top_k_videos if "score" in video]
        if not similarity_scores:
            similarity_scores = [0.5]*len(top_k_videos)
        
        # Step 1: 判断召回结果的质量
        avg_sim = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0    
        low_quality = avg_sim < 0.15
        
        # Step 2: 判断是否是第一轮交互
        round_num = len(conversation_history) if conversation_history else 0
        is_first_round = (round_num == 0)
        
        # Step 3: 视频描述拼接
        video_descriptions = []
        for i, video in enumerate(top_k_videos):
            caption = video.get("caption", "")
            video_descriptions.append(f"视频 {i+1}: {caption}")
        
        videos_text = "\n".join(video_descriptions)
        
        # Step 4: 历史对话拼接
        history_text = ""
        if conversation_history:
            for i, exchange in enumerate(conversation_history):
                history_text += f"问题 {i+1}: {exchange['question']}\n"
                history_text += f"回答 {i+1}: {exchange['answer']}\n"
        
        history_display = ""
        if history_text:
            history_display = " 之前的对话历史:\n" + history_text
        
        # Step 5: 根据判断构建 Prompt
        if low_quality:
            if is_first_round:
                # 第一轮 + 候选质量低：意图澄清模式
                prompt = f"""
                    Your task is to improve video search by generating a specific and effective question.

                    Original query: "{query}"
                    The initial search results were of low quality. Please generate a clear and focused question that can guide the search better. If your first attempt is not effective, generate a different question to try again.

                    Your question should meet the following requirements:
                    1. Ask about specific visual elements, such as objects, people, actions, facial expressions, or scenes.
                    2. Help clarify what makes the target video distinct or unique compared to others.
                    3. Ask for observable, factual details—something that can be seen or confirmed in the video (e.g., "Is the person smiling?" or "What color is the car?"), not personal opinions or abstract descriptions.

                    Return only the question itself. Do not include explanations or additional comments.
                """
            else:
                # 后续轮次 + 候选依然低质量 → fallback 提问（继续澄清意图）
                prompt = f"""
                    You are helping to improve video search. After {round_num} rounds, the correct video still hasn't been found and current results are low quality.

                    Original query: "{query}"
                    Previous conversation:{history_text}

                    Ask a new question focusing on visual aspects not yet mentioned.
                    Prioritize specific details such as:
                    - Number of people
                    - Specific objects, animals, or text
                    - Colors of key items
                    - Type of location or setting
                    - Facial expressions or actions

                    Return only the question itself. No explanation.
                """
        else:
            # 候选质量 OK：生成区分性问题（原有逻辑优化版）
            prompt = f"""
                You are analyzing {len(top_k_videos)} video descriptions to help find a specific video.

                Original query: "{query}"

                Video descriptions:
                {videos_text}
                {history_display}

                We are in a later round of search, and these videos still look similar but have subtle differences. Please ask a new question to help narrow down the correct video.
                
                Rules:
                - Focus on visual differences between videos (objects, people, counts, colors, actions, facial expressions)
                - Ask about elements that vary across multiple videos
                - Question must be specific and fact-based (not subjective or vague)
                - Avoid yes/no questions or repeating earlier ones
                - Keep it simple and direct

                Return only the question itself. No explanation.
            """
        # Step 6: 调用 LLM 生成问题
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages = [{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.4,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"生成问题时发生错误: {e}")
            return "这个视频里有什么内容?"

    def generate_answer(self, query, question, video_description):
        prompt = f"""
        You are playing the role of a video viewer. Answer the question based on the given video description.
        
        Video description: {video_description}
        Question: {question}
        
        Rules:
        1. Your answer must be completely based on facts from the video description
        2. If not mentioned in the video description, answer "Not sure"
        3. Keep your answer concise, no more than 20 characters
        4. Do not explain your answer
        
        Only return your short answer.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages = [{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.1,
                max_tokens=50,
            )
            return response.choices[0].message.content.strip()
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
        stats_info = ""
        if current_rank and best_rank:
            stats_info = f"当前目标视频排名: {current_rank}, 最佳排名: {best_rank}。"
        
        prompt = f"""
            You are part of a video retrieval system. Update the search query to better match the video the user is looking for.

            Current query: {current_query}

            Latest interaction:
            Question: {question}
            Answer: {answer}

            {stats_info}

            Task:
            Update the query using key visual details from the user's answer (e.g., objects, people, actions, facial expressions, settings).

            Rules:
            - Keep core ideas from the original query
            - Add confirmed new details (e.g., appearance, expressions, gestures)
            - Remove elements the user said are not present
            - Focus on concrete visual content, not abstract ideas
            - Keep it concise — fewer than 20 words

            Return only the updated query text. No explanation.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.2,
                max_tokens=100,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"更新查询时发生错误: {e}")
            return current_query
    
    def run_interactive_retrieval(self, query, video_captions, target_video_id=None, max_turns=MAX_ROUNDS):
        # 初始化状态变量
        best_rank = float('inf')
        best_turn = -1  # 记录最佳排名出现的轮次
        best_query = ""  # 记录最佳排名对应的查询
        best_top_videos = []  # 记录最佳排名时的top视频
        conversation_history = []
        expanded_query = query
        
        # 打印基本调试信息
        print(f"查询: '{query}'")
        print(f"目标视频ID: '{target_video_id}'")
        
        # 初始搜索
        query_embedding = self.encode_text([expanded_query])
        top_k_videos = self.retrieve_videos(expanded_query, TOP_K, query_embedding)
        
        # 如果有目标视频，计算初始排名 - 搜索全部视频以获取准确排名
        current_rank = None
        if target_video_id is not None:
            current_rank = self.find_target_rank(target_video_id, query_embedding)
            if current_rank is not None:
                print(f"初始排名: {current_rank}")
                best_rank = current_rank
            else:
                print(f"错误：目标视频未在检索结果中找到，这不应该发生")
                # 由于数据集是一一对应的，这种情况不应该发生
                # 如果发生了，可能是ID匹配问题，我们设置为一个很大的数字
                best_rank = 1000
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
        }
        
        # 记录每轮的top_k视频列表
        all_turns_top_videos = []
        all_turns_top_videos.append(top_k_videos)
        
        # 如果初始搜索已经达到成功阈值，可以提前结束
        if current_rank is not None and current_rank <= TOP_K:
            print(f"初始搜索已达到成功阈值，排名为: {current_rank}")
            search_results['best_turn'] = -1
            return search_results, all_turns_top_videos
        
        # 交互循环
        for turn in range(max_turns):
            # 生成问题 - 使用全部对话历史
            question = self.generate_question(expanded_query, top_k_videos, conversation_history)
            
            # 如果有目标视频，使用模拟用户回答
            if target_video_id is not None:
                target_caption = ""
                # 获取目标视频caption (先移除所有扩展名)
                if isinstance(video_captions, dict):
                    # 直接查找
                    if target_video_id in video_captions:
                        target_caption = video_captions[target_video_id]
                    # 检查字典中是否有带扩展名的版本
                    else:
                        for vid_id, caption in video_captions.items():
                            # 去除扩展名比较
                            if '.' in vid_id and os.path.splitext(vid_id)[0] == target_video_id:
                                target_caption = caption
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
                
                if target_caption:
                    answer = self.generate_answer(query, question, target_caption)
                else:
                    print(f"警告: 无法找到目标视频 {target_video_id} 的描述")
                    answer = "I don't know."
            else:
                # 否则，让用户回答
                print(f"\nQuestion: {question}")
                answer = input("Your answer: ")
            
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
            
            # 使用更新后的查询重新检索
            query_embedding = self.encode_text([expanded_query])
            top_k_videos = self.retrieve_videos(expanded_query, TOP_K, query_embedding)
            
            # 保存本轮的top_k视频
            all_turns_top_videos.append(top_k_videos)
            
            # 如果有目标视频，更新排名
            if target_video_id is not None:
                current_rank = self.find_target_rank(target_video_id, query_embedding)
                
                # 检查是否是最佳排名
                if current_rank is not None and current_rank < best_rank:
                    best_rank = current_rank
                    best_turn = turn
                    best_query = expanded_query
                    best_top_videos = top_k_videos.copy()
            
            # 记录本轮结果
            turn_result = {
                'turn': turn + 1,
                'question': question,
                'answer': answer,
                'query': expanded_query,
                'rank': current_rank,
            }
            search_results['turns'].append(turn_result)
            
            # 日志输出
            status = f"Turn {turn+1}: "
            if current_rank is not None:
                status += f"Rank {current_rank}"
                if current_rank <= TOP_K:
                    status += " (in top-k)"
                if current_rank == best_rank:
                    status += " (new best)"
            status += f" | Query: {expanded_query}"
            print(status)
            
            # 如果排名为1，提前结束当前视频的交互
            if current_rank is not None and current_rank == 1:
                print(f"目标视频排名第1，提前结束当前视频的交互")
                break
        
        # 更新最终结果 - 使用最佳排名的信息
        search_results['final_query'] = best_query if best_turn >= 0 else expanded_query
        search_results['final_rank'] = current_rank
        search_results['best_rank'] = best_rank
        search_results['best_turn'] = best_turn
        
        return search_results, all_turns_top_videos

def main():
    # ==================== 配置区域 ====================
    # 修改这些路径来适配你的新数据
    
    # 1. 输入文件路径 - 修改为你的新 xlsx 文件路径
    captions_file = "/home/peterchen/M2/MER2024/llava_next_video_caption.xlsx"  # 修改这里
    
    # 2. 输出文件路径 - 结果保存位置
    output_file = "/home/peterchen/M2/本番/llava_next_video_deepseek/mer2024/search_results.xlsx"
    
    # 3. 处理范围 - 处理整个数据集
    # start_idx = 0      # 起始索引
    # end_idx = 10       # 结束索引（包含）- 建议先用小范围测试
    
    # 4. FAISS 索引路径 - 保存索引供下次使用
    faiss_index_path = "/home/peterchen/M2/本番/llava_next_video_deepseek/mer2024/faiss_index.index"
    id2video_path = "/home/peterchen/M2/本番/llava_next_video_deepseek/mer2024/id2video_mapping.pkl"
    
    # ==================== 配置结束 ====================
    
    print("=" * 60)
    print("交互式视频检索系统启动")
    print("=" * 60)
    print(f"输入文件: {captions_file}")
    print(f"输出文件: {output_file}")
    print(f"处理范围: 整个数据集")
    print(f"FAISS索引: {'自动创建' if faiss_index_path is None else faiss_index_path}")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录已创建: {output_dir}")
    
    # 初始化检索系统 - 系统会自动创建FAISS索引
    print("\n正在初始化检索系统...")
    system = InteractiveVideoRetrievalSystem(captions_file, faiss_index_path, id2video_path)
    
    # 获取DataFrame
    df = system.caption_df.copy()
    print(f"成功加载 {len(df)} 个视频的caption")
    
    # 检查必要的列是否存在
    required_columns = ['name', 'model_caption', 'eng_caption']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"错误: 缺少必要的列: {missing_columns}")
        print(f"当前文件的列: {df.columns.tolist()}")
        return
    
    print(f"数据文件包含的列: {df.columns.tolist()}")
    
    # 直接添加新列
    df['interactive_initial_rank'] = None
    df['interactive_final_rank'] = None
    df['interactive_best_rank'] = None
    df['interactive_best_turn'] = None
    df['interactive_turns'] = None
    df['interactive_query'] = None
    df['top10_videos'] = None
    df['top10_scores'] = None
    df['target_rank'] = None  # 目标视频的实际排名
    
    # 为每轮交互添加列(最多5轮)
    for i in range(1, MAX_ROUNDS + 1):
        df[f'question_{i}'] = None
        df[f'answer_{i}'] = None
        df[f'query_{i}'] = None
        df[f'rank_{i}'] = None
    
    # 统计变量
    initial_ranks = []
    final_ranks = []
    best_ranks = []
    
    # 修改为字典形式，更容易通过ID查找
    all_captions = dict(zip(df['name'].tolist(), df['model_caption'].tolist()))
    print(f"创建了包含 {len(all_captions)} 个视频的caption字典")
    
    # 主循环 - 处理整个数据集
    total_samples = len(df)
    print(f"\n开始处理整个数据集，共 {total_samples} 个样本...")
    
    for i in tqdm(range(total_samples), total=total_samples):
        try:
            row = df.iloc[i]
            print(f"\n{'='*50}")
            print(f"处理样本 {i+1}/{total_samples}: {row['name']}")
            print(f"{'='*50}")
            
            # 获取查询
            query = row['eng_caption']
            target_id = row['name']
            
            print(f"目标视频: {target_id}")
            print(f"初始查询: {query}")
            print(f"目标视频描述: {row['model_caption'][:100]}...")
            
            # 运行交互式检索
            search_results, all_turns_top_videos = system.run_interactive_retrieval(
                query=query,
                video_captions=all_captions,
                target_video_id=target_id,
                max_turns=MAX_ROUNDS
            )
            
            # 获取top10视频和分数
            final_turn_videos = []
            final_turn_scores = []
            
            # 确定要使用哪一轮的结果 - 使用最佳排名的那一轮
            best_turn_idx = search_results['best_turn']
            
            # 获取对应轮次的top视频
            if best_turn_idx >= 0 and best_turn_idx < len(all_turns_top_videos):
                # 使用最佳排名轮次的结果
                result_videos = all_turns_top_videos[best_turn_idx + 1]  # +1是因为all_turns_top_videos第一项是初始轮
            else:
                # 如果最佳是初始轮或没有更好的结果，使用最后一轮
                result_videos = all_turns_top_videos[-1]
            
            # 提取前10个视频ID和分数
            for video_info in result_videos[:10]:
                if isinstance(video_info, dict):
                    video_id = video_info.get('video_id', '')
                    score = video_info.get('score', 0.0)
                    final_turn_videos.append(video_id)
                    final_turn_scores.append(score)
            
            # 保存基本结果
            df.loc[i, 'interactive_initial_rank'] = search_results['initial_rank']
            df.loc[i, 'interactive_final_rank'] = search_results['final_rank']
            df.loc[i, 'interactive_best_rank'] = search_results['best_rank']
            df.loc[i, 'interactive_best_turn'] = search_results['best_turn']
            df.loc[i, 'interactive_turns'] = len(search_results['turns'])
            df.loc[i, 'interactive_query'] = search_results['final_query']
            df.loc[i, 'top10_videos'] = str(final_turn_videos)
            df.loc[i, 'top10_scores'] = str(final_turn_scores)
            df.loc[i, 'target_rank'] = search_results['best_rank'] # 保存目标视频的实际排名
            
            # 保存每轮交互的信息
            completed_turns = len(search_results['turns'])
            for turn_num in range(1, MAX_ROUNDS + 1):
                if turn_num <= completed_turns:
                    # 有实际交互的轮次
                    turn_data = search_results['turns'][turn_num - 1]
                    df.loc[i, f'question_{turn_num}'] = turn_data.get('question', '')
                    df.loc[i, f'answer_{turn_num}'] = turn_data.get('answer', '')
                    df.loc[i, f'query_{turn_num}'] = turn_data.get('query', '')
                    df.loc[i, f'rank_{turn_num}'] = turn_data.get('rank', '')
                else:
                    # 提前停止后的轮次，question和answer记为None，rank记为1
                    df.loc[i, f'question_{turn_num}'] = None
                    df.loc[i, f'answer_{turn_num}'] = None
                    df.loc[i, f'query_{turn_num}'] = None
                    df.loc[i, f'rank_{turn_num}'] = 1
            
            print(f"✓ 保存了 {len(search_results['turns'])} 轮交互")
            print(f"✓ 初始排名: {search_results['initial_rank']} → 最佳排名: {search_results['best_rank']}")
            
            # 收集统计信息
            if search_results['initial_rank'] is not None:
                initial_ranks.append(search_results['initial_rank'])
            if search_results['final_rank'] is not None:
                final_ranks.append(search_results['final_rank'])
            if search_results['best_rank'] is not None:
                best_ranks.append(search_results['best_rank'])
            
            # 每处理10个样本保存一次中间结果
            if (i + 1) % 10 == 0:
                df.to_excel(output_file, index=False)
                print(f"✓ 已保存中间结果到: {output_file}")
                
        except Exception as e:
            print(f"❌ 处理样本 {i} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存最终结果
    print(f"\n{'='*60}")
    print(f"保存最终结果到: {output_file}")
    df.to_excel(output_file, index=False)
    print(f"✓ 已保存所有结果")
    
    # 简化的统计信息
    print("\n" + "="*60)
    print("结果统计:")
    print("="*60)
    if initial_ranks:
        print(f"平均初始排名: {np.mean(initial_ranks):.2f}")
    if best_ranks:
        print(f"平均最佳排名: {np.mean(best_ranks):.2f}")
    if final_ranks:
        print(f"平均最终排名: {np.mean(final_ranks):.2f}")
    
    # 召回率统计
    ranks_to_evaluate = [10, 50, 100, 200]
    
    print("\n召回率统计:")
    for k in ranks_to_evaluate:
        if initial_ranks:
            initial_r_at_k = sum(r <= k for r in initial_ranks) / len(initial_ranks) * 100
            print(f"初始 R@{k}: {initial_r_at_k:.2f}%")
        
        if best_ranks:
            best_r_at_k = sum(r <= k for r in best_ranks) / len(best_ranks) * 100
            print(f"最佳 R@{k}: {best_r_at_k:.2f}%")
    
    print("\n" + "="*60)
    print("🎉 完成所有处理!")
    print("="*60)
    
if __name__ == "__main__":
    main()
                    
                                    
                                 
        
        

        
