#!/usr/bin/env python3
"""
MERLIN 参数调优系统
用于找到最优的 (m, α, β) 参数组合
"""

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
import itertools
import asyncio
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 自动设置PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logger import logger, setup_logger
from utils.env_utils import load_env_variables
from utils.data_utils import DatasetConfig, DatasetPaths, DATASET_CONFIGS
from merlin.questioner import Questioner
from merlin.reranker import Reranker
from human_agent.answerer import Answerer

class SimpleReranker:
    """简化的重排序器，用于测试"""
    
    def __init__(self, queries, video_ext):
        self.queries = queries
        self.video_ext = video_ext
        self.current_description = ""
    
    def reset_reformatter(self, initial_description: str = ""):
        self.current_description = initial_description
    
    def reformat_dialogue(self, question: str, answer: str, max_tokens: int = 500) -> str:
        # 简化的重构逻辑
        new_description = f"{self.current_description} Question: {question} Answer: {answer}"
        self.current_description = new_description
        return new_description
    
    def get_current_description(self) -> str:
        return self.current_description
    
    def init_embedding(self, target_vid):
        pass
    
    def get_image_video_text_embeddings(self, contextual_text: str = None):
        # 返回一个模拟的嵌入对象
        class MockEmbedding:
            def __init__(self, text_embedding):
                self.text_embedding = text_embedding
        
        # 创建一个简单的文本嵌入（随机向量）
        import numpy as np
        text_embedding = np.random.randn(1408)  # 1408维向量
        text_embedding = text_embedding / np.linalg.norm(text_embedding)  # 归一化
        
        return MockEmbedding(text_embedding)
    
    def rerank(self, target_vid, video_embeddings, current_query_embedding):
        # 简化的重排序逻辑
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 计算相似度
        similarities = cosine_similarity([current_query_embedding], video_embeddings)
        
        # 获取top-k
        top_k_indices = np.argsort(-similarities[0])[:10]
        top_k_ids = []
        
        for idx in top_k_indices:
            video_name = self.queries[idx]["video"].replace(self.video_ext, "")
            top_k_ids.append(video_name)
        
        # 找到目标视频的排名
        target_rank = None
        for idx, k_index in enumerate(top_k_indices):
            if self.queries[k_index]["video"].replace(self.video_ext, "") == target_vid:
                target_rank = idx + 1
                break
        
        if target_rank is None:
            target_rank = len(top_k_indices) + 1
        
        return top_k_ids, target_rank

class ParameterTuner:
    """
    参数调优系统，用于找到最优的 (m, α, β) 参数组合
    """
    
    def __init__(self, 
                 dataset: str,
                 data_path: str,
                 excel_path: str = None,
                 video_base_dir: str = None,
                 output_dir: str = "outputs",
                 env_file: str = "/home/peterchen/M2/ADEPT/.env",
                 sample_size: int = 10):
        """
        初始化参数调优器
        
        Args:
            dataset: 数据集名称
            data_path: 数据路径
            excel_path: Excel文件路径（MAFW数据集需要）
            video_base_dir: 视频基础目录（MAFW数据集需要）
            output_dir: 输出目录
            env_file: 环境变量文件路径
            sample_size: 用于参数调优的样本数量
        """
        self.dataset = dataset
        self.data_path = data_path
        self.excel_path = excel_path
        self.video_base_dir = video_base_dir
        self.output_dir = output_dir
        self.env_file = env_file
        self.sample_size = sample_size
        
        # 设置环境
        self._setup_environment()
        
        # 加载数据
        self._load_data()
        
        # 数据分区
        self._partition_data(sample_size=self.sample_size)
        
        # 初始化组件（在数据加载之后）
        self._init_components()
        
        # 参数搜索空间 - 基于熵分析结果优化
        self.param_space = {
            'm': [8, 10, 12, 15],  # K-means簇数（适合top_k=85的m值）
            'alpha': [0.0075, 0.0105, 0.0140, 0.0150],  # 簇间熵阈值（基于熵分析分布）
            'beta': [0.040, 0.047, 0.058, 0.062]   # 簇内熵阈值（基于熵分析分布）
        }
        
        # 结果存储
        self.results = []
        self.best_params = None
        self.best_score = 0.0
        
        logger.info(f"参数调优器初始化完成")
    
    def _setup_environment(self):
        """设置环境变量"""
        if os.path.exists(self.env_file):
            load_env_variables(self.env_file)
            logger.info(f"已加载环境变量文件: {self.env_file}")
        else:
            logger.warning(f"环境变量文件不存在: {self.env_file}")
    
    def _load_data(self):
        """加载数据集"""
        logger.info(f"加载数据集: {self.dataset}")
        
        # 数据集配置
        self.dataset_config = DATASET_CONFIGS[self.dataset]
        self.dataset_paths = DatasetPaths.from_base_path(self.data_path, self.dataset_config)
        
        # 加载Excel数据
        if self.excel_path and os.path.exists(self.excel_path):
            self.df = pd.read_excel(self.excel_path)
            logger.info(f"加载了 {len(self.df)} 条数据记录")
        else:
            raise FileNotFoundError(f"Excel文件不存在: {self.excel_path}")
        
        # 加载嵌入和描述
        self._load_embeddings_and_captions()
    
    def _load_embeddings_and_captions(self):
        """加载视频嵌入和描述"""
        # 嵌入向量保存路径 - 针对 MAFW 数据集
        embedding_base_path = Path("/home/peterchen/M2/ADEPT/data/mafw")
        video_emb_dir = embedding_base_path / "video_embeddings"
        text_emb_dir = embedding_base_path / "text_embeddings"
        
        # 加载视频嵌入
        if video_emb_dir.exists():
            # 加载所有视频嵌入
            video_emb_files = list(video_emb_dir.glob("*.npy"))
            self.video_embs = []
            self.valid_video_ids = []
            for emb_file in sorted(video_emb_files):
                video_id = emb_file.stem  # 文件名（不含扩展名）
                emb = np.load(str(emb_file))
                self.video_embs.append(emb)
                self.valid_video_ids.append(video_id)
            logger.info(f"加载了 {len(self.video_embs)} 个视频嵌入")
        else:
            raise FileNotFoundError(f"视频嵌入目录不存在: {video_emb_dir}")
        
        # 加载文本嵌入
        if text_emb_dir.exists():
            # 加载所有文本嵌入
            text_emb_files = list(text_emb_dir.glob("*.npy"))
            self.text_embs = []
            self.valid_text_ids = []
            for emb_file in sorted(text_emb_files):
                text_id = emb_file.stem  # 文件名（不含扩展名）
                emb = np.load(str(emb_file))
                self.text_embs.append(emb)
                self.valid_text_ids.append(text_id)
            logger.info(f"加载了 {len(self.text_embs)} 个文本嵌入")
        else:
            raise FileNotFoundError(f"文本嵌入目录不存在: {text_emb_dir}")
        
        # 从Excel文件创建视频描述字典 - 针对 MAFW 数据集
        self.video_captions = {}
        for idx, row in self.df.iterrows():
            video_name = row['video_name']  # MAFW数据集使用'video_name'列
            caption = row['eng_caption']
            self.video_captions[video_name] = caption
        
        logger.info(f"从Excel文件创建了 {len(self.video_captions)} 个视频描述")
    
    def _partition_data(self, sample_size: int = 10, random_state: int = 42):
        """
        数据分区：直接选择样本用于参数调优
        
        Args:
            sample_size: 样本数量
            random_state: 随机种子
        """
        logger.info(f"选择 {sample_size} 个样本用于参数调优")
        
        # 设置随机种子
        random.seed(random_state)
        np.random.seed(random_state)
        
        # 过滤出有完整嵌入的样本
        valid_indices = []
        for idx, row in self.df.iterrows():
            video_name = row['video_name']
            video_id = video_name.replace('.mp4', '')
            
            # 检查是否有视频和文本嵌入
            if video_id in self.valid_video_ids and video_id in self.valid_text_ids:
                valid_indices.append(idx)
            else:
                logger.warning(f"跳过视频 {video_name}：缺少嵌入文件")
        
        logger.info(f"有效样本数量: {len(valid_indices)} (总样本: {len(self.df)})")
        
        # 随机选择样本
        if sample_size > len(valid_indices):
            logger.warning(f"请求的样本数量 {sample_size} 超过有效数据量 {len(valid_indices)}，使用全部有效数据")
            sample_size = len(valid_indices)
        
        # 随机选择索引 - 与熵分析脚本保持一致
        random.seed(random_state)
        selected_indices = random.sample(valid_indices, min(sample_size, len(valid_indices)))
        
        # 创建查询列表
        self.queries = []
        
        # 选择样本 - 针对 MAFW 数据集
        for idx in selected_indices:
            self.queries.append({
                'video': self.df.iloc[idx]['video_name'],  # MAFW数据集使用'video_name'列
                'text': self.df.iloc[idx]['eng_caption'],
                'emotion': self.df.iloc[idx].get('label', 'unknown')  # MAFW数据集使用'label'列
            })
        
        # 获取对应的嵌入
        self.video_embs = [self.video_embs[self.valid_video_ids.index(self.df.iloc[idx]['video_name'].replace('.mp4', ''))] 
                          for idx in selected_indices]
        self.text_embs = [self.text_embs[self.valid_text_ids.index(self.df.iloc[idx]['video_name'].replace('.mp4', ''))] 
                         for idx in selected_indices]
        
        logger.info(f"数据选择完成: 选择了 {len(self.queries)} 个有效样本")
    

    
    def _init_components(self):
        """初始化MERLIN组件"""
        logger.info("初始化MERLIN组件...")
        
        # 初始化Questioner（使用搜索范围的中位数作为默认值）
        self.questioner = Questioner(
            n_clusters=8,  # 搜索范围的中位数
            alpha_threshold=0.00068,  # 搜索范围的中位数
            beta_threshold=0.475      # 搜索范围的中位数
        )
        
        # 初始化Reranker（使用环境变量中的Google Cloud配置）
        try:
            # 从环境变量读取Google Cloud配置
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
            location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
            
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT_ID 环境变量未设置")
            
            logger.info(f"使用Google Cloud配置: project_id={project_id}, location={location}")
            
            self.reranker = Reranker(
                location=location,
                project_id=project_id,
                memory_path="/path/to/memory",
                queries=self.queries,
                video_ext=self.dataset_config.video_ext
            )
            logger.info("Reranker初始化成功")
            
        except Exception as e:
            logger.warning(f"Reranker初始化失败，使用简化版本: {str(e)}")
            # 创建一个简化的Reranker
            self.reranker = SimpleReranker(self.queries, self.dataset_config.video_ext)
        
        # 初始化Answerer
        self.vqa = Answerer()
        
        logger.info("MERLIN组件初始化完成")
    
    def _get_zero_shot_ranking(self, query_text_emb: np.ndarray, video_embs: List[np.ndarray], top_k: int) -> List[int]:
        """
        零样本检索排名
        
        Args:
            query_text_emb: 查询文本嵌入
            video_embs: 视频嵌入列表
            top_k: 返回前k个结果
            
        Returns:
            排名索引列表
        """
        # 计算余弦相似度
        similarities = []
        for video_emb in video_embs:
            similarity = np.dot(query_text_emb, video_emb) / (np.linalg.norm(query_text_emb) * np.linalg.norm(video_emb))
            similarities.append(similarity)
        
        # 获取top-k索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices.tolist()
    
    def _evaluate_single_video(self, 
                              query_idx: int, 
                              params: Dict[str, Any], 
                              max_rounds: int = 3) -> Dict[str, Any]:  # 减少最大轮数
        """
        评估单个视频的参数组合
        
        Args:
            query_idx: 查询索引
            params: 参数字典 {'m': int, 'alpha': float, 'beta': float}
            max_rounds: 最大交互轮数
            
        Returns:
            评估结果字典
        """
        # 获取查询信息
        query = self.queries[query_idx]
        target_vid = query["video"].replace(self.dataset_config.video_ext, "")
        query_text_emb = self.text_embs[query_idx]
        
        # 零样本检索 - 固定使用top_k=85，与熵分析保持一致
        top_k = 85
        initial_ranking = self._get_zero_shot_ranking(query_text_emb, self.video_embs, top_k)
        
        # 获取初始排名
        try:
            target_idx = next(i for i, q in enumerate(self.queries) 
                            if q["video"].replace(self.dataset_config.video_ext, "") == target_vid)
            initial_rank = initial_ranking.index(target_idx) + 1
        except (StopIteration, ValueError):
            initial_rank = len(initial_ranking) + 1
        
        # 确保initial_rank不为None
        if initial_rank is None:
            initial_rank = len(initial_ranking) + 1
        
        # 初始化Questioner（使用当前参数）
        questioner = Questioner(
            n_clusters=params['m'],
            alpha_threshold=params['alpha'],
            beta_threshold=params['beta']
        )
        
        # 清理GPU内存
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 重置对话
        questioner.reset_conversation(target_video_id=target_vid)
        
        # 获取初始描述
        anchor_captions = ""
        for k in initial_ranking:
            try:
                anchor_captions = self.video_captions[self.queries[k]["video"]]
                break
            except:
                pass
        
        # 重置重构器
        self.reranker.reset_reformatter(initial_description=anchor_captions)
        
        # 初始化嵌入
        self.reranker.init_embedding(target_vid)
        
        # 交互循环
        current_rank = initial_rank
        final_rank = initial_rank
        final_round = 0
        
        for round_num in range(max_rounds):
            # 获取当前top-k的嵌入
            current_top_k = initial_ranking[:top_k]
            topk_embeddings = [self.video_embs[i] for i in current_top_k]
            embeddings_array = np.array(topk_embeddings) if topk_embeddings else None
            
            # 生成问题
            question_result = questioner.generate_question(
                video_captions=anchor_captions,
                embeddings=embeddings_array,
                top_k_videos=[self.queries[i]["video"].replace(self.dataset_config.video_ext, "") 
                             for i in current_top_k]
            )
            
            question = question_result["question"]
            
            # 获取答案（模拟）
            try:
                # 加载目标视频
                target_path = self.dataset_paths.find_video_path(target_vid)
                if target_path is None:
                    raise FileNotFoundError(f"Could not find target video {target_vid}")
                
                self.vqa.load_video(str(target_path))
                
                # 加载top-k视频
                topk_video_paths = []
                for idx in current_top_k:
                    vid = self.queries[idx]["video"].replace(self.dataset_config.video_ext, "")
                    video_path = self.dataset_paths.find_video_path(vid)
                    if video_path is not None:
                        topk_video_paths.append(str(video_path))
                
                if topk_video_paths:
                    # self.vqa.load_topk(topk_video_paths)  # Method removed
                    pass
                
                # 获取答案
                answer, _ = asyncio.run(self.vqa.async_ask(question))
                
            except Exception as e:
                logger.warning(f"获取答案失败: {str(e)}")
                answer = "无法获取答案"
            
            # 重构描述
            reformatted_description = self.reranker.reformat_dialogue(
                question=question,
                answer=answer,
                max_tokens=500
            )
            print(f"描述长度: {len(reformatted_description)}")
            
            # 重新排序
            emb = self.reranker.get_image_video_text_embeddings(contextual_text=reformatted_description)
            reranked_topk, target_rank = self.reranker.rerank(target_vid, self.video_embs, emb.text_embedding)
            
            current_rank = target_rank
            final_rank = target_rank
            final_round = round_num + 1
            
            # 记录答案
            reranked_top1_caption = ""
            for k in reranked_topk:
                try:
                    reranked_top1_caption = self.video_captions[k]
                    break
                except:
                    pass
            
            questioner.record_answer(
                answer=answer,
                reranked_caption=reranked_top1_caption,
                target_rank=target_rank,
                reranked_topk=reranked_topk,
                reformatted_description=reformatted_description
            )
            
            # 检查停止条件
            if target_rank == 1:
                break
        
        # 确保所有值都不为None
        if initial_rank is None:
            initial_rank = len(initial_ranking) + 1
        if final_rank is None:
            final_rank = len(initial_ranking) + 1
        
        return {
            'target_vid': target_vid,
            'initial_rank': initial_rank,
            'final_rank': final_rank,
            'final_round': final_round,
            'improvement': initial_rank - final_rank
        }
    
    def _evaluate_parameter_combination(self, params: Dict[str, Any], sample_size: int = None) -> Dict[str, Any]:
        """
        评估单个参数组合
        
        Args:
            params: 参数字典
            sample_size: 评估样本数量（None表示全部）
            
        Returns:
            评估结果
        """
        logger.info(f"评估参数组合: m={params['m']}, α={params['alpha']}, β={params['beta']}")
        
        # 确定评估样本
        if sample_size is None:
            eval_queries = self.queries
        else:
            eval_queries = self.queries[:sample_size]
        
        logger.info(f"评估样本数量: {len(eval_queries)}")
        
        # 验证样本数量
        if len(eval_queries) == 0:
            logger.error("没有可评估的样本")
            return {
                'params': params,
                'avg_final_rank': float('inf'),
                'avg_improvement': 0.0,
                'top1_accuracy': 0.0,
                'top5_accuracy': 0.0,
                'top10_accuracy': 0.0,
                'recall_at_10': 0.0,
                'num_samples': 0
            }
        
        results = []
        
        # 评估每个视频
        for i in tqdm(range(len(eval_queries)), desc=f"评估参数组合"):
            try:
                result = self._evaluate_single_video(i, params)
                results.append(result)
            except Exception as e:
                logger.error(f"评估视频 {i} 失败: {str(e)}")
                continue
        
        if not results:
            logger.warning("没有有效的评估结果")
            return {
                'params': params,
                'avg_final_rank': float('inf'),
                'avg_improvement': 0.0,
                'top1_accuracy': 0.0,
                'top5_accuracy': 0.0,
                'top10_accuracy': 0.0,
                'recall_at_10': 0.0,
                'num_samples': 0
            }
        
        # 计算指标
        final_ranks = [r['final_rank'] for r in results]
        improvements = [r['improvement'] for r in results]
        
        # 计算准确率
        top1_count = sum(1 for rank in final_ranks if rank == 1)
        top5_count = sum(1 for rank in final_ranks if rank <= 5)
        top10_count = sum(1 for rank in final_ranks if rank <= 10)
        
        # 计算Recall@10（这里我们使用最终排名在10以内的比例）
        recall_at_10 = top10_count / len(results)
        
        avg_final_rank = np.mean(final_ranks)
        avg_improvement = np.mean(improvements)
        top1_accuracy = top1_count / len(results)
        top5_accuracy = top5_count / len(results)
        top10_accuracy = top10_count / len(results)
        
        logger.info(f"参数组合评估完成: Recall@10={recall_at_10:.4f}, Top1={top1_accuracy:.4f}")
        
        return {
            'params': params,
            'avg_final_rank': avg_final_rank,
            'avg_improvement': avg_improvement,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'top10_accuracy': top10_accuracy,
            'recall_at_10': recall_at_10,
            'num_samples': len(results),
            'detailed_results': results
        }
    
    def run_grid_search(self, sample_size: int = None, save_results: bool = True) -> pd.DataFrame:
        """
        运行网格搜索
        
        Args:
            sample_size: 每个参数组合评估的样本数量
            save_results: 是否保存结果
            
        Returns:
            结果DataFrame
        """
        logger.info("开始网格搜索...")
        
        # 生成所有参数组合
        param_combinations = list(itertools.product(
            self.param_space['m'],
            self.param_space['alpha'],
            self.param_space['beta']
        ))
        
        logger.info(f"总共 {len(param_combinations)} 个参数组合需要评估")
        
        results = []
        
        # 评估每个参数组合
        for m, alpha, beta in tqdm(param_combinations, desc="网格搜索进度"):
            params = {'m': m, 'alpha': alpha, 'beta': beta}
            
            try:
                result = self._evaluate_parameter_combination(params, sample_size)
                results.append(result)
                
                # 更新最佳参数
                if result['recall_at_10'] > self.best_score:
                    self.best_score = result['recall_at_10']
                    self.best_params = params
                    logger.info(f"发现新的最佳参数: {params}, Recall@10={self.best_score:.4f}")
                
            except Exception as e:
                logger.error(f"评估参数组合失败: {params}, 错误: {str(e)}")
                continue
        
        # 转换为DataFrame
        df_results = []
        for result in results:
            df_results.append({
                'm': result['params']['m'],
                'alpha': result['params']['alpha'],
                'beta': result['params']['beta'],
                'recall_at_10': result['recall_at_10'],
                'top1_accuracy': result['top1_accuracy'],
                'top5_accuracy': result['top5_accuracy'],
                'top10_accuracy': result['top10_accuracy'],
                'avg_final_rank': result['avg_final_rank'],
                'avg_improvement': result['avg_improvement'],
                'num_samples': result['num_samples']
            })
        
        self.results = results
        df = pd.DataFrame(df_results)
        
        # 保存结果
        if save_results:
            self._save_results(df)
        
        return df
    
    def _save_results(self, df: pd.DataFrame):
        """保存结果到文件"""
        # 创建输出目录 - 保存到 MAFW 数据集的 best_parameter 目录
        output_dir = Path("/home/peterchen/M2/ADEPT/data/mafw/best_parameter")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存DataFrame
        df.to_csv(output_dir / f"grid_search_results_{self.dataset}.csv", index=False)
        
        # 保存详细结果
        with open(output_dir / f"detailed_results_{self.dataset}.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # 保存最佳参数
        if self.best_params:
            best_params_info = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'dataset': self.dataset,
                'timestamp': time.time()
            }
            with open(output_dir / f"best_params_{self.dataset}.json", 'w') as f:
                json.dump(best_params_info, f, indent=2)
        
        logger.info(f"结果已保存到: {output_dir}")
    
    def create_heatmaps(self, df: pd.DataFrame, save_plots: bool = True):
        """
        创建热力图可视化
        
        Args:
            df: 结果DataFrame
            save_plots: 是否保存图片
        """
        logger.info("创建热力图...")
        
        # 创建输出目录 - 保存到 MAFW 数据集的 best_parameter 目录
        output_dir = Path("/home/peterchen/M2/ADEPT/data/mafw/best_parameter/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每个m值创建热力图
        for m in self.param_space['m']:
            # 筛选当前m的数据
            m_data = df[df['m'] == m].copy()
            
            if m_data.empty:
                logger.warning(f"没有m={m}的数据")
                continue
            
            # 创建pivot table
            pivot_data = m_data.pivot(index='beta', columns='alpha', values='recall_at_10')
            
            # 创建热力图
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='viridis', 
                       cbar_kws={'label': 'Recall@10'})
            plt.title(f'Recall@10 热力图 (m={m}) - MAFW Dataset')
            plt.xlabel('α (簇间熵阈值)')
            plt.ylabel('β (簇内熵阈值)')
            
            if save_plots:
                plt.savefig(output_dir / f"heatmap_m{m}_{self.dataset}.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        
        # 创建综合热力图（所有m值的平均值）
        plt.figure(figsize=(12, 10))
        
        # 计算每个(alpha, beta)组合的平均Recall@10
        avg_data = df.groupby(['alpha', 'beta'])['recall_at_10'].mean().reset_index()
        pivot_avg = avg_data.pivot(index='beta', columns='alpha', values='recall_at_10')
        
        sns.heatmap(pivot_avg, annot=True, fmt='.4f', cmap='viridis',
                   cbar_kws={'label': '平均 Recall@10'})
        plt.title(f'平均 Recall@10 热力图 (所有m值) - MAFW Dataset')
        plt.xlabel('α (簇间熵阈值)')
        plt.ylabel('β (簇内熵阈值)')
        
        if save_plots:
            plt.savefig(output_dir / f"heatmap_avg_{self.dataset}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        logger.info(f"热力图已保存到: {output_dir}")
    

    
    def run_complete_tuning(self, sample_size: int = None):
        """
        运行完整的参数调优流程
        
        Args:
            sample_size: 每个参数组合评估的样本数量
        """
        logger.info("开始完整的参数调优流程...")
        
        # 1. 网格搜索
        df_results = self.run_grid_search(sample_size=sample_size)
        
        # 2. 创建可视化
        self.create_heatmaps(df_results)
        
        # 3. 输出总结
        self._print_summary(df_results)
        
        logger.info("参数调优流程完成！")
    
    def _print_summary(self, df_results: pd.DataFrame):
        """打印调优总结"""
        print("\n" + "="*50)
        print("MAFW 数据集参数调优总结")
        print("="*50)
        
        if self.best_params:
            print(f"最佳参数: m={self.best_params['m']}, α={self.best_params['alpha']}, β={self.best_params['beta']}")
            print(f"Recall@10: {self.best_score:.4f}")
        
        print(f"总参数组合数: {len(df_results)}")
        print(f"评估样本数: {len(self.queries)}")
        print(f"结果保存位置: /home/peterchen/M2/ADEPT/data/mafw/best_parameter/")
        print("="*50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MERLIN 参数调优系统 - MAFW 数据集")
    
    # 数据集参数
    parser.add_argument("--dataset", type=str, choices=["mafw"], required=True, help="数据集名称")
    parser.add_argument("--data_path", type=str, default="/home/peterchen/M2/ADEPT/data/mafw", help="数据路径")
    parser.add_argument("--excel_path", type=str, 
                       default="/home/peterchen/M2/ADEPT/data/mafw/labels/sampled_850.xlsx",
                       help="Excel文件路径（MAFW数据集）")
    parser.add_argument("--video_base_dir", type=str,
                       default="/home/peterchen/M2/ADEPT/data/mafw/videos",
                       help="视频基础目录（MAFW数据集）")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="outputs", help="输出目录")
    parser.add_argument("--env_file", type=str, default="/home/peterchen/M2/ADEPT/.env", help="环境变量文件")
    
    # 调优参数
    parser.add_argument("--sample_size", type=int, default=10, help="用于参数调优的样本数量")
    
    args = parser.parse_args()
    
    try:
        # 创建参数调优器
        tuner = ParameterTuner(
            dataset=args.dataset,
            data_path=args.data_path,
            excel_path=args.excel_path,
            video_base_dir=args.video_base_dir,
            output_dir=args.output_dir,
            env_file=args.env_file,
            sample_size=args.sample_size
        )
        
        # 运行完整调优流程
        tuner.run_complete_tuning(sample_size=args.sample_size)
        
    except Exception as e:
        logger.error(f"参数调优失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 