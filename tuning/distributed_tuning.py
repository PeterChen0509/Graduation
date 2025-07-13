#!/usr/bin/env python3
"""
真正的分布式参数调优系统
支持多GPU并行处理视频对话，大幅加速参数调优过程
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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import itertools
import torch
import multiprocessing as mp
from functools import partial

# 自动设置PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from parameter_tuning import ParameterTuner
from utils.logger import logger, setup_logger

class TrueDistributedParameterTuner:
    """
    真正的分布式参数调优器
    支持多GPU并行处理视频对话，每个GPU处理一个视频
    """
    
    def __init__(self, 
                 dataset: str,
                 data_path: str,
                 excel_path: str = None,
                 video_base_dir: str = None,
                 output_dir: str = "outputs",
                 env_file: str = "/home/peterchen/M2/ADEPT/.env",
                 n_gpus: int = None):
        """
        初始化真正的分布式参数调优器
        
        Args:
            n_gpus: 使用的GPU数量，None表示自动检测
        """
        # 自动检测GPU数量
        if n_gpus is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                n_gpus = 1  # 如果没有GPU，使用CPU
                logger.warning("未检测到GPU，将使用CPU模式")
        
        self.n_gpus = n_gpus
        self.dataset = dataset
        self.data_path = data_path
        self.excel_path = excel_path
        self.video_base_dir = video_base_dir
        self.output_dir = output_dir
        self.env_file = env_file
        
        # 结果存储
        self.results = []
        self.best_params = None
        self.best_score = 0.0
        
        logger.info(f"真正的分布式调优器初始化完成，使用 {n_gpus} 个GPU")
        logger.info(f"每个GPU将并行处理一个视频对话")
    
    def _evaluate_single_video_parallel(self, 
                                      video_idx: int, 
                                      params: Dict[str, Any],
                                      gpu_id: int,
                                      val_queries: List,
                                      val_video_embs: List,
                                      val_text_embs: List,
                                      video_captions: Dict,
                                      dataset_config: Any,
                                      dataset_paths: Any) -> Dict[str, Any]:
        """
        在指定GPU上评估单个视频
        
        Args:
            video_idx: 视频索引
            params: 参数字典
            gpu_id: GPU ID
            val_queries: 验证集查询
            val_video_embs: 验证集视频嵌入
            val_text_embs: 验证集文本嵌入
            video_captions: 视频描述
            dataset_config: 数据集配置
            dataset_paths: 数据集路径
            
        Returns:
            评估结果字典
        """
        try:
            # 设置GPU
            if torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                device = f"cuda:{gpu_id}"
            else:
                device = "cpu"
            
            # 创建独立的tuner实例（避免GPU冲突）
            tuner = ParameterTuner(
                dataset=self.dataset,
                data_path=self.data_path,
                excel_path=self.excel_path,
                video_base_dir=self.video_base_dir,
                output_dir=self.output_dir,
                env_file=self.env_file
            )
            
            # 使用传入的数据（避免重复加载）
            tuner.val_queries = val_queries
            tuner.val_video_embs = val_video_embs
            tuner.val_text_embs = val_text_embs
            tuner.video_captions = video_captions
            tuner.dataset_config = dataset_config
            tuner.dataset_paths = dataset_paths
            
            # 评估单个视频
            result = tuner._evaluate_single_video(video_idx, params)
            
            logger.info(f"GPU {gpu_id} 完成视频 {video_idx}: 初始排名={result['initial_rank']}, 最终排名={result['final_rank']}")
            
            return result
            
        except Exception as e:
            logger.error(f"GPU {gpu_id} 评估视频 {video_idx} 失败: {str(e)}")
            return {
                'target_vid': f"video_{video_idx}",
                'initial_rank': float('inf'),
                'final_rank': float('inf'),
                'final_round': 0,
                'improvement': 0,
                'error': str(e)
            }
    
    def _evaluate_parameter_combination_parallel(self, params: Dict[str, Any], sample_size: int = 50) -> Dict[str, Any]:
        """
        并行评估单个参数组合
        
        Args:
            params: 参数字典
            sample_size: 评估样本数量
            
        Returns:
            评估结果
        """
        logger.info(f"开始并行评估参数组合: m={params['m']}, α={params['alpha']}, β={params['beta']}")
        
        # 创建基础tuner来获取数据
        base_tuner = ParameterTuner(
            dataset=self.dataset,
            data_path=self.data_path,
            excel_path=self.excel_path,
            video_base_dir=self.video_base_dir,
            output_dir=self.output_dir,
            env_file=self.env_file
        )
        
        # 确定评估样本
        if sample_size is None:
            eval_queries = base_tuner.val_queries
        else:
            eval_queries = base_tuner.val_queries[:sample_size]
        
        logger.info(f"并行评估样本数量: {len(eval_queries)}")
        
        # 准备共享数据
        val_queries = base_tuner.val_queries
        val_video_embs = base_tuner.val_video_embs
        val_text_embs = base_tuner.val_text_embs
        video_captions = base_tuner.video_captions
        dataset_config = base_tuner.dataset_config
        dataset_paths = base_tuner.dataset_paths
        
        # 分配GPU任务
        video_indices = list(range(len(eval_queries)))
        gpu_assignments = []
        
        for i, video_idx in enumerate(video_indices):
            gpu_id = i % self.n_gpus
            gpu_assignments.append((video_idx, gpu_id))
        
        logger.info(f"任务分配: {len(gpu_assignments)} 个视频分配到 {self.n_gpus} 个GPU")
        
        # 并行处理
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=self.n_gpus) as executor:
            # 提交所有任务
            future_to_video = {}
            for video_idx, gpu_id in gpu_assignments:
                future = executor.submit(
                    self._evaluate_single_video_parallel,
                    video_idx, params, gpu_id,
                    val_queries, val_video_embs, val_text_embs,
                    video_captions, dataset_config, dataset_paths
                )
                future_to_video[future] = video_idx
            
            # 收集结果
            for future in tqdm(as_completed(future_to_video), 
                             total=len(gpu_assignments), 
                             desc=f"并行评估参数组合"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    video_idx = future_to_video[future]
                    logger.error(f"视频 {video_idx} 评估失败: {str(e)}")
                    results.append({
                        'target_vid': f"video_{video_idx}",
                        'initial_rank': float('inf'),
                        'final_rank': float('inf'),
                        'final_round': 0,
                        'improvement': 0,
                        'error': str(e)
                    })
        
        end_time = time.time()
        parallel_time = end_time - start_time
        
        # 过滤有效结果
        valid_results = [r for r in results if 'error' not in r]
        error_results = [r for r in results if 'error' in r]
        
        logger.info(f"并行评估完成: 成功={len(valid_results)}, 失败={len(error_results)}")
        logger.info(f"并行耗时: {parallel_time:.2f}秒")
        
        if not valid_results:
            logger.warning("没有有效的评估结果")
            return {
                'params': params,
                'avg_final_rank': float('inf'),
                'avg_improvement': 0.0,
                'top1_accuracy': 0.0,
                'top5_accuracy': 0.0,
                'top10_accuracy': 0.0,
                'recall_at_10': 0.0,
                'num_samples': 0,
                'parallel_time': parallel_time
            }
        
        # 计算指标
        final_ranks = [r['final_rank'] for r in valid_results]
        improvements = [r['improvement'] for r in valid_results]
        
        # 计算准确率
        top1_count = sum(1 for rank in final_ranks if rank == 1)
        top5_count = sum(1 for rank in final_ranks if rank <= 5)
        top10_count = sum(1 for rank in final_ranks if rank <= 10)
        
        # 计算Recall@10
        recall_at_10 = top10_count / len(valid_results)
        
        avg_final_rank = np.mean(final_ranks)
        avg_improvement = np.mean(improvements)
        top1_accuracy = top1_count / len(valid_results)
        top5_accuracy = top5_count / len(valid_results)
        top10_accuracy = top10_count / len(valid_results)
        
        logger.info(f"参数组合并行评估完成: Recall@10={recall_at_10:.4f}, Top1={top1_accuracy:.4f}")
        
        return {
            'params': params,
            'avg_final_rank': avg_final_rank,
            'avg_improvement': avg_improvement,
            'top1_accuracy': top1_accuracy,
            'top5_accuracy': top5_accuracy,
            'top10_accuracy': top10_accuracy,
            'recall_at_10': recall_at_10,
            'num_samples': len(valid_results),
            'detailed_results': valid_results,
            'parallel_time': parallel_time
        }
    
    def run_parallel_grid_search(self, save_results: bool = True) -> pd.DataFrame:
        """
        运行并行网格搜索
        
        Args:
            save_results: 是否保存结果
            
        Returns:
            结果DataFrame
        """
        logger.info("开始真正的并行网格搜索...")
        
        # 生成所有参数组合
        param_space = {
            'm': [5, 8, 10, 12],  # K-means簇数
            'alpha': [0.3, 0.5, 0.7],  # 簇间熵阈值
            'beta': [0.2, 0.4, 0.6]   # 簇内熵阈值
        }
        
        param_combinations = list(itertools.product(
            param_space['m'],
            param_space['alpha'],
            param_space['beta']
        ))
        
        logger.info(f"总共 {len(param_combinations)} 个参数组合需要评估")
        logger.info(f"使用 {self.n_gpus} 个GPU并行处理")
        
        # 估算时间
        estimated_time_per_combination = 2 * 50 / self.n_gpus / 60  # 分钟
        total_estimated_time = len(param_combinations) * estimated_time_per_combination
        logger.info(f"预计总时间: {total_estimated_time:.1f} 分钟 ({total_estimated_time/60:.1f} 小时)")
        
        # 串行评估参数组合（但每个组合内部并行）
        start_time = time.time()
        results = []
        
        for m, alpha, beta in tqdm(param_combinations, desc="参数组合进度"):
            params = {'m': m, 'alpha': alpha, 'beta': beta}
            
            try:
                result = self._evaluate_parameter_combination_parallel(params, sample_size=50)
                results.append(result)
                
                # 更新最佳参数
                if result['recall_at_10'] > self.best_score:
                    self.best_score = result['recall_at_10']
                    self.best_params = params
                    logger.info(f"发现新的最佳参数: {params}, Recall@10={self.best_score:.4f}")
                
            except Exception as e:
                logger.error(f"评估参数组合失败: {params}, 错误: {str(e)}")
                continue
        
        end_time = time.time()
        actual_time = (end_time - start_time) / 60  # 分钟
        logger.info(f"实际耗时: {actual_time:.1f} 分钟")
        if total_estimated_time > 0:
            logger.info(f"加速比: {total_estimated_time/actual_time:.1f}x")
        
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
                'num_samples': result['num_samples'],
                'parallel_time': result.get('parallel_time', 0)
            })
        
        self.results = results
        df = pd.DataFrame(df_results)
        
        # 保存结果
        if save_results:
            self._save_parallel_results(df)
        
        return df
    
    def _save_parallel_results(self, df: pd.DataFrame):
        """保存并行调优结果"""
        output_dir = Path("/home/peterchen/M2/ADEPT/data/mafw/best_parameter")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存DataFrame
        df.to_csv(output_dir / f"parallel_grid_search_results_{self.dataset}.csv", index=False)
        
        # 保存详细结果
        with open(output_dir / f"parallel_detailed_results_{self.dataset}.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # 保存最佳参数
        if self.best_params:
            best_params_info = {
                'best_params': self.best_params,
                'best_score': self.best_score,
                'dataset': self.dataset,
                'timestamp': time.time(),
                'n_gpus': self.n_gpus,
                'total_combinations': len(self.results)
            }
            with open(output_dir / f"parallel_best_params_{self.dataset}.json", 'w') as f:
                json.dump(best_params_info, f, indent=2)
        
        logger.info(f"并行调优结果已保存到: {output_dir}")
    
    def run_complete_parallel_tuning(self):
        """运行完整的并行调优流程"""
        logger.info("开始完整的并行参数调优流程...")
        
        # 运行并行网格搜索
        df_results = self.run_parallel_grid_search(save_results=True)
        
        # 在测试集上评估最佳参数
        if self.best_params:
            logger.info(f"在测试集上评估最佳参数: {self.best_params}")
            
            # 创建基础tuner进行测试集评估
            base_tuner = ParameterTuner(
                dataset=self.dataset,
                data_path=self.data_path,
                excel_path=self.excel_path,
                video_base_dir=self.video_base_dir,
                output_dir=self.output_dir,
                env_file=self.env_file
            )
            
            test_result = base_tuner.evaluate_best_params_on_test_set()
            
            # 保存测试结果
            output_dir = Path("/home/peterchen/M2/ADEPT/data/mafw/best_parameter")
            test_result_info = {
                'best_params': self.best_params,
                'test_results': test_result,
                'dataset': self.dataset,
                'timestamp': time.time(),
                'n_gpus': self.n_gpus
            }
            
            with open(output_dir / f"parallel_test_evaluation_{self.dataset}.json", 'w') as f:
                json.dump(test_result_info, f, indent=2, default=str)
            
            logger.info(f"并行调优完成！最佳参数: {self.best_params}")
            logger.info(f"测试集Recall@10: {test_result.get('recall_at_10', 0):.4f}")
        else:
            logger.error("没有找到最佳参数")

# 保持向后兼容性
class DistributedParameterTuner(TrueDistributedParameterTuner):
    """向后兼容的分布式参数调优器"""
    pass

def main():
    parser = argparse.ArgumentParser(description="真正的多GPU并行MERLIN参数调优")
    
    # 必需参数
    parser.add_argument("--dataset", type=str, choices=["mafw"], default="mafw", 
                       help="数据集名称")
    
    # 可选参数
    parser.add_argument("--data_path", type=str, default="data", 
                       help="数据路径")
    parser.add_argument("--excel_path", type=str, 
                       default="/home/peterchen/M2/ADEPT/data/mafw/labels/sampled_850.xlsx",
                       help="Excel文件路径")
    parser.add_argument("--video_base_dir", type=str,
                       default="/home/peterchen/M2/MAFW/data/clips/unzip",
                       help="视频基础目录")
    parser.add_argument("--output_dir", type=str, default="outputs", 
                       help="输出目录")
    parser.add_argument("--env_file", type=str, 
                       default="/home/peterchen/M2/ADEPT/.env",
                       help="环境变量文件路径")
    
    # GPU参数
    parser.add_argument("--n_gpus", type=int, default=None,
                       help="使用的GPU数量 (默认: 自动检测)")
    
    args = parser.parse_args()
    
    print(f"🚀 开始真正的多GPU并行 MERLIN 参数调优")
    print(f"📊 数据集: {args.dataset}")
    print(f"🔢 GPU数量: {args.n_gpus or '自动检测'}")
    print(f"📁 数据路径: {args.data_path}")
    print(f"📊 Excel文件: {args.excel_path}")
    print(f"🎬 视频目录: {args.video_base_dir}")
    
    try:
        # 创建真正的分布式调优器
        parallel_tuner = TrueDistributedParameterTuner(
            dataset=args.dataset,
            data_path=args.data_path,
            excel_path=args.excel_path,
            video_base_dir=args.video_base_dir,
            output_dir=args.output_dir,
            env_file=args.env_file,
            n_gpus=args.n_gpus
        )
        
        # 运行完整并行调优流程
        parallel_tuner.run_complete_parallel_tuning()
        
        print("✅ 真正的并行参数调优完成！")
        print(f"📊 结果保存在: /home/peterchen/M2/ADEPT/data/mafw/best_parameter/")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断了并行调优过程")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 并行参数调优失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 