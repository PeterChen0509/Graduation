#!/usr/bin/env python3
"""
使用真正 Google embedding 的 MERLIN 参数调优系统
用于找到最优的 (m, α, β) 参数组合

实验设计：
阶段一：熵值分布测量 - 使用10-20个样本测量真实的熵值分布
阶段二：基于分布结果确定参数搜索空间
阶段三：网格搜索找到最佳参数组合
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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# 自动设置PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logger import logger, setup_logger
from utils.env_utils import load_env_variables, get_required_env
from merlin.questioner import Questioner
from merlin.reranker import Reranker

class EntropyDistributionAnalyzer:
    """
    熵值分布分析器 - 用于测量系统内在的熵值分布情况
    """
    
    def __init__(self, 
                 data_path: str = "/home/peterchen/M2/ADEPT/data/mafw",
                 excel_path: str = "/home/peterchen/M2/ADEPT/data/mafw/labels/sampled_850.xlsx",
                 video_dir: str = "/home/peterchen/M2/ADEPT/data/mafw/videos",
                 output_dir: str = "/home/peterchen/M2/ADEPT/data/mafw/entropy_analysis_outputs",
                 env_file: str = "/home/peterchen/M2/ADEPT/.env",
                 sample_size: int = 400):
        """
        初始化熵值分布分析器
        
        Args:
            data_path: 数据路径
            excel_path: Excel文件路径
            video_dir: 视频目录
            output_dir: 输出目录
            env_file: 环境变量文件路径
        """
        self.data_path = data_path
        self.excel_path = excel_path
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.env_file = env_file
        
        # 设置环境
        self._setup_environment()
        
        # 加载数据
        self._load_data()
        
        # 实验参数
        self.sample_size = sample_size  # 查询样本数量
        self.k = 85  # top-k = 85 (数据集的约10%)
        self.m = 9  # 固定簇数用于分布测量
        
        # 初始化组件
        self._init_components()
        
        logger.info(f"熵值分布分析器初始化完成")
        logger.info(f"样本数量: {self.sample_size}, k: {self.k}, m: {self.m}")
    
    def _setup_environment(self):
        """设置环境变量"""
        if os.path.exists(self.env_file):
            load_env_variables(self.env_file)
            logger.info(f"已加载环境变量文件: {self.env_file}")
        else:
            logger.warning(f"环境变量文件不存在: {self.env_file}")
        
        # 验证必要的环境变量
        try:
            self.project_id = get_required_env("GOOGLE_CLOUD_PROJECT_ID")
            self.location = get_required_env("GOOGLE_CLOUD_LOCATION")
            logger.info(f"Google Cloud 配置: Project ID = {self.project_id}, Location = {self.location}")
        except Exception as e:
            logger.error(f"缺少必要的 Google Cloud 环境变量: {str(e)}")
            raise
    
    def _load_data(self):
        """加载MAFW数据集"""
        logger.info(f"加载MAFW数据集: {self.excel_path}")
        
        # 加载Excel数据
        if self.excel_path and os.path.exists(self.excel_path):
            self.df = pd.read_excel(self.excel_path)
            logger.info(f"加载了 {len(self.df)} 条数据记录")
        else:
            raise FileNotFoundError(f"Excel文件不存在: {self.excel_path}")
        
        # 检查必要的列
        required_columns = ['video_name', 'eng_caption']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Excel文件缺少必要的列: {missing_columns}")
        
        # 创建查询列表
        self.queries = []
        for _, row in self.df.iterrows():
            self.queries.append({
                "video": row['video_name'],
                "caption": row['eng_caption'],
                "video_path": os.path.join(self.video_dir, f"{row['video_name']}.mp4")
            })
        
        logger.info(f"创建了 {len(self.queries)} 个查询")
    
    def _init_components(self):
        """初始化组件"""
        logger.info("初始化 MERLIN 组件...")
        
        # 初始化 Questioner（用于熵值计算）
        self.questioner = Questioner(
            n_clusters=self.m,
            alpha_threshold=0.5,  # 临时值，不影响熵值计算
            beta_threshold=0.3    # 临时值，不影响熵值计算
        )
        
        # 初始化 Reranker（使用 Google embedding）
        try:
            self.reranker = Reranker(
                location=self.location,
                project_id=self.project_id,
                memory_path=self.data_path,
                queries=self.queries,
                video_ext=".mp4"
            )
            logger.info("✅ 成功初始化真正的 Google embedding Reranker")
        except Exception as e:
            logger.error(f"❌ Reranker 初始化失败: {str(e)}")
            raise
    
    def _get_zero_shot_ranking(self, query_text_emb: np.ndarray, video_embs: List[np.ndarray], top_k: int) -> List[int]:
        """零样本检索排名"""
        # 计算余弦相似度
        similarities = []
        for video_emb in video_embs:
            similarity = np.dot(query_text_emb, video_emb) / (np.linalg.norm(query_text_emb) * np.linalg.norm(video_emb))
            similarities.append(similarity)
        
        # 获取top-k索引
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices.tolist()
    
    def _load_video_embeddings(self) -> List[np.ndarray]:
        """加载所有视频的嵌入向量"""
        logger.info("加载视频嵌入向量...")
        
        # 检查是否有预计算的嵌入
        embedding_dir = Path(self.data_path) / "video_embeddings"
        if embedding_dir.exists():
            # 加载预计算的嵌入
            video_embs = []
            valid_queries = []
            for query in self.queries:
                video_name = query["video"]
                video_id = video_name.replace('.mp4', '')  # 去掉.mp4后缀
                emb_file = embedding_dir / f"{video_id}.npy"
                if emb_file.exists():
                    emb = np.load(str(emb_file))
                    video_embs.append(emb)
                    valid_queries.append(query)
                else:
                    logger.warning(f"跳过视频 {video_name}：缺少视频嵌入文件")
            
            # 更新queries列表，只保留有效的查询
            self.queries = valid_queries
            logger.info(f"加载了 {len(video_embs)} 个有效视频嵌入")
            return video_embs
        else:
            logger.warning("未找到预计算的嵌入，将使用随机向量")
            # 使用随机向量作为占位符
            return [np.random.randn(1408) for _ in self.queries]
    
    def _load_text_embeddings(self) -> List[np.ndarray]:
        """加载所有文本的嵌入向量"""
        logger.info("加载文本嵌入向量...")
        
        # 检查是否有预计算的嵌入
        embedding_dir = Path(self.data_path) / "text_embeddings"
        if embedding_dir.exists():
            # 加载预计算的嵌入
            text_embs = []
            for query in self.queries:
                video_name = query["video"]
                video_id = video_name.replace('.mp4', '')  # 去掉.mp4后缀
                emb_file = embedding_dir / f"{video_id}.npy"
                if emb_file.exists():
                    emb = np.load(str(emb_file))
                    text_embs.append(emb)
                else:
                    logger.warning(f"跳过视频 {video_name}：缺少文本嵌入文件")
                    # 使用随机向量作为占位符
                    text_embs.append(np.random.randn(1408))
            
            logger.info(f"加载了 {len(text_embs)} 个文本嵌入")
            return text_embs
        else:
            logger.warning("未找到预计算的嵌入，将使用随机向量")
            # 使用随机向量作为占位符
            return [np.random.randn(1408) for _ in self.queries]
    
    def measure_entropy_distribution(self) -> pd.DataFrame:
        """测量熵值分布"""
        logger.info("开始熵值分布测量...")
        
        # 加载嵌入向量
        video_embs = self._load_video_embeddings()
        text_embs = self._load_text_embeddings()
        
        # 确保样本数量不超过查询数量
        actual_sample_size = min(self.sample_size, len(self.queries))
        logger.info(f"查询数量: {len(self.queries)}, 实际采样数量: {actual_sample_size}")
        
        # 随机选择查询样本
        random.seed(42)
        sample_indices = random.sample(range(len(self.queries)), actual_sample_size)
        
        # 记录结果
        results = []
        
        for i, query_idx in enumerate(tqdm(sample_indices, desc="测量熵值分布")):
            try:
                # 获取查询信息
                query = self.queries[query_idx]
                query_text_emb = text_embs[query_idx]
                
                # 创建候选视频列表（排除查询视频本身）
                candidate_indices = [j for j in range(len(self.queries)) if j != query_idx]
                candidate_video_embs = [video_embs[j] for j in candidate_indices]
                
                # 零样本检索，获取top-k候选
                top_k_indices = self._get_zero_shot_ranking(query_text_emb, candidate_video_embs, self.k)
                
                # 获取top-k候选的嵌入向量
                top_k_embeddings = [candidate_video_embs[idx] for idx in top_k_indices]
                embeddings_array = np.array(top_k_embeddings)
                
                # 计算熵值
                inter_cluster_entropy, intra_cluster_entropy, cluster_info = self.questioner.compute_entropy_metrics(embeddings_array)
                
                # 记录结果
                result = {
                    'Query_ID': f"Sample_{i+1:03d}",
                    'k': self.k,
                    'm': self.m,
                    'Calculated_Inter_Cluster_Entropy': inter_cluster_entropy,
                    'Calculated_Intra_Cluster_Entropy': intra_cluster_entropy,
                    'query_video': query["video"],
                    'query_caption': query["caption"][:100] + "..." if len(query["caption"]) > 100 else query["caption"]
                }
                results.append(result)
                
                logger.debug(f"样本 {i+1}: 簇间熵={inter_cluster_entropy:.4f}, 簇内熵={intra_cluster_entropy:.4f}")
                
            except Exception as e:
                logger.error(f"处理样本 {i+1} 失败: {str(e)}")
                continue
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 保存结果
        self._save_entropy_results(df)
        
        return df
    
    def _save_entropy_results(self, df: pd.DataFrame):
        """保存熵值分布结果"""
        output_path = Path(self.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存CSV
        csv_path = output_path / "entropy_distribution_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"熵值分布结果已保存到: {csv_path}")
        
        # 保存JSON（用于后续分析）
        json_path = output_path / "entropy_distribution_results.json"
        df.to_json(json_path, orient='records', indent=2)
        logger.info(f"熵值分布结果已保存到: {json_path}")
    
    def analyze_entropy_distribution(self, df: pd.DataFrame):
        """分析熵值分布并生成可视化"""
        logger.info("分析熵值分布...")
        
        output_path = Path(self.output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MAFW Dataset - Entropy Distribution Analysis Results', fontsize=16)
        
        # 1. 簇间熵分布直方图
        axes[0,0].hist(df['Calculated_Inter_Cluster_Entropy'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_xlabel('Inter-Cluster Entropy (α)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Inter-Cluster Entropy Distribution')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 平均簇内熵分布直方图
        axes[0,1].hist(df['Calculated_Intra_Cluster_Entropy'], bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0,1].set_xlabel('Intra-Cluster Entropy (β)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Intra-Cluster Entropy Distribution')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 散点图：簇间熵 vs 簇内熵
        scatter = axes[1,0].scatter(df['Calculated_Inter_Cluster_Entropy'], 
                                  df['Calculated_Intra_Cluster_Entropy'], 
                                  alpha=0.6, s=50)
        axes[1,0].set_xlabel('Inter-Cluster Entropy (α)')
        axes[1,0].set_ylabel('Intra-Cluster Entropy (β)')
        axes[1,0].set_title('Inter-Cluster vs Intra-Cluster Entropy')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 箱线图
        data_for_box = [df['Calculated_Inter_Cluster_Entropy'], df['Calculated_Intra_Cluster_Entropy']]
        axes[1,1].boxplot(data_for_box, labels=['Inter-Cluster Entropy (α)', 'Intra-Cluster Entropy (β)'])
        axes[1,1].set_ylabel('Entropy Value')
        axes[1,1].set_title('Entropy Box Plot')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = output_path / "entropy_distribution_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"熵值分布分析图表已保存到: {plot_path}")
        
        plt.show()
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """计算统计量"""
        logger.info("计算统计量...")
        
        # 计算百分位数
        alpha_stats = df['Calculated_Inter_Cluster_Entropy'].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
        beta_stats = df['Calculated_Intra_Cluster_Entropy'].describe(percentiles=[0.25, 0.5, 0.75, 0.9])
        
        statistics = {
            'alpha_statistics': {
                'min': alpha_stats['min'],
                '25%': alpha_stats['25%'],
                '50%': alpha_stats['50%'],
                '75%': alpha_stats['75%'],
                '90%': alpha_stats['90%'],
                'max': alpha_stats['max'],
                'mean': alpha_stats['mean'],
                'std': alpha_stats['std']
            },
            'beta_statistics': {
                'min': beta_stats['min'],
                '25%': beta_stats['25%'],
                '50%': beta_stats['50%'],
                '75%': beta_stats['75%'],
                '90%': beta_stats['90%'],
                'max': beta_stats['max'],
                'mean': beta_stats['mean'],
                'std': beta_stats['std']
            }
        }
        
        # 保存统计结果
        output_path = Path(self.output_dir)
        output_path.mkdir(exist_ok=True)
        
        stats_path = output_path / "entropy_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        logger.info(f"统计结果已保存到: {stats_path}")
        
        return statistics
    
    def suggest_parameter_ranges(self, statistics: Dict[str, Any]) -> Dict[str, List[float]]:
        """基于统计结果建议参数搜索范围"""
        logger.info("基于统计结果建议参数搜索范围...")
        
        alpha_stats = statistics['alpha_statistics']
        beta_stats = statistics['beta_statistics']
        
        # 基于百分位数确定搜索范围
        # 对于 Alpha，使用更小的偏移量（因为值域较小）
        alpha_offset = min(0.001, alpha_stats['std'] * 2)  # 使用标准差或0.001的较小值
        alpha_range = [
            round(alpha_stats['25%'] - alpha_offset, 4),  # 25%百分位数附近
            round(alpha_stats['25%'], 4),
            round(alpha_stats['50%'], 4),                 # 50%百分位数
            round(alpha_stats['75%'], 4),                 # 75%百分位数
            round(alpha_stats['90%'], 4),                 # 90%百分位数
            round(alpha_stats['90%'] + alpha_offset, 4)   # 90%百分位数附近
        ]
        
        # 对于 Beta，使用相对偏移量
        beta_offset = min(0.01, beta_stats['std'] * 2)  # 使用标准差或0.01的较小值
        beta_range = [
            round(beta_stats['25%'] - beta_offset, 3),   # 25%百分位数附近
            round(beta_stats['25%'], 3),
            round(beta_stats['50%'], 3),                 # 50%百分位数
            round(beta_stats['75%'], 3),                 # 75%百分位数
            round(beta_stats['90%'], 3),                 # 90%百分位数
            round(beta_stats['90%'] + beta_offset, 3)    # 90%百分位数附近
        ]
        
        # 去重并排序
        alpha_range = sorted(list(set(alpha_range)))
        beta_range = sorted(list(set(beta_range)))
        
        # m的范围（基于实验设计）
        m_range = [4, 6, 8, 10, 12]
        
        suggested_ranges = {
            'm': m_range,
            'alpha': alpha_range,
            'beta': beta_range
        }
        
        # 保存建议的参数范围
        output_path = Path(self.output_dir)
        output_path.mkdir(exist_ok=True)
        
        ranges_path = output_path / "suggested_parameter_ranges.json"
        with open(ranges_path, 'w') as f:
            json.dump(suggested_ranges, f, indent=2)
        logger.info(f"建议的参数范围已保存到: {ranges_path}")
        
        return suggested_ranges
    
    def run_entropy_analysis(self):
        """运行完整的熵值分析流程"""
        logger.info("开始完整的熵值分布分析流程...")
        
        # 1. 测量熵值分布
        df = self.measure_entropy_distribution()
        
        # 2. 分析熵值分布
        self.analyze_entropy_distribution(df)
        
        # 3. 计算统计量
        statistics = self.calculate_statistics(df)
        
        # 4. 建议参数搜索范围
        suggested_ranges = self.suggest_parameter_ranges(statistics)
        
        # 5. 打印总结
        self._print_entropy_summary(df, statistics, suggested_ranges)
    
    def _print_entropy_summary(self, df: pd.DataFrame, statistics: Dict[str, Any], suggested_ranges: Dict[str, List[float]]):
        """打印熵值分析总结"""
        print("\n" + "="*60)
        print("MAFW 数据集熵值分布分析总结")
        print("="*60)
        
        print(f"📊 分析样本数量: {len(df)}")
        print(f"🔢 固定参数: k={self.k}, m={self.m}")
        
        print(f"\n📈 簇间熵 (α) 统计:")
        alpha_stats = statistics['alpha_statistics']
        print(f"   - 最小值: {alpha_stats['min']:.4f}")
        print(f"   - 25%百分位数: {alpha_stats['25%']:.4f}")
        print(f"   - 50%百分位数: {alpha_stats['50%']:.4f}")
        print(f"   - 75%百分位数: {alpha_stats['75%']:.4f}")
        print(f"   - 90%百分位数: {alpha_stats['90%']:.4f}")
        print(f"   - 最大值: {alpha_stats['max']:.4f}")
        print(f"   - 均值: {alpha_stats['mean']:.4f}")
        print(f"   - 标准差: {alpha_stats['std']:.4f}")
        
        print(f"\n📈 平均簇内熵 (β) 统计:")
        beta_stats = statistics['beta_statistics']
        print(f"   - 最小值: {beta_stats['min']:.4f}")
        print(f"   - 25%百分位数: {beta_stats['25%']:.4f}")
        print(f"   - 50%百分位数: {beta_stats['50%']:.4f}")
        print(f"   - 75%百分位数: {beta_stats['75%']:.4f}")
        print(f"   - 90%百分位数: {beta_stats['90%']:.4f}")
        print(f"   - 最大值: {beta_stats['max']:.4f}")
        print(f"   - 均值: {beta_stats['mean']:.4f}")
        print(f"   - 标准差: {beta_stats['std']:.4f}")
        
        print(f"\n🎯 建议的参数搜索范围:")
        print(f"   - m: {suggested_ranges['m']}")
        print(f"   - α: {suggested_ranges['alpha']}")
        print(f"   - β: {suggested_ranges['beta']}")
        
        total_combinations = len(suggested_ranges['m']) * len(suggested_ranges['alpha']) * len(suggested_ranges['beta'])
        print(f"   - 总参数组合数: {total_combinations}")
        
        print(f"\n💾 结果文件:")
        print(f"   - 熵值分布结果: {self.output_dir}/entropy_distribution_results.csv")
        print(f"   - 统计结果: {self.output_dir}/entropy_statistics.json")
        print(f"   - 建议参数范围: {self.output_dir}/suggested_parameter_ranges.json")
        print(f"   - 可视化图表: {self.output_dir}/entropy_distribution_analysis.png")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="MAFW 数据集熵值分布分析 - 为参数调优做准备")
    
    parser.add_argument("--data_path", type=str, default="/home/peterchen/M2/ADEPT/data/mafw",
                       help="数据路径 (默认: /home/peterchen/M2/ADEPT/data/mafw)")
    parser.add_argument("--excel_path", type=str, default="/home/peterchen/M2/ADEPT/data/mafw/labels/sampled_850.xlsx",
                       help="Excel文件路径")
    parser.add_argument("--video_dir", type=str, default="/home/peterchen/M2/ADEPT/data/mafw/videos",
                       help="视频目录")
    parser.add_argument("--output_dir", type=str, default="/home/peterchen/M2/ADEPT/data/mafw/entropy_analysis_outputs",
                       help="输出目录 (默认: /home/peterchen/M2/ADEPT/data/mafw/entropy_analysis_outputs)")
    parser.add_argument("--env_file", type=str, default="/home/peterchen/M2/ADEPT/.env",
                       help="环境变量文件路径")
    parser.add_argument("--sample_size", type=int, default=400,
                       help="样本数量 (默认: 400)")
    
    args = parser.parse_args()
    
    print(f"🔬 开始 MAFW 数据集熵值分布分析")
    print(f"📊 数据集: MAFW")
    print(f"📁 数据路径: {args.data_path}")
    print(f"📊 样本数量: {args.sample_size}")
    print(f"🔧 将使用真正的 Google Multimodal Embedding API")
    
    try:
        # 创建熵值分布分析器
        analyzer = EntropyDistributionAnalyzer(
            data_path=args.data_path,
            excel_path=args.excel_path,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            env_file=args.env_file,
            sample_size=args.sample_size
        )
        
        # 运行熵值分析流程
        analyzer.run_entropy_analysis()
        
        print("✅ MAFW 数据集熵值分布分析完成！")
        print(f"📊 结果保存在: {args.output_dir}/")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断了分析过程")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 熵值分布分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 