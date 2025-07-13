#!/usr/bin/env python3
"""
分析实际的alpha和beta分布
使用更大的m值和top_k来模拟完整数据集的情况
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import random
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# 自动设置PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.logger import logger, setup_logger
from utils.env_utils import load_env_variables
from utils.data_utils import DatasetConfig, DATASET_CONFIGS
from merlin.questioner import Questioner

class EntropyAnalyzer:
    """分析实际的熵值分布"""
    
    def __init__(self, 
                 data_path: str = "/home/peterchen/M2/ADEPT/data/mafw",
                 excel_path: str = "/home/peterchen/M2/ADEPT/data/mafw/labels/sampled_850.xlsx",
                 env_file: str = "/home/peterchen/M2/ADEPT/.env"):
        
        self.data_path = data_path
        self.excel_path = excel_path
        self.env_file = env_file
        
        # 设置环境
        setup_logger()
        if os.path.exists(self.env_file):
            load_env_variables(self.env_file)
        
        # 加载数据
        self._load_data()
        
        # 测试参数
        self.test_m_values = [8, 12]  # 只测两个m值
        self.test_top_k = 85  # 模拟完整数据集的top_k
        
    def _load_data(self):
        """加载数据"""
        logger.info("加载数据...")
        
        # 加载Excel数据
        self.df = pd.read_excel(self.excel_path)
        logger.info(f"加载了 {len(self.df)} 条数据记录")
        
        # 加载嵌入向量
        embedding_base_path = Path(self.data_path)
        video_emb_dir = embedding_base_path / "video_embeddings"
        text_emb_dir = embedding_base_path / "text_embeddings"
        
        # 加载视频嵌入
        video_emb_files = list(video_emb_dir.glob("*.npy"))
        self.video_embs = []
        self.valid_video_ids = []
        for emb_file in sorted(video_emb_files):
            video_id = emb_file.stem
            emb = np.load(str(emb_file))
            self.video_embs.append(emb)
            self.valid_video_ids.append(video_id)
        
        # 加载文本嵌入
        text_emb_files = list(text_emb_dir.glob("*.npy"))
        self.text_embs = []
        self.valid_text_ids = []
        for emb_file in sorted(text_emb_files):
            text_id = emb_file.stem
            emb = np.load(str(emb_file))
            self.text_embs.append(emb)
            self.valid_text_ids.append(text_id)
        
        logger.info(f"加载了 {len(self.video_embs)} 个视频嵌入")
        logger.info(f"加载了 {len(self.text_embs)} 个文本嵌入")
        
        # 创建查询列表
        self.queries = []
        for idx, row in self.df.iterrows():
            video_name = row['video_name']
            video_id = video_name.replace('.mp4', '')
            
            if video_id in self.valid_video_ids and video_id in self.valid_text_ids:
                self.queries.append({
                    'video': video_name,
                    'text': row['eng_caption'],
                    'video_id': video_id
                })
        
        logger.info(f"创建了 {len(self.queries)} 个有效查询")
    
    def _get_zero_shot_ranking(self, query_text_emb: np.ndarray, video_embs: List[np.ndarray], top_k: int) -> List[int]:
        """零样本检索排名"""
        similarities = []
        for video_emb in video_embs:
            similarity = np.dot(query_text_emb, video_emb) / (np.linalg.norm(query_text_emb) * np.linalg.norm(video_emb))
            similarities.append(similarity)
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return top_indices.tolist()
    
    def analyze_entropy_distribution(self, sample_size: int = 50):
        """分析熵值分布"""
        logger.info(f"开始分析熵值分布，样本数量: {sample_size}")
        
        # 随机选择样本
        random.seed(42)
        selected_indices = random.sample(range(len(self.queries)), min(sample_size, len(self.queries)))
        
        results = []
        
        for m in self.test_m_values:
            logger.info(f"测试 m={m}")
            
            for idx in selected_indices:
                query = self.queries[idx]
                query_text_emb = self.text_embs[self.valid_text_ids.index(query['video_id'])]
                
                # 获取top_k排名
                top_k_indices = self._get_zero_shot_ranking(query_text_emb, self.video_embs, self.test_top_k)
                top_k_embeddings = [self.video_embs[i] for i in top_k_indices]
                
                if len(top_k_embeddings) < m:
                    logger.warning(f"样本数量 {len(top_k_embeddings)} 小于簇数 {m}，跳过")
                    continue
                
                # 计算熵值
                try:
                    questioner = Questioner(
                        n_clusters=m,
                        alpha_threshold=0.01,  # 临时值
                        beta_threshold=0.01    # 临时值
                    )
                    
                    # 计算熵值
                    embeddings_array = np.array(top_k_embeddings)
                    inter_cluster_entropy, intra_cluster_entropy, cluster_info = questioner.compute_entropy_metrics(embeddings_array)
                    
                    results.append({
                        'm': m,
                        'query_idx': idx,
                        'video_id': query['video_id'],
                        'inter_cluster_entropy': inter_cluster_entropy,
                        'intra_cluster_entropy': intra_cluster_entropy,
                        'top_k': len(top_k_embeddings)
                    })
                    
                except Exception as e:
                    logger.error(f"计算熵值失败: {str(e)}")
                    continue
        
        return results
    
    def save_results(self, results: List[Dict], output_dir: str = "entropy_analysis"):
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 保存CSV
        df.to_csv(output_path / "entropy_analysis_results.csv", index=False)
        
        # 计算统计量 - 简化版本，只关注关键百分位数
        stats = {}
        
        # 合并所有m值的数据，计算总体分布
        if not df.empty:
            stats['alpha_statistics'] = {
                'min': df['inter_cluster_entropy'].min(),
                '25%': df['inter_cluster_entropy'].quantile(0.25),
                '50%': df['inter_cluster_entropy'].quantile(0.5),
                '75%': df['inter_cluster_entropy'].quantile(0.75),
                '90%': df['inter_cluster_entropy'].quantile(0.9),
                'max': df['inter_cluster_entropy'].max(),
                'mean': df['inter_cluster_entropy'].mean(),
                'std': df['inter_cluster_entropy'].std()
            }
            
            stats['beta_statistics'] = {
                'min': df['intra_cluster_entropy'].min(),
                '25%': df['intra_cluster_entropy'].quantile(0.25),
                '50%': df['intra_cluster_entropy'].quantile(0.5),
                '75%': df['intra_cluster_entropy'].quantile(0.75),
                '90%': df['intra_cluster_entropy'].quantile(0.9),
                'max': df['intra_cluster_entropy'].max(),
                'mean': df['intra_cluster_entropy'].mean(),
                'std': df['intra_cluster_entropy'].std()
            }
        
        # 保存统计结果
        import json
        with open(output_path / "entropy_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"结果已保存到: {output_path}")
        return stats
    
    def _create_visualizations(self, df: pd.DataFrame, output_path: Path):
        """创建可视化图表"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 簇间熵分布（按m值）
        for m in self.test_m_values:
            m_data = df[df['m'] == m]
            if not m_data.empty:
                axes[0,0].hist(m_data['inter_cluster_entropy'], alpha=0.7, label=f'm={m}', bins=20)
        axes[0,0].set_xlabel('簇间熵 (α)')
        axes[0,0].set_ylabel('频次')
        axes[0,0].set_title('簇间熵分布（按m值）')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 簇内熵分布（按m值）
        for m in self.test_m_values:
            m_data = df[df['m'] == m]
            if not m_data.empty:
                axes[0,1].hist(m_data['intra_cluster_entropy'], alpha=0.7, label=f'm={m}', bins=20)
        axes[0,1].set_xlabel('簇内熵 (β)')
        axes[0,1].set_ylabel('频次')
        axes[0,1].set_title('簇内熵分布（按m值）')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 箱线图 - 簇间熵
        inter_data = [df[df['m'] == m]['inter_cluster_entropy'].values for m in self.test_m_values if not df[df['m'] == m].empty]
        axes[1,0].boxplot(inter_data, labels=[f'm={m}' for m in self.test_m_values if not df[df['m'] == m].empty])
        axes[1,0].set_xlabel('m值')
        axes[1,0].set_ylabel('簇间熵 (α)')
        axes[1,0].set_title('簇间熵箱线图')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 箱线图 - 簇内熵
        intra_data = [df[df['m'] == m]['intra_cluster_entropy'].values for m in self.test_m_values if not df[df['m'] == m].empty]
        axes[1,1].boxplot(intra_data, labels=[f'm={m}' for m in self.test_m_values if not df[df['m'] == m].empty])
        axes[1,1].set_xlabel('m值')
        axes[1,1].set_ylabel('簇内熵 (β)')
        axes[1,1].set_title('簇内熵箱线图')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "entropy_analysis_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self, stats: Dict):
        """打印分析总结"""
        print("\n" + "="*60)
        print("熵值分布分析总结")
        print("="*60)
        print(f"测试参数: top_k={self.test_top_k}, m值={self.test_m_values}")
        
        if 'alpha_statistics' in stats:
            alpha_stats = stats['alpha_statistics']
            beta_stats = stats['beta_statistics']
            
            print(f"\n📊 Alpha (簇间熵) 统计结果:")
            print(f"  最小值: {alpha_stats['min']:.6f}")
            print(f"  25%分位数: {alpha_stats['25%']:.6f}")
            print(f"  50%分位数: {alpha_stats['50%']:.6f}")
            print(f"  75%分位数: {alpha_stats['75%']:.6f}")
            print(f"  90%分位数: {alpha_stats['90%']:.6f}")
            print(f"  最大值: {alpha_stats['max']:.6f}")
            print(f"  均值: {alpha_stats['mean']:.6f}")
            print(f"  标准差: {alpha_stats['std']:.6f}")
            
            print(f"\n📊 Beta (簇内熵) 统计结果:")
            print(f"  最小值: {beta_stats['min']:.6f}")
            print(f"  25%分位数: {beta_stats['25%']:.6f}")
            print(f"  50%分位数: {beta_stats['50%']:.6f}")
            print(f"  75%分位数: {beta_stats['75%']:.6f}")
            print(f"  90%分位数: {beta_stats['90%']:.6f}")
            print(f"  最大值: {beta_stats['max']:.6f}")
            print(f"  均值: {beta_stats['mean']:.6f}")
            print(f"  标准差: {beta_stats['std']:.6f}")
            
            print(f"\n💡 推荐参数范围:")
            print(f"  Alpha: [{alpha_stats['25%']:.6f}, {alpha_stats['50%']:.6f}, {alpha_stats['75%']:.6f}]")
            print(f"  Beta: [{beta_stats['25%']:.6f}, {beta_stats['50%']:.6f}, {beta_stats['75%']:.6f}]")
        
        print("\n" + "="*60)


def main():
    """主函数"""
    analyzer = EntropyAnalyzer()
    
    # 分析熵值分布
    results = analyzer.analyze_entropy_distribution(sample_size=10)
    
    # 保存结果
    stats = analyzer.save_results(results)
    
    # 打印总结
    analyzer.print_summary(stats)


if __name__ == "__main__":
    main() 