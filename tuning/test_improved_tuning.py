#!/usr/bin/env python3
"""
测试改进后的重排序逻辑
验证多轮交互是否能够产生更合理的排名改进
"""

import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 自动设置PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from complete_parallel_tuning import CompleteParallelTuner

def test_improved_reranking():
    """测试改进后的重排序逻辑"""
    print("🧪 测试改进后的重排序逻辑...")
    
    # 创建调优器（使用小样本）
    tuner = CompleteParallelTuner(n_gpus=1, sample_size=10)
    
    # 加载数据
    tuner.load_data()
    
    # 测试单个参数组合
    test_params = {'m': 8, 'alpha': 0.5, 'beta': 0.4}
    
    print(f"📊 测试参数: {test_params}")
    print(f"📁 样本数量: {len(tuner.df)}")
    
    # 评估单个视频
    result = tuner.evaluate_single_video_complete(0, test_params, 0)
    
    print(f"\n📈 测试结果:")
    print(f"   目标视频: {result['target_vid']}")
    print(f"   初始排名: {result['initial_rank']}")
    print(f"   最终排名: {result['final_rank']}")
    print(f"   改进幅度: {result['improvement']}")
    print(f"   交互轮数: {result['final_round']}")
    
    # 分析对话历史
    if 'conversation_history' in result:
        print(f"\n💬 对话历史:")
        for i, conv in enumerate(result['conversation_history']):
            print(f"   轮次 {i+1}:")
            print(f"     问题: {conv.get('question', 'N/A')}")
            print(f"     答案: {conv.get('answer', 'N/A')[:50]}...")
            print(f"     排名: {conv.get('target_rank', 'N/A')}")
    
    # 统计改进情况
    improvements = []
    for i in range(min(5, len(tuner.df))):
        try:
            result = tuner.evaluate_single_video_complete(i, test_params, 0)
            improvements.append(result['improvement'])
            print(f"视频 {i}: 初始={result['initial_rank']}, 最终={result['final_rank']}, 改进={result['improvement']}")
        except Exception as e:
            print(f"视频 {i} 评估失败: {e}")
    
    if improvements:
        avg_improvement = np.mean(improvements)
        positive_improvements = sum(1 for imp in improvements if imp > 0)
        print(f"\n📊 改进统计:")
        print(f"   平均改进: {avg_improvement:.2f}")
        print(f"   正改进比例: {positive_improvements}/{len(improvements)} ({positive_improvements/len(improvements)*100:.1f}%)")
    
    print("\n✅ 测试完成！")

if __name__ == "__main__":
    test_improved_reranking() 