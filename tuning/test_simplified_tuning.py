#!/usr/bin/env python3
"""
测试简化后的参数调优系统
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from parameter_tuning import ParameterTuner

def test_parameter_tuning():
    """测试参数调优系统"""
    print("🧪 测试简化后的参数调优系统")
    
    try:
        # 创建参数调优器
        tuner = ParameterTuner(
            dataset="mer2024",
            data_path="data",
            excel_path="/home/peterchen/M2/MER2024/llava_next_video_caption.xlsx",
            video_base_dir="/home/peterchen/M2/MER2024/video-selected",
            output_dir="outputs",
            env_file="/home/peterchen/M2/ADEPT/.env"
        )
        
        print("✅ 参数调优器初始化成功")
        print(f"📊 样本数量: {len(tuner.queries)}")
        print(f"🔢 参数组合数: {len(tuner.param_space['m']) * len(tuner.param_space['alpha']) * len(tuner.param_space['beta'])}")
        print(f"📋 参数空间: m={tuner.param_space['m']}, α={tuner.param_space['alpha']}, β={tuner.param_space['beta']}")
        
        # 测试单个参数组合评估
        test_params = {'m': 4, 'alpha': 0.006, 'beta': 0.006}
        print(f"\n🧪 测试参数组合: {test_params}")
        
        # 使用少量样本进行快速测试
        result = tuner._evaluate_parameter_combination(test_params, sample_size=2)
        
        print(f"✅ 测试完成")
        print(f"📈 Recall@10: {result['recall_at_10']:.4f}")
        print(f"🎯 Top1准确率: {result['top1_accuracy']:.4f}")
        print(f"📊 评估样本数: {result['num_samples']}")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parameter_tuning() 