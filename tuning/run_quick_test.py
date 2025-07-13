#!/usr/bin/env python3
"""
快速测试脚本
用小样本快速验证完整的并行参数调优系统
"""

import os
import sys
import argparse

# 自动设置PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from complete_parallel_tuning import CompleteParallelTuner

def main():
    parser = argparse.ArgumentParser(description="快速测试完整的多GPU并行MERLIN参数调优")
    
    # GPU参数
    parser.add_argument("--n_gpus", type=int, default=4,
                       help="使用的GPU数量 (默认: 4)")
    parser.add_argument("--sample_size", type=int, default=10,
                       help="随机抽取的样本数量 (默认: 10，快速测试)")
    
    args = parser.parse_args()
    
    print(f"🚀 快速测试完整的多GPU并行 MERLIN 参数调优")
    print(f"🔢 GPU数量: {args.n_gpus}")
    print(f"📊 样本数量: {args.sample_size}")
    print(f"💡 每个视频将进行真正的多轮交互：提问→计算熵→决策ask/refine→综合回答→重排序")
    print(f"📈 将自动生成可视化图表")
    print(f"⏱️  预计时间: 约10-15分钟")
    
    try:
        # 创建完整并行调优器
        complete_tuner = CompleteParallelTuner(
            n_gpus=args.n_gpus,
            sample_size=args.sample_size
        )
        
        # 运行完整并行调优流程
        complete_tuner.run_complete_tuning()
        
        print("✅ 快速测试完成！")
        print(f"📊 结果保存在: /home/peterchen/M2/ADEPT/data/mafw/best_parameter/")
        print(f"📈 可视化图表已生成")
        print(f"🎯 如果结果满意，可以运行完整版本：")
        print(f"   python3 run_complete_tuning.py --n_gpus 4 --sample_size 50")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断了测试过程")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 