#!/usr/bin/env python3
"""
运行真正的分布式参数调优
支持多GPU并行处理视频对话
"""

import os
import sys
import argparse

# 自动设置PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from distributed_tuning import TrueDistributedParameterTuner

def main():
    parser = argparse.ArgumentParser(description="运行真正的多GPU并行MERLIN参数调优")
    
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
    print(f"💡 每个GPU将并行处理一个视频对话，大幅加速计算")
    
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