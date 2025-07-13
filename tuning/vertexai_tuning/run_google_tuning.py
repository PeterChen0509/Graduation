#!/usr/bin/env python3
"""
运行熵值分布分析 - 为参数调优做准备
"""

import os
import sys
import argparse

# 自动设置PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from run_google_embedding_tuning import EntropyDistributionAnalyzer

def main():
    parser = argparse.ArgumentParser(description="运行熵值分布分析")
    
    parser.add_argument("--output_dir", type=str, default="entropy_analysis_outputs",
                       help="输出目录 (默认: entropy_analysis_outputs)")
    
    args = parser.parse_args()
    
    print(f"🔬 开始熵值分布分析（快速测试模式）")
    print(f"📊 数据集: MER2024")
    print(f"📊 样本数量: 10")
    print(f"📁 输出目录: {args.output_dir}")
    print(f"🔧 将使用真正的 Google Multimodal Embedding API")
    print(f"📈 将生成详细的熵值分布分析")
    print(f"⚡ 快速测试：仅处理10个样本")
    
    try:
        # 创建熵值分布分析器
        analyzer = EntropyDistributionAnalyzer(
            data_path="/home/peterchen/M2/MER2024",
            excel_path="/home/peterchen/M2/MER2024/llava_next_video_caption.xlsx",
            video_dir="/home/peterchen/M2/MER2024/video-selected",
            output_dir=args.output_dir,
            env_file="/home/peterchen/M2/ADEPT/.env"
        )
        
        # 运行熵值分析流程
        analyzer.run_entropy_analysis()
        
        print("✅ 熵值分布分析完成！")
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