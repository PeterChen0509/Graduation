#!/usr/bin/env python3
"""
测试环境变量加载
"""

import os
import sys

# 自动设置PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.env_utils import load_env_variables

def test_env_loading():
    """测试环境变量加载"""
    env_file = "/home/peterchen/M2/ADEPT/.env"
    
    print("=== 环境变量加载测试 ===")
    print(f"环境变量文件路径: {env_file}")
    print(f"文件是否存在: {os.path.exists(env_file)}")
    
    if os.path.exists(env_file):
        print("\n文件内容:")
        with open(env_file, 'r') as f:
            print(f.read())
    
    print("\n=== 加载环境变量 ===")
    load_env_variables(env_file)
    
    print("\n=== 检查关键环境变量 ===")
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT_ID')
    location = os.getenv('GOOGLE_CLOUD_LOCATION')
    
    print(f"GOOGLE_CLOUD_PROJECT_ID: {project_id}")
    print(f"GOOGLE_CLOUD_LOCATION: {location}")
    
    if project_id:
        print("✅ Google Cloud配置正确")
    else:
        print("❌ Google Cloud配置缺失")

if __name__ == "__main__":
    test_env_loading() 