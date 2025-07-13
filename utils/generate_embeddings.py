import os
import pandas as pd
from pathlib import Path
import numpy as np
from utils.logger import logger, setup_logger
from utils.env_utils import load_env_variables, get_required_env
from merlin.reranker import Reranker

def process_videos_from_excel(reranker, video_dir, video_emb_dir, df):
    """从 Excel 文件中读取视频名称，在指定目录找到视频文件并生成特征向量"""
    logger.info("开始处理视频特征向量...")
    
    # 从 Excel 文件中获取视频名称列表
    video_names = df['video_name'].tolist()
    total_videos = len(video_names)
    logger.info(f"需要处理 {total_videos} 个视频文件")
    
    processed_count = 0
    for i, video_name in enumerate(video_names, 1):
        video_path = video_dir / video_name
        video_id = video_name.replace('.mp4', '')
        
        # 检查视频文件是否存在
        if not video_path.exists():
            logger.warning(f"视频文件不存在: {video_path}")
            continue
            
        logger.info(f"处理视频 [{i}/{total_videos}]: {video_id}")
        
        try:
            # 生成视频特征向量
            video_emb = reranker.get_image_video_text_embeddings(video_path=str(video_path))
            # 保存视频特征向量
            np.save(video_emb_dir / f"{video_id}.npy", video_emb.video_embeddings[0].embedding)
            logger.info(f"成功生成视频特征向量: {video_id}")
            processed_count += 1
        except Exception as e:
            logger.error(f"生成视频特征向量失败 {video_id}: {str(e)}")
            continue
    
    logger.info(f"视频特征向量处理完成，成功处理 {processed_count}/{total_videos} 个视频")

def process_captions(reranker, df, text_emb_dir):
    """处理有文本描述的视频，生成文本特征向量"""
    logger.info("开始处理文本特征向量...")
    total_captions = len(df)
    logger.info(f"找到 {total_captions} 个文本描述")
    
    processed_count = 0
    for i, (_, row) in enumerate(df.iterrows(), 1):
        video_name = row['video_name']
        video_id = video_name.replace('.mp4', '')
        caption = row['eng_caption']
        
        logger.info(f"处理文本描述 [{i}/{total_captions}]: {video_id}")
        
        try:
            # 生成文本特征向量
            text_emb = reranker.get_image_video_text_embeddings(contextual_text=caption)
            # 保存文本特征向量
            np.save(text_emb_dir / f"{video_id}.npy", text_emb.text_embedding)
            logger.info(f"成功生成文本特征向量: {video_id}")
            processed_count += 1
        except Exception as e:
            logger.error(f"生成文本特征向量失败 {video_id}: {str(e)}")
            continue
    
    logger.info(f"文本特征向量处理完成，成功处理 {processed_count}/{total_captions} 个文本")

def main():
    # 设置环境变量
    load_env_variables()
    
    # 设置日志
    setup_logger(level="INFO")
    
    # 设置路径
    excel_path = Path("/home/peterchen/M2/ADEPT/data/mafw/labels/sampled_850.xlsx")
    video_dir = Path("/home/peterchen/M2/MAFW/data/clips/unzip")
    
    # 创建特征向量目录 - 保存在 ADEPT 项目的 MAFW 数据目录下
    data_path = Path("/home/peterchen/M2/ADEPT/data/mafw")
    video_emb_dir = data_path / "video_embeddings"
    text_emb_dir = data_path / "text_embeddings"
    video_emb_dir.mkdir(exist_ok=True)
    text_emb_dir.mkdir(exist_ok=True)
    
    logger.info(f"Excel 文件路径: {excel_path}")
    logger.info(f"视频目录路径: {video_dir}")
    logger.info(f"视频嵌入向量保存路径: {video_emb_dir}")
    logger.info(f"文本嵌入向量保存路径: {text_emb_dir}")
    
    # 读取 xlsx 文件
    df = pd.read_excel(excel_path)
    logger.info(f"成功读取 Excel 文件，共 {len(df)} 条记录")
    
    # 初始化 Reranker
    reranker = Reranker(
        project_id=os.environ["GOOGLE_CLOUD_PROJECT_ID"],
        location=os.environ["GOOGLE_CLOUD_LOCATION"],
        memory_path=str(data_path),
        queries=[],  # 空列表，因为我们只是用来生成特征向量
        video_ext=".mp4"
    )
    
    # 处理视频特征向量
    process_videos_from_excel(reranker, video_dir, video_emb_dir, df)
    
    # 处理文本特征向量
    process_captions(reranker, df, text_emb_dir)
    
    logger.info("所有特征向量生成完成！")

if __name__ == "__main__":
    main() 