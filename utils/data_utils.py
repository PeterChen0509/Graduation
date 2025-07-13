import json
import os
import pandas as pd
import numpy as np
from glob import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass

from typing import Optional

# Add the project root to Python path when running as script
if __name__ == "__main__":
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.append(project_root)

from utils.logger import logger

class DatasetConfig(NamedTuple):
    """Configuration for dataset-specific file paths and settings."""
    name: str
    test_file: str
    video_ext: str
    video_subdirs: List[str]
    additional_files: Dict[str, str]
    sample_limit: Optional[int] = None

# Dataset-specific configurations
DATASET_CONFIGS = {
    "mafw": DatasetConfig(
        name="mafw",
        test_file="sampled_850.xlsx",
        video_ext=".mp4",
        video_subdirs=["/home/peterchen/M2/MAFW/data/clips/unzip"],  # 使用绝对路径
        additional_files={},
        sample_limit=None
    ),
    "mer2024": DatasetConfig(
        name="mer2024",
        test_file="llava_next_video_caption.xlsx",
        video_ext=".mp4",
        video_subdirs=["/home/peterchen/M2/MER2024/video-selected"],  # 使用绝对路径
        additional_files={},
        sample_limit=None
    ),
}

@dataclass
class DatasetPaths:
    """Holds paths for dataset files and directories."""
    base_path: Path
    video_paths: List[Path]  # List of video paths
    gpt4v_caption_path: Path
    video_embeddings_path: Path
    text_embeddings_path: Path
    config: DatasetConfig
    
    @classmethod
    def from_base_path(cls, base_path: Union[str, Path], config: DatasetConfig) -> 'DatasetPaths':
        """Create DatasetPaths from a base directory path."""
        base = Path(base_path)
        # Create a list of video paths from the video_subdirs list
        # Handle both absolute and relative paths
        video_paths = []
        for subdir in config.video_subdirs:
            if os.path.isabs(subdir):
                # If subdir is absolute path, use it directly
                video_paths.append(Path(subdir))
            else:
                # If subdir is relative path, join with base
                video_paths.append(base / subdir)
        
        return cls(
            base_path=base,
            video_paths=video_paths,
            gpt4v_caption_path=base / "gpt4v_captions",
            video_embeddings_path=base / "video_embeddings",
            text_embeddings_path=base / "text_embeddings",
            config=config
        )

    def get_video_paths(self) -> List[Path]:
        """Get all video paths."""
        return self.video_paths
        
    def find_video_path(self, video_id: str) -> Optional[Path]:
        """
        Find the path to a video file by checking all video subdirectories.
        
        Args:
            video_id: The ID of the video
            
        Returns:
            Path to the video file if found, None otherwise
        """
        for video_path in self.video_paths:
            full_path = video_path / f"{video_id}{self.config.video_ext}"
            if full_path.exists():
                return full_path
        return None

def load_excel_data(excel_path: str, video_base_dir: str = "/home/peterchen/M2/MAFW/data/clips/unzip", dataset: str = "mafw") -> Tuple[List[Dict], Dict[str, str]]:
    """
    从 Excel 文件加载视频-字幕对数据
    
    Args:
        excel_path: Excel 文件路径，包含视频名称和描述列
        video_base_dir: 视频文件的基础目录路径
        dataset: 数据集名称，用于确定列名映射
        
    Returns:
        Tuple of (queries, video_captions_dict)
        - queries: List of query dictionaries with 'video' and 'caption' keys
        - video_captions_dict: Dictionary mapping video names to captions
    """
    logger.info(f"正在加载 Excel 数据: {excel_path}")
    
    try:
        # 读取 Excel 文件
        df = pd.read_excel(excel_path)
        logger.info(f"成功读取 Excel 文件，共 {len(df)} 条记录")
        
        # 根据数据集确定列名映射
        if dataset == "mafw":
            video_col = 'video_name'
            caption_col = 'eng_caption'
        elif dataset == "mer2024":
            video_col = 'name'
            caption_col = 'eng_caption'
        else:
            raise ValueError(f"不支持的数据集: {dataset}")
        
        # 检查必要的列是否存在
        required_columns = [video_col, caption_col]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Excel 文件缺少必要的列: {missing_columns}")
        
        # 构建查询列表和视频字幕字典
        queries = []
        video_captions_dict = {}
        video_base_path = Path(video_base_dir)
        
        for idx, row in df.iterrows():
            video_name = row[video_col]
            caption = row[caption_col]
            
            # 构建完整的视频文件路径（添加.mp4后缀）
            video_path = str(video_base_path / f"{video_name}.mp4")
            
            # 检查视频文件是否存在
            if not Path(video_path).exists():
                logger.warning(f"视频文件不存在: {video_path}")
                continue
            
            # 创建查询字典
            query = {
                "video": video_name,
                "caption": caption
            }
            queries.append(query)
            
            # 添加到视频字幕字典
            video_captions_dict[video_name] = caption
        
        logger.info(f"成功加载 {len(queries)} 个有效的视频-字幕对")
        return queries, video_captions_dict
        
    except Exception as e:
        logger.error(f"加载 Excel 数据时出错: {str(e)}")
        raise

def load_embeddings(video_id: str, paths: DatasetPaths) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load video and text embeddings for a given video ID.
    
    Args:
        video_id: ID of the video
        paths: DatasetPaths object containing relevant paths
        
    Returns:
        Tuple of (video_embedding, text_embedding)
    """
    # Load video embedding
    video_emb_path = paths.video_embeddings_path / f"{video_id}.npy"
    video_emb = np.load(str(video_emb_path))
    
    # Load text embedding
    text_emb_path = next(paths.text_embeddings_path.glob(f"{video_id}*"))
    text_emb = np.load(str(text_emb_path))
    
    return video_emb, text_emb

def prepare_mafw_data(excel_path: str, video_base_dir: str = "/home/peterchen/M2/MAFW/data/clips/unzip") -> Tuple[List[Dict], Dict[str, str], List[np.ndarray], List[np.ndarray]]:
    """Handle MAFW dataset loading from Excel file."""
    # Load data from Excel
    queries, video_captions = load_excel_data(excel_path, video_base_dir, dataset="mafw")
    
    # 尝试加载嵌入向量
    video_embs = []
    text_embs = []
    
    # 嵌入向量保存路径
    embedding_base_path = Path("/home/peterchen/M2/ADEPT/data/mafw")
    video_emb_dir = embedding_base_path / "video_embeddings"
    text_emb_dir = embedding_base_path / "text_embeddings"
    
    # 检查嵌入向量目录是否存在
    if video_emb_dir.exists() and text_emb_dir.exists():
        logger.info("正在加载预生成的嵌入向量...")
        
        # 首先确定嵌入向量的维度
        embedding_dim = None
        for query in queries:
            video_id = query["video"].replace('.mp4', '')
            video_emb_path = video_emb_dir / f"{video_id}.npy"
            if video_emb_path.exists():
                sample_emb = np.load(str(video_emb_path))
                embedding_dim = sample_emb.shape[0]
                logger.info(f"检测到嵌入向量维度: {embedding_dim}")
                break
        
        if embedding_dim is None:
            logger.warning("未找到任何嵌入向量文件，使用默认维度 1408")
            embedding_dim = 1408
        
        # 过滤掉缺失嵌入向量的查询
        valid_queries = []
        for query in queries:
            video_id = query["video"].replace('.mp4', '')
            
            # 检查视频和文本嵌入向量是否都存在
            video_emb_path = video_emb_dir / f"{video_id}.npy"
            text_emb_path = text_emb_dir / f"{video_id}.npy"
            
            if video_emb_path.exists() and text_emb_path.exists():
                valid_queries.append(query)
            else:
                logger.warning(f"跳过视频 {video_id}：嵌入向量文件不完整")
        
        logger.info(f"有效查询数量: {len(valid_queries)}/{len(queries)}")
        
        for query in valid_queries:
            video_id = query["video"].replace('.mp4', '')
            
            # 加载视频嵌入向量
            video_emb_path = video_emb_dir / f"{video_id}.npy"
            video_emb = np.load(str(video_emb_path))
            video_embs.append(video_emb)
            
            # 加载文本嵌入向量
            text_emb_path = text_emb_dir / f"{video_id}.npy"
            text_emb = np.load(str(text_emb_path))
            text_embs.append(text_emb)
        
        logger.info(f"成功加载 {len(video_embs)} 个视频嵌入向量和 {len(text_embs)} 个文本嵌入向量")
    else:
        logger.warning("嵌入向量目录不存在，返回空列表。请先运行 generate_embeddings.py 生成嵌入向量。")
    
    return valid_queries, video_captions, video_embs, text_embs

def prepare_mer2024_data(excel_path: str, video_base_dir: str = "/home/peterchen/M2/MER2024/video-selected") -> Tuple[List[Dict], Dict[str, str], List[np.ndarray], List[np.ndarray]]:
    """Handle MER2024 dataset loading from Excel file."""
    # Load data from Excel
    queries, video_captions = load_excel_data(excel_path, video_base_dir, dataset="mer2024")
    
    # 尝试加载嵌入向量
    video_embs = []
    text_embs = []
    
    # 嵌入向量保存路径
    embedding_base_path = Path("/home/peterchen/M2/ADEPT/data/mer2024")
    video_emb_dir = embedding_base_path / "video_embeddings"
    text_emb_dir = embedding_base_path / "text_embeddings"
    
    logger.info(f"视频嵌入向量目录: {video_emb_dir}")
    logger.info(f"文本嵌入向量目录: {text_emb_dir}")
    
    # 检查嵌入向量目录是否存在
    if video_emb_dir.exists() and text_emb_dir.exists():
        logger.info("正在加载预生成的嵌入向量...")
        
        # 首先确定嵌入向量的维度
        embedding_dim = None
        for query in queries:
            video_id = query["video"]  # MER2024 不需要去除扩展名
            video_emb_path = video_emb_dir / f"{video_id}.npy"
            if video_emb_path.exists():
                sample_emb = np.load(str(video_emb_path))
                embedding_dim = sample_emb.shape[0]
                logger.info(f"检测到嵌入向量维度: {embedding_dim}")
                break
        
        if embedding_dim is None:
            logger.warning("未找到任何嵌入向量文件，使用默认维度 1408")
            embedding_dim = 1408
        
        # 过滤掉缺失嵌入向量的查询
        valid_queries = []
        for query in queries:
            video_id = query["video"]  # MER2024 不需要去除扩展名
            
            # 检查视频和文本嵌入向量是否都存在
            video_emb_path = video_emb_dir / f"{video_id}.npy"
            text_emb_path = text_emb_dir / f"{video_id}.npy"
            
            if video_emb_path.exists() and text_emb_path.exists():
                valid_queries.append(query)
            else:
                logger.warning(f"跳过视频 {video_id}：嵌入向量文件不完整")
        
        logger.info(f"有效查询数量: {len(valid_queries)}/{len(queries)}")
        
        for query in valid_queries:
            video_id = query["video"]  # MER2024 不需要去除扩展名
            
            # 加载视频嵌入向量
            video_emb_path = video_emb_dir / f"{video_id}.npy"
            video_emb = np.load(str(video_emb_path))
            video_embs.append(video_emb)
            
            # 加载文本嵌入向量
            text_emb_path = text_emb_dir / f"{video_id}.npy"
            text_emb = np.load(str(text_emb_path))
            text_embs.append(text_emb)
        
        logger.info(f"成功加载 {len(video_embs)} 个视频嵌入向量和 {len(text_embs)} 个文本嵌入向量")
    else:
        logger.warning("嵌入向量目录不存在，返回空列表。请先运行 generate_embeddings.py 生成嵌入向量。")
    
    return valid_queries, video_captions, video_embs, text_embs

def prepare_data(
    dataset: str,
    video_path: Union[str, Path],
    caption: Any = None,
    excel_path: str = None,
    video_base_dir: str = "/home/peterchen/M2/MAFW/data/clips/unzip"
) -> Tuple[List[Dict], Dict, List[np.ndarray], List[np.ndarray]]:
    """
    Prepare data for a specified dataset.
    
    Args:
        dataset: Name of the dataset ('msvd', 'msrvtt', 'anet', or 'mafw')
        video_path: Path to the dataset directory
        caption: Optional caption data (currently unused)
        excel_path: Path to Excel file (required for 'mafw' dataset)
        video_base_dir: Base directory for video files (for 'mafw' dataset)
        
    Returns:
        Tuple containing:
        - queries: List of query dictionaries
        - video_captions: Dictionary of video captions
        - video_embs: List of video embeddings
        - text_embs: List of text embeddings
        
    Raises:
        ValueError: If dataset is not supported
    """
    if dataset == "mafw":
        if excel_path is None:
            excel_path = "/home/peterchen/M2/ADEPT/data/mafw/labels/sampled_850.xlsx"
        return prepare_mafw_data(excel_path, video_base_dir)
    
    if dataset == "mer2024":
        if excel_path is None:
            excel_path = "/home/peterchen/M2/MER2024/llava_next_video_caption.xlsx"
        return prepare_mer2024_data(excel_path, video_base_dir)
    
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset} is not supported")
    
    # config = DATASET_CONFIGS[dataset]
    # paths = DatasetPaths.from_base_path(video_path, config)
    
    # dataset_handlers = {
    #     "mafw": prepare_mafw_data,
    # }
    
    # return dataset_handlers[dataset](paths, config)