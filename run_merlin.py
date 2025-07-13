import os
import json
import time
import asyncio
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import argparse
from sklearn.metrics.pairwise import cosine_similarity

# Fix for PyTorch 2.7.0 compatibility with older libraries
import torch
if not hasattr(torch, "compiler"):
    import types
    torch.compiler = types.SimpleNamespace()
    torch.compiler.is_compiling = lambda: False
elif not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False

# Additional fixes for PyTorch 2.7.0 compatibility
if hasattr(torch, 'compiler'):
    # Add any missing attributes that might be accessed
    if not hasattr(torch.compiler, 'compile'):
        torch.compiler.compile = lambda *args, **kwargs: args[0] if args else lambda x: x
    if not hasattr(torch.compiler, 'is_compiling'):
        torch.compiler.is_compiling = lambda: False

# Global monkey patch for any torch.compiler issues
import sys
import types

def create_torch_compiler_patch():
    """Create a comprehensive torch.compiler patch"""
    compiler_module = types.SimpleNamespace()
    compiler_module.is_compiling = lambda: False
    compiler_module.compile = lambda *args, **kwargs: args[0] if args else lambda x: x
    return compiler_module

# Apply the patch globally
if not hasattr(torch, 'compiler') or not hasattr(torch.compiler, 'is_compiling'):
    torch.compiler = create_torch_compiler_patch()

from utils.data_utils import prepare_data, DATASET_CONFIGS, DatasetPaths
from utils.video_utils import video_frame_generator
from utils.logger import logger, setup_logger
from utils.setup_directories import verify_structure, create_directory_structure
from utils.env_utils import load_env_variables, get_required_env, get_optional_env

from typing import Optional
from human_agent.answerer import Answerer
from merlin.reranker import Reranker
from merlin.questioner import Questioner

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MERLIN: Multimodal Embedding Refinement via LLM-based Iterative Navigation"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mafw", "mer2024"],
        required=True,
        help="Dataset to process"
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        default="data",
        help="Path to dataset directory"
    )
    
    # Excel file arguments (for MAFW/MER2024 dataset)
    parser.add_argument(
        "--excel_path",
        type=str,
        default="/home/peterchen/M2/MER2024/llava_next_video_caption.xlsx",
        help="Path to Excel file (for MAFW/MER2024 dataset)"
    )
    
    parser.add_argument(
        "--video_base_dir",
        type=str,
        default="/home/peterchen/M2/MER2024/video-selected",
        help="Base directory for video files (for MAFW/MER2024 dataset)"
    )
    
    # Model configuration (can be overridden from .env)
    parser.add_argument(
        "--model_name",
        type=str,
        help="OpenAI model to use (default from .env)"
    )
    
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="Maximum tokens for model response (default from .env)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default from .env)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    # Environment configuration
    parser.add_argument(
        "--env_file",
        type=str,
        default="/home/peterchen/M2/ADEPT/.env",
        help="Path to .env file"
    )
    
    # Number of rounds for question-answering
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=5,
        help="Number of rounds for question-answering iteration"
    )
    
    return parser.parse_args()

def setup_environment(args: argparse.Namespace) -> None:
    """
    Set up the environment for running MERLIN.
    
    Args:
        args: Parsed command line arguments
    """
    # Load environment variables
    load_env_variables(args.env_file)
    
    # Set up logging
    log_level = args.log_level or get_optional_env("LOG_LEVEL", "INFO")
    setup_logger(level=log_level)
    
    # Set debug mode from environment if not set in args
    if not args.debug and get_optional_env("DEBUG", "false").lower() == "true":
        args.debug = True
    
    # Verify/create directory structure
    if not verify_structure():
        logger.info("Creating directory structure...")
        create_directory_structure(debug=args.debug)
    
    # Set up model configuration from environment if not provided in args
    if not args.model_name:
        args.model_name = get_optional_env("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
    if not args.max_tokens:
        args.max_tokens = int(get_optional_env("MAX_TOKENS", "300"))
    
    # Set up Google Cloud configuration
    os.environ["GOOGLE_CLOUD_PROJECT_ID"] = get_required_env("GOOGLE_CLOUD_PROJECT_ID")
    os.environ["GOOGLE_CLOUD_LOCATION"] = get_required_env("GOOGLE_CLOUD_LOCATION")

def main() -> None:
    """Main entry point for MERLIN."""
    args = parse_args()
    
    try:
        # Set up environment
        setup_environment(args)
        
        # Load dataset
        logger.info(f"Processing dataset: {args.dataset}")
        
        # Get dataset-specific configuration early
        dataset_config = DATASET_CONFIGS[args.dataset]
        
        # Create dataset paths
        dataset_paths = DatasetPaths.from_base_path(args.data_path, dataset_config)
        
        # Get video paths from dataset_paths
        video_base_paths = [str(path) for path in dataset_paths.get_video_paths()]
        video_ext = dataset_config.video_ext
        
        # Load dataset
        if args.dataset in ["mafw", "mer2024"]:
            queries, video_captions, video_embs, text_embs = prepare_data(
                dataset=args.dataset,
                video_path=args.data_path,
                caption=os.path.join(args.data_path, "gpt4o_caption"),
                excel_path=args.excel_path,
                video_base_dir=args.video_base_dir
            )
        else:
            queries, video_captions, video_embs, text_embs = prepare_data(
                dataset=args.dataset,
                video_path=args.data_path,
                caption=os.path.join(args.data_path, "gpt4o_caption")
            )
        
        # Set up other paths
        video_caption_path = f"{args.data_path}/gpt4o_caption"
        embedding_base_path = f"{args.data_path}/"
        base_chat_log_dir = f"{args.output_dir}/chatlog_rerank_{args.dataset}"
        
        # 显示输出目录的实际路径
        logger.info(f"当前工作目录: {os.getcwd()}")
        logger.info(f"输出目录: {os.path.abspath(args.output_dir)}")
        logger.info(f"聊天日志目录: {os.path.abspath(base_chat_log_dir)}")

        # Retrieve top-k video candidates based on cosine similarity
        # Set top_k to 10% of dataset size, rounded to nearest integer
        total_videos = len(video_embs)
        top_k = max(1, round(total_videos * 0.1))  # At least 1, rounded to nearest integer
        logger.info(f"Zero-shot retrieval evaluation...")
        logger.info(f"Dataset size: {total_videos}, Top-k set to: {top_k} (10% of dataset)")
        predictions = []

        for i, query_text_emb in tqdm(enumerate(text_embs)):
            similarities = cosine_similarity([query_text_emb], video_embs)
            top_k_indices = np.argsort(-similarities[0])[:top_k]
            
            # Create a dictionary for the prediction
            prediction = {
                "query_id": queries[i]["video"].replace(dataset_config.video_ext, ""),
                "org_ranking": [queries[j]["video"].replace(dataset_config.video_ext, "") for j in top_k_indices.squeeze()]
            }
            predictions.append(prediction)

        # Calculate top-1, top-5, and top-10 retrieval accuracies
        top_1_acc = 0
        top_5_acc = 0
        top_10_acc = 0
        for pred in predictions:
            target_vid = pred["query_id"]
            ranking = pred["org_ranking"]
            if target_vid == ranking[0]:
                top_1_acc += 1
            if target_vid in ranking[:5]:
                top_5_acc += 1
            if target_vid in ranking[:10]:
                top_10_acc += 1

        total_queries = len(predictions)
        logger.info(f"Top-1 retrieval accuracy: {top_1_acc / total_queries * 100:.2f}% {top_1_acc}/{total_queries}")
        logger.info(f"Top-5 retrieval accuracy: {top_5_acc / total_queries * 100:.2f}% {top_5_acc}/{total_queries}")
        logger.info(f"Top-10 retrieval accuracy: {top_10_acc / total_queries * 100:.2f}% {top_10_acc}/{total_queries}")

        total = 0
        cannot_check = []
        rank_sum = [0, 0, 0, 0, 0, 0]
        top1_acc = 0
        top5_acc = 0
        top10_acc = 0 

        zs_top1_acc = 0
        zs_top5_acc = 0
        zs_top10_acc = 0

        vqa = Answerer()
        reranker = Reranker(
            project_id=os.environ["GOOGLE_CLOUD_PROJECT_ID"],
            location=os.environ["GOOGLE_CLOUD_LOCATION"],
            memory_path=embedding_base_path,
            queries=queries,
            video_ext=video_ext
        )
        
        # Initialize the Questioner with the number of rounds from args
        # 使用调优得到的最佳参数组合
        questioner = Questioner(
            n_clusters=4,        # m=4 (MER2024调优结果)
            alpha_threshold=0.006, # α=0.006 (MER2024调优结果)
            beta_threshold=0.007   # β=0.007 (MER2024调优结果)
        )
        
        logger.info("🎯 使用调优得到的最佳参数组合:")
        logger.info(f"   K-means簇数 (m): 4")
        logger.info(f"   簇间熵阈值 (α): 0.006")
        logger.info(f"   簇内熵阈值 (β): 0.007")
        logger.info(f"   交互轮数: {args.num_rounds}")
        
        # iterate with query
        # 处理全部数据
        max_videos = len(predictions)
        logger.info(f"📊 开始处理全部 {max_videos} 个视频")
        
        for idx, row in enumerate(predictions):
            record = {}
            rank_history = []
            
            total+=1
            
            target_vid = row["query_id"]
            topk = row["org_ranking"]
            
            logger.info(f"🎬 处理视频 {idx+1}/{max_videos}: {target_vid}")
            logger.debug(f"Top-k videos: {topk}")
            
            # 每次都重新运行，不跳过已存在的文件
            # 如果文件已存在，直接覆盖
            
            # Load videos for VQA
            logger.info("Loading videos for VQA...")
            try:
                # Find target video path
                target_path = dataset_paths.find_video_path(target_vid)
                if target_path is None:
                    raise FileNotFoundError(f"Could not find target video {target_vid} in any of the video paths")
                
                # Load target video
                vqa.load_video(str(target_path))
                
                # Find and load top-k videos
                topk_video_paths = []
                for vid in topk:
                    video_path = dataset_paths.find_video_path(vid)
                    if video_path is not None:
                        topk_video_paths.append(str(video_path))
                    else:
                        # If video not found, try constructing path manually as fallback
                        for video_base_path in video_base_paths:
                            fallback_path = os.path.join(video_base_path, f"{vid}{video_ext}")
                            if os.path.exists(fallback_path):
                                topk_video_paths.append(fallback_path)
                                break
                
                # Load top-k videos (no longer needed since we removed load_topk method)
                if not topk_video_paths:
                    logger.warning(f"No top-k videos found for {target_vid}")
            
            except Exception as e:
                logger.error(f"Error loading videos: {str(e)}")
                continue
            if target_path is None:
                raise FileNotFoundError(f"Could not find target video {target_vid} in any of the video paths")
            
            anchor_captions = ""
            anchor = topk[0]
            for k in topk:
                try:
                    anchor_captions = video_captions[k]
                    break
                except:
                    pass

            reranker.init_embedding(target_vid)
            # 修复：为初始排名计算提供查询嵌入
            # 找到当前视频在text_embs中的索引
            current_video_idx = None
            for query_idx, query in enumerate(queries):
                # 根据数据集类型处理视频ID
                if args.dataset == "mer2024":
                    # MER2024: 直接使用name作为video_id，不需要去除扩展名
                    if query["video"] == target_vid:
                        current_video_idx = query_idx
                        break
                else:
                    # MAFW: 需要去除扩展名
                    if query["video"].replace(video_ext, "") == target_vid:
                        current_video_idx = query_idx
                        break
            
            if current_video_idx is not None:
                initial_query_emb = text_embs[current_video_idx]  # 使用当前查询的文本嵌入
                _, initial_rank = reranker.rerank(target_vid, video_embs, initial_query_emb)
            else:
                logger.error(f"Could not find text embedding for video {target_vid}")
                continue
            logger.info(f"Initial rank: {initial_rank}")
            
            # On experiment, we did not use this condition
            # You can use this condition to skip the videos if initial rank is good enough
            # if initial_rank==1:
            #     total-=1  
            #     continue
            # if initial_rank < 10:
            #     total-=1
            #     continue

            # 初始化记录结构
            record = {
                'video_name': target_vid,  # 目标视频的id/name
                'initial_rank': int(initial_rank),  # 初始排名
            }
            rank_history.append(initial_rank)
            
            # Reset the questioner for a new conversation
            questioner.reset_conversation(
                target_video_id=target_vid
            )
            
            # Reset the reformatter with initial description
            reranker.reset_reformatter(initial_description=anchor_captions)
            
            # Get embeddings for top-k candidates for entropy analysis
            topk_embeddings = []
            for vid in topk:
                try:
                    # Find the index of the video in the queries list
                    vid_index = None
                    for query_idx, query in enumerate(queries):
                        # 根据数据集类型处理视频ID
                        if args.dataset == "mer2024":
                            # MER2024: 直接使用name作为video_id，不需要去除扩展名
                            if query["video"] == vid:
                                vid_index = query_idx
                                break
                        else:
                            # MAFW: 需要去除扩展名
                            if query["video"].replace(video_ext, "") == vid:
                                vid_index = query_idx
                                break
                    
                    if vid_index is not None:
                        topk_embeddings.append(video_embs[vid_index])
                    else:
                        logger.warning(f"Could not find embeddings for video {vid}")
                except Exception as e:
                    logger.warning(f"Error getting embeddings for video {vid}: {str(e)}")
            
            # Convert to numpy array if we have embeddings
            embeddings_array = np.array(topk_embeddings) if topk_embeddings else None
            
            # Generate the first question using the Questioner with entropy analysis
            question_result = questioner.generate_question(
                video_captions=anchor_captions,
                embeddings=embeddings_array,
                top_k_videos=topk  # Pass the top-k video IDs for micro-scrutiny
            )
            
            # Extract the question and record entropy analysis for first round
            response = question_result["question"]
            
            # 记录第一轮的策略信息
            strategy = question_result.get('strategy', 'unknown')
            if strategy == 'ASK':
                # 判断是macro还是micro
                entropy_info = question_result.get('entropy_info', {})
                inter_entropy = entropy_info.get('inter_cluster_entropy', 0)
                intra_entropy = entropy_info.get('intra_cluster_entropy', 0)
                alpha_threshold = 0.006
                beta_threshold = 0.007
                
                if inter_entropy > alpha_threshold:
                    strategy_record = 'ask_macro'
                else:
                    strategy_record = 'ask_micro'
            else:
                strategy_record = 'refine'
            
            record['strategy_1'] = strategy_record
            
            # 标记是否已经达到rank 1
            reached_rank1 = False
            
            for i in range(args.num_rounds):
                round_num = i + 1
                logger.info(f"Processing round {round_num} of {args.num_rounds}")
                logger.debug(f"Question: {response}")
                
                # 记录问题
                record[f'question_{round_num}'] = response
                
                # Process the question and get an answer
                answer, _ = asyncio.run(vqa.async_ask(response))  # 忽略 before_aggr，直接使用 answer
                
                # 记录答案
                record[f'answer_{round_num}'] = answer
                
                # 使用重构器生成新的描述文本
                reformatted_description = reranker.reformat_dialogue(
                    question=response,
                    answer=answer,
                    max_tokens=300  # 减少token数，确保文本更短
                )
                print(f"描述长度: {len(reformatted_description)}")
                
                # 记录重述查询
                record[f'reformat_{round_num}'] = reformatted_description

                # 使用重构后的描述生成Google嵌入
                emb = reranker.get_image_video_text_embeddings(contextual_text=reformatted_description)
                reranked_topk, target_rank = reranker.rerank(target_vid, video_embs, emb.text_embedding)
                
                # 检查是否达到rank 1
                if target_rank == 1:
                    reached_rank1 = True
                    logger.info(f"Target video {target_vid} reached rank 1 in round {round_num}")
                
                # 记录排名（如果已经达到rank 1，后续轮次都记为1）
                if reached_rank1:
                    record[f'rank_{round_num}'] = 1
                    rank_history.append(1)
                else:
                    record[f'rank_{round_num}'] = int(target_rank)
                    rank_history.append(target_rank)
                
                # 记录top10视频列表
                record[f'top10_{round_num}'] = reranked_topk[:10]  # 只取前10个
                
                reranked_top1_caption = ""
                for k in reranked_topk:
                    try:
                        reranked_top1_caption = video_captions[k]
                        break
                    except:
                        pass
                
                # Record the answer in the questioner's conversation log
                questioner.record_answer(
                    answer=answer,
                    reranked_caption=reranked_top1_caption,
                    target_rank=target_rank,
                    reranked_topk=reranked_topk,
                    reformatted_description=reformatted_description
                )
                
                logger.info(f"Answer: {answer}")
                logger.info(f"Reformatted description: {reformatted_description}")
                logger.info(f"Target rank: {target_rank}")

                # Check if this is the last round based on the loop counter
                is_last_round = (i == args.num_rounds - 1)
                
                # Generate the next question if not the last round
                if not is_last_round:
                    next_round = round_num + 1
                    logger.info(f"Generating question for round {next_round}")
                    
                    # Update embeddings for the new top-k candidates after reranking
                    topk_embeddings = []
                    for vid in reranked_topk:
                        try:
                            # Find the index of the video in the queries list
                            vid_index = None
                            for query_idx, query in enumerate(queries):
                                # 根据数据集类型处理视频ID
                                if args.dataset == "mer2024":
                                    # MER2024: 直接使用name作为video_id，不需要去除扩展名
                                    if query["video"] == vid:
                                        vid_index = query_idx
                                        break
                                else:
                                    # MAFW: 需要去除扩展名
                                    if query["video"].replace(video_ext, "") == vid:
                                        vid_index = query_idx
                                        break
                            
                            if vid_index is not None:
                                topk_embeddings.append(video_embs[vid_index])
                            else:
                                logger.warning(f"Could not find embeddings for video {vid}")
                        except Exception as e:
                            logger.warning(f"Error getting embeddings for video {vid}: {str(e)}")
                    
                    # Convert to numpy array if we have embeddings
                    embeddings_array = np.array(topk_embeddings) if topk_embeddings else None
                    
                    # Generate the next question using the Questioner with entropy analysis
                    question_result = questioner.generate_question(
                        video_captions=anchor_captions,
                        embeddings=embeddings_array,
                        temperature=0.7  # Use higher temperature after first round
                    )
                    response = question_result["question"]
                    
                    # 记录下一轮的策略信息
                    strategy = question_result.get('strategy', 'unknown')
                    if strategy == 'ASK':
                        # 判断是macro还是micro
                        entropy_info = question_result.get('entropy_info', {})
                        inter_entropy = entropy_info.get('inter_cluster_entropy', 0)
                        intra_entropy = entropy_info.get('intra_cluster_entropy', 0)
                        alpha_threshold = 0.006
                        beta_threshold = 0.007
                        
                        if inter_entropy > alpha_threshold:
                            strategy_record = 'ask_macro'
                        else:
                            strategy_record = 'ask_micro'
                    else:
                        strategy_record = 'refine'
                    
                    record[f'strategy_{next_round}'] = strategy_record
                else:
                    logger.info(f"Final round {round_num} completed, no more questions will be generated")

            # 记录总轮数
            record['total_rounds'] = len(rank_history) - 1  # 减去初始轮次
            
            # Create the output directory if it doesn't exist
            os.makedirs(base_chat_log_dir, exist_ok=True)
            
            # 保存文件并显示保存位置
            output_file = os.path.join(base_chat_log_dir, f'log_{target_vid}.json')
            with open(output_file, 'w') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 已保存文件 ({idx+1}/{max_videos}): {output_file}")
            logger.debug(f"Rank history: {rank_history}")
            
            # avg rank
            for _idx, r in enumerate(rank_history):
                rank_sum[_idx]+=r
                logger.info(f"Average ranking in round {_idx}: {round(rank_sum[_idx]/total, 1)} among {total} samples")
            
            # topk
            if target_vid == reranked_topk[0]:
                top1_acc += 1
            if target_vid in reranked_topk[:5]:
                top5_acc += 1
            if target_vid in reranked_topk[:10]:
                top10_acc += 1
            
            ## zs retrieval ##
            ranking = row["org_ranking"]
            if target_vid == ranking[0]:
                zs_top1_acc += 1
            if target_vid in ranking[:5]:
                zs_top5_acc += 1
            if target_vid in ranking[:10]:
                zs_top10_acc += 1

            logger.info(f"Top-1 accuracy: {top1_acc / total * 100:.2f}% ({zs_top1_acc}->{top1_acc})/{total}")
            logger.info(f"Top-5 accuracy: {top5_acc / total * 100:.2f}% ({zs_top5_acc}->{top5_acc})/{total}")
            logger.info(f"Top-10 accuracy: {top10_acc / total * 100:.2f}% ({zs_top10_acc}->{top10_acc})/{total}")

    except Exception as e:
        logger.error(f"Error running MERLIN: {str(e)}")
        if args.debug:
            raise
        sys.exit(1)
if __name__ == "__main__":
    # python run_merlin.py --dataset msvd --data_path /path/to/data
    main()
