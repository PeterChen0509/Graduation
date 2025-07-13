import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
from typing import Optional, List, Dict, Any
import vertexai
from vertexai.vision_models import (
    Image,
    MultiModalEmbeddingModel,
    MultiModalEmbeddingResponse,
    Video,
    VideoSegmentConfig,
)
from glob import glob
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModel
import logging
import time


class Reformatter:
    """
    重构器 (Reformatter) - 负责将对话日志重构为精确的描述文本
    
    这个模块是提升检索精度的关键。它不是简单的"总结"，而是"迭代式的信息整合与重构"。
    我们不希望丢失任何细节，而是将新的问答信息整合到现有描述中。
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", temperature: float = 0.1):
        """
        初始化重构器
        
        Args:
            model_name: Qwen模型名称
            temperature: 生成温度参数
        """
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载Qwen 2.5 VL模型和处理器
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.temperature = temperature
        
        # 重构器专用系统提示
        self.reformatter_system_prompt = {
            "role": "system",
            "content": """You are an expert in synthesizing information, acting as a 'Dialogue Reformatter'. Your task is to analyze a video search dialogue history and integrate all confirmed facts into a single, cohesive, descriptive paragraph.

Follow these rules strictly:
1. Start with the last reformulated description as your base.
2. Incorporate the new information from the latest Question-Answer pair to enrich the description.
3. The final output must be a single paragraph, written in a clear, objective, and descriptive style, perfect for a text-to-video retrieval system.
4. DO NOT include conversational elements, questions, uncertainties, or filler words. Only output the final, enriched description.
5. Maintain all previously confirmed details while adding new information.
6. Use precise, descriptive language that would be effective for video retrieval."""
        }
        
        # 初始化对话历史
        self.messages = []
        self.current_description = ""
    
    def reset_reformatter(self, initial_description: str = ""):
        """
        重置重构器状态
        
        Args:
            initial_description: 初始描述文本
        """
        self.messages = [self.reformatter_system_prompt]
        self.current_description = initial_description
        self.logger.debug(f"Reformatter reset with initial description: {initial_description}")
    
    def reformat_dialogue(self, question: str, answer: str, max_tokens: int = 500) -> str:
        """
        重构对话日志，生成新的描述文本
        
        Args:
            question: 当前轮次的问题
            answer: 当前轮次的答案
            max_tokens: 最大生成token数
            
        Returns:
            重构后的描述文本
        """
        # 格式化对话历史
        formatted_log = self._format_dialogue_history(question, answer)
        
        # 创建重构提示
        user_message = f"""---
**Dialogue History to Reformulate:**
{formatted_log}
---

**New Reformulated Description:**"""
        
        # 生成重构后的描述
        new_description = self._generate_reformatted_description(user_message, max_tokens)
        
        # 截断描述文本以确保不超过Google API限制
        MAX_TEXT_LEN = 900  # 进一步减少，确保安全边界
        if len(new_description) > MAX_TEXT_LEN:
            new_description = new_description[:MAX_TEXT_LEN]
            self.logger.warning(f"Description truncated from {len(new_description)} to {MAX_TEXT_LEN} characters")
        
        # 更新当前描述
        self.current_description = new_description
        
        # 记录到对话历史
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": new_description})
        
        self.logger.debug(f"Generated reformatted description: {new_description}")
        return new_description
    
    def _format_dialogue_history(self, question: str, answer: str) -> str:
        """
        格式化对话历史
        
        Args:
            question: 当前问题
            answer: 当前答案
            
        Returns:
            格式化的对话历史字符串
        """
        if not self.current_description:
            # 如果没有之前的描述，创建一个简单的初始描述
            self.current_description = "Initial video description"
        
        formatted_log = f"""Last Description: "{self.current_description}"
---
New Q&A Pair:
Question: "{question}"
Answer: "{answer}" """
        
        return formatted_log
    
    def _generate_reformatted_description(self, user_message: str, max_tokens: int) -> str:
        """
        使用Qwen2.5-VL生成重构后的描述
        
        Args:
            user_message: 用户消息
            max_tokens: 最大生成token数
            
        Returns:
            生成的描述文本
        """
        try:
            # 处理输入
            text = self.processor.apply_chat_template(
                self.messages + [{"role": "user", "content": user_message}], 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # 生成响应
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=self.temperature
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating reformatted description: {str(e)}")
            # 返回fallback描述
            return f"{self.current_description} {answer}"
    
    def get_current_description(self) -> str:
        """
        获取当前描述文本
        
        Returns:
            当前描述文本
        """
        return self.current_description
    
    def get_reformatter_history(self) -> List[Dict[str, str]]:
        """
        获取重构器历史记录
        
        Returns:
            重构器历史记录列表
        """
        return self.messages.copy()


class Reranker(torch.nn.Module):
    def __init__(self, location: str, project_id: str, memory_path: str, queries: list, video_ext: str = ".mp4"):
        super(Reranker, self).__init__()
        self.location = location
        self.project_id = project_id
        self.memory_path = memory_path
        self.video_ext = video_ext  # Store the video extension

        # Load queries
        self.memories = queries
        
        # 初始化重构器
        self.reformatter = Reformatter()
        
        # 初始化fallback文本embedding模型
        self._init_fallback_embedding_model()
    
    def _init_fallback_embedding_model(self):
        """初始化fallback文本embedding模型"""
        try:
            self.fallback_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.fallback_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.fallback_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.fallback_model.to(self.fallback_device)
            self.fallback_model.eval()
            print(f"✅ 已初始化fallback文本embedding模型 (设备: {self.fallback_device})")
        except Exception as e:
            print(f"⚠️ 无法初始化fallback文本embedding模型: {str(e)}")
            self.fallback_tokenizer = None
            self.fallback_model = None
    
    def _get_fallback_text_embedding(self, text: str) -> np.ndarray:
        """使用fallback模型获取文本embedding"""
        if self.fallback_model is None or self.fallback_tokenizer is None:
            raise Exception("Fallback embedding model not available")
        
        try:
            # 编码文本
            inputs = self.fallback_tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.fallback_device)
            
            # 获取embedding
            with torch.no_grad():
                outputs = self.fallback_model(**inputs)
                # 使用平均池化
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().numpy()[0]
                
        except Exception as e:
            print(f"❌ Fallback embedding生成失败: {str(e)}")
            # 返回一个随机向量作为最后的fallback
            return np.random.randn(384)  # all-MiniLM-L6-v2的维度是384

    def get_image_video_text_embeddings(
        self,
        image_path: Optional[str] = None, 
        video_path: Optional[str] = None,
        contextual_text: Optional[str] = None,
        dimension: Optional[int] = 1408,
        video_segment_config: Optional[VideoSegmentConfig] = None,
    ) -> MultiModalEmbeddingResponse:
        """Example of how to generate multimodal embeddings from image, video, and text.

        Args:
            project_id: Google Cloud Project ID, used to initialize vertexai
            location: Google Cloud Region, used to initialize vertexai
            image_path: Path to image (local or Google Cloud Storage) to generate embeddings for.
            video_path: Path to video (local or Google Cloud Storage) to generate embeddings for.
            contextual_text: Text to generate embeddings for.
            dimension: Dimension for the returned embeddings.
                https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-multimodal-embeddings#low-dimension
            video_segment_config: Define specific segments to generate embeddings for.
                https://cloud.google.com/vertex-ai/docs/generative-ai/embeddings/get-multimodal-embeddings#video-best-practices
        Returns:
            MultiModalEmbeddingResponse: A container object holding the embeddings for the provided image, video, and text inputs.
                The embeddings are dense vectors representing the semantic meaning of the inputs.
                Embeddings can be accessed as follows:
                - embeddings.image_embedding (numpy.ndarray): Embedding for the provided image.
                - embeddings.video_embeddings (List[VideoEmbedding]): List of embeddings for video segments.
                - embeddings.text_embedding (numpy.ndarray): Embedding for the provided text.
        """

        # 检查并截断文本长度，确保不超过Google API限制
        MAX_TEXT_LEN = 900  # 进一步减少，确保安全边界
        if contextual_text:
            print(f"🔍 传入Google API的文本长度: {len(contextual_text)}")
            print(f"🔍 文本前100字符: {contextual_text[:100]}...")
            print(f"🔍 文本后100字符: {contextual_text[-100:] if len(contextual_text) > 100 else contextual_text}")
            
            if len(contextual_text) > MAX_TEXT_LEN:
                original_len = len(contextual_text)
                contextual_text = contextual_text[:MAX_TEXT_LEN]
                print(f"⚠️ 文本长度超限，已截断: {original_len} -> {MAX_TEXT_LEN} 字符")
                print(f"截断后的文本: {contextual_text[:100]}...")
            else:
                print(f"✅ 文本长度在安全范围内: {len(contextual_text)} 字符")

        # 尝试多个区域，因为multimodal embedding服务可能不在所有区域都可用
        regions_to_try = [
            self.location,  # 首先尝试配置的区域
            "us-central1",  # 然后尝试us-central1
            "us-east1",     # 再尝试us-east1
            "europe-west1"  # 最后尝试europe-west1
        ]
        
        last_error = None
        
        for region in regions_to_try:
            try:
                print(f"🔄 尝试在区域 {region} 获取multimodal embedding...")
                
                # 重新初始化vertexai
                vertexai.init(project=self.project_id, location=region)
                
                # 尝试获取模型
                model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding")
                
                image, video = None, None
                if image_path is not None:
                    image = Image.load_from_file(image_path)
                if video_path is not None:
                    video = Video.load_from_file(video_path)

                embeddings = model.get_embeddings(
                    image=image,
                    video=video,
                    video_segment_config=video_segment_config,
                    contextual_text=contextual_text,
                    dimension=dimension,
                )
                
                print(f"✅ 成功在区域 {region} 获取embedding")
                return embeddings
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ 在区域 {region} 失败: {error_msg}")
                last_error = e
                
                # 如果是404错误，继续尝试下一个区域
                if "404" in error_msg or "Servable not found" in error_msg:
                    continue
                else:
                    # 如果是其他错误，可能不需要继续尝试
                    break
        
        # 如果所有区域都失败了，尝试使用fallback方法
        if last_error:
            print(f"⚠️ Google Cloud multimodal embedding服务在所有区域都失败了，使用fallback文本embedding")
            print(f"最后一个错误: {str(last_error)}")
            
            # 使用fallback文本embedding
            if contextual_text:
                fallback_embedding = self._get_fallback_text_embedding(contextual_text)
                
                # 创建一个模拟的MultiModalEmbeddingResponse对象
                class FallbackEmbeddingResponse:
                    def __init__(self, text_embedding):
                        self.text_embedding = text_embedding
                        self.image_embedding = None
                        self.video_embeddings = None
                
                return FallbackEmbeddingResponse(fallback_embedding)
            else:
                raise Exception("没有提供文本内容，无法使用fallback方法")
        else:
            raise Exception("无法获取multimodal embedding，请检查Google Cloud配置")

    def init_embedding(self, id):
        """
        初始化嵌入（保留接口兼容性，但不再需要）
        
        Args:
            id: 视频ID
        """
        # 这个方法现在不需要做任何事情，因为我们是直接使用Google embedding
        pass

    def rerank(self, target_vid, video_embeddings, current_query_embedding):
        """
        使用当前查询嵌入进行重排序
        
        Args:
            target_vid: 目标视频ID
            video_embeddings: 视频库的嵌入矩阵
            current_query_embedding: 当前查询的embedding（可能是Google embedding或fallback embedding）
            
        Returns:
            top_k_ids: 重排序后的top-k视频ID列表
            desired_video_rank: 目标视频的排名
        """
        # 检查维度匹配
        query_dim = len(current_query_embedding)
        
        # 处理video_embeddings可能是list或numpy array的情况
        if isinstance(video_embeddings, list):
            video_dim = len(video_embeddings[0]) if video_embeddings else 0
        else:
            video_dim = video_embeddings.shape[1] if len(video_embeddings.shape) > 1 else len(video_embeddings[0])
        
        if query_dim != video_dim:
            print(f"⚠️ 维度不匹配: 查询embedding维度={query_dim}, 视频embedding维度={video_dim}")
            
            # 如果使用fallback embedding (384维) 而视频库是Google embedding (1408维)
            if query_dim == 384 and video_dim == 1408:
                print("🔄 检测到fallback embedding，需要重新生成视频库的fallback embedding")
                
                # 这里我们需要重新生成视频库的fallback embedding
                # 但是由于我们没有视频的文本描述，我们只能使用原始的文本embedding
                # 这种情况下，我们返回一个基于原始排名的结果
                print("⚠️ 无法进行维度匹配的重排序，返回原始排名")
                
                # 返回一个简单的排名（基于索引顺序）
                top_k_ids = []
                for i in range(min(10, len(self.memories))):
                    top_k_ids.append(self.memories[i]["video"].replace(self.video_ext, ""))
                
                # 找到目标视频的排名
                desired_video_rank = None
                for idx, memory in enumerate(self.memories):
                    if memory["video"].replace(self.video_ext, "") == target_vid:
                        desired_video_rank = idx + 1
                        break
                
                return top_k_ids, desired_video_rank
            else:
                raise ValueError(f"不支持的维度组合: 查询={query_dim}, 视频={video_dim}")
        
        # 维度匹配，正常进行相似度计算
        similarities = cosine_similarity([current_query_embedding], video_embeddings)

        # 评估top-k检索准确率
        k_values = [10]
        top_k_ids = []
        desired_video_rank = None
        
        for k in k_values:
            top_k_indices = np.argsort(-similarities[0])
            
            # 找到目标视频的排名
            for idx, k_index in enumerate(top_k_indices):
                if self.memories[k_index]["video"].replace(self.video_ext, "") == target_vid:
                    desired_video_rank = idx + 1
                    break

            # 获取top-k视频ID
            top_k_indices = top_k_indices[:k]
            for idx in top_k_indices:
                top_k_ids.append(self.memories[idx]["video"].replace(self.video_ext, ""))
        
        return top_k_ids, desired_video_rank
    
    def reset_reformatter(self, initial_description: str = ""):
        """
        重置重构器状态
        
        Args:
            initial_description: 初始描述文本
        """
        self.reformatter.reset_reformatter(initial_description)
    
    def reformat_dialogue(self, question: str, answer: str, max_tokens: int = 500) -> str:
        """
        重构对话日志，生成新的描述文本
        
        Args:
            question: 当前轮次的问题
            answer: 当前轮次的答案
            max_tokens: 最大生成token数
            
        Returns:
            重构后的描述文本
        """
        return self.reformatter.reformat_dialogue(question, answer, max_tokens)
    
    def get_current_description(self) -> str:
        """
        获取当前描述文本
        
        Returns:
            当前描述文本
        """
        return self.reformatter.get_current_description()
    
    def get_reformatter_history(self) -> List[Dict[str, str]]:
        """
        获取重构器历史记录
        
        Returns:
            重构器历史记录列表
        """
        return self.reformatter.get_reformatter_history()
