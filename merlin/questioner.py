import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class Questioner:
    """
    A class responsible for generating questions about video content to help
    refine search results in the MERLIN system.
    
    This class maintains conversation history internally and can be reset when a new conversation begins.
    Now includes entropy-based strategy selection for ASK vs REFINE decisions.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", temperature: float = 0.2,
                 n_clusters: int = 3, alpha_threshold: float = 0.5, beta_threshold: float = 0.3):
        """
        Initialize the Questioner with Qwen 2.5 VL model and entropy analysis parameters.
        
        Args:
            model_name: Qwen model name to use for question generation
            temperature: Temperature parameter for question generation
            n_clusters: Number of clusters for K-means clustering
            alpha_threshold: Threshold for inter-cluster entropy (ASK if > alpha)
            beta_threshold: Threshold for intra-cluster entropy (ASK if > beta)
        """
        self.logger = logging.getLogger("MERLIN")
        
        # Set up device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Qwen 2.5 VL model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        self.default_temperature = temperature
        
        # Entropy analysis parameters
        self.n_clusters = n_clusters
        self.alpha_threshold = alpha_threshold
        self.beta_threshold = beta_threshold
        
        # Define the default system prompt
        self.default_system_prompt = {
            "role": "system",
            "content": """
            You are given caption about certain video(anchor video) and query used to retrieve the anchor video. However this video may not be the exact video the I am looking for. 
            Your role is to ask question about the video I have in mind to get more information about video. You have 3 rounds and you can only ask one question at a time.
            Don't just answer in yes or no. Answer concisely.
            Focus on attributes like number of people, color, shape.
            """
        }
        
        # Initialize conversation state
        self.messages = []
        self.system_prompt = self.default_system_prompt
        self.conversation_log = []
        self.current_video_captions = ""
        self.target_video_id = None
        
        # Initialize entropy analysis state
        self.asked_topics = []  # Track asked question types for macro-exploration
        self.current_embeddings = None  # Store current top-k embeddings
        self.cluster_labels = None  # Store cluster assignments
        self.cluster_centers = None  # Store cluster centers
    
    def reset_conversation(self, target_video_id: Optional[str] = None):
        """
        Reset the conversation history to start a new conversation.
        
        Args:
            target_video_id: Optional ID of the target video for this conversation
        """
        self.messages = []
        self.system_prompt = self.default_system_prompt
        self.conversation_log = []
        self.current_video_captions = ""
        self.target_video_id = target_video_id
        
        # Reset entropy analysis state
        self.asked_topics = []
        self.current_embeddings = None
        self.current_top_k_videos = None
        self.cluster_labels = None
        self.cluster_centers = None
        
        self.logger.debug(f"Conversation history reset for video ID: {target_video_id}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            List of message dictionaries representing the conversation history
        """
        return self.messages.copy()
    
    def get_conversation_log(self) -> List[Dict[str, Any]]:
        """
        Get the structured conversation log with questions and answers.
        
        Returns:
            List of dictionaries containing question-answer pairs and metadata
        """
        return self.conversation_log.copy()
    
    def add_to_conversation(self, role: str, content: str):
        """
        Add a message to the conversation history.
        
        Args:
            role: The role of the message sender ("system", "user", or "assistant")
            content: The content of the message
        """
        message = {"role": role, "content": content}
        self.messages.append(message)
        self.logger.debug(f"Added {role} message to conversation history")
    
    def compute_entropy_metrics(self, embeddings: np.ndarray) -> Tuple[float, float, Dict[str, Any]]:
        """
        Compute inter-cluster and intra-cluster entropy metrics.
        
        Args:
            embeddings: Array of shape (n_videos, embedding_dim) containing video embeddings
            
        Returns:
            Tuple of (inter_cluster_entropy, intra_cluster_entropy, cluster_info)
        """
        if len(embeddings) < self.n_clusters:
            # If we have fewer videos than clusters, adjust cluster number
            actual_clusters = min(len(embeddings), 2)
            self.logger.warning(f"Adjusting clusters from {self.n_clusters} to {actual_clusters} due to insufficient data")
        else:
            actual_clusters = self.n_clusters
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_
        
        # Store clustering results
        self.cluster_labels = cluster_labels
        self.cluster_centers = cluster_centers
        
        # Calculate cluster sizes
        cluster_sizes = np.bincount(cluster_labels, minlength=actual_clusters)
        total_videos = len(embeddings)
        
        # Compute inter-cluster entropy
        inter_cluster_entropy = self._compute_inter_cluster_entropy(cluster_centers, cluster_sizes)
        
        # Compute intra-cluster entropy
        intra_cluster_entropy = self._compute_intra_cluster_entropy(embeddings, cluster_labels, cluster_centers, cluster_sizes)
        
        cluster_info = {
            'n_clusters': actual_clusters,
            'cluster_sizes': cluster_sizes.tolist(),
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': cluster_centers.tolist()
        }
        
        self.logger.info(f"Inter-cluster entropy: {inter_cluster_entropy:.6f}")
        self.logger.info(f"Intra-cluster entropy: {intra_cluster_entropy:.6f}")
        
        return inter_cluster_entropy, intra_cluster_entropy, cluster_info
    
    def _compute_inter_cluster_entropy(self, cluster_centers: np.ndarray, cluster_sizes: np.ndarray) -> float:
        """
        Compute inter-cluster entropy using weighted variance of cosine distances.
        
        Args:
            cluster_centers: Array of shape (n_clusters, embedding_dim)
            cluster_sizes: Array of shape (n_clusters,) containing cluster sizes
            
        Returns:
            Inter-cluster entropy value
        """
        n_clusters = len(cluster_centers)
        if n_clusters < 2:
            return 0.0
        
        # Compute cosine distances between all pairs of cluster centers
        distances = []
        weights = []
        
        for i in range(n_clusters - 1):
            for j in range(i + 1, n_clusters):
                # Cosine similarity
                cos_sim = np.dot(cluster_centers[i], cluster_centers[j]) / (
                    np.linalg.norm(cluster_centers[i]) * np.linalg.norm(cluster_centers[j])
                )
                # Cosine distance
                distance = 1 - cos_sim
                distances.append(distance)
                
                # Weight based on cluster sizes
                weight = cluster_sizes[i] + cluster_sizes[j]
                weights.append(weight)
        
        if not distances:
            return 0.0
        
        distances = np.array(distances)
        weights = np.array(weights)
        
        # Compute weighted mean
        weighted_mean = np.average(distances, weights=weights)
        
        # Compute weighted variance
        weighted_variance = np.average((distances - weighted_mean) ** 2, weights=weights)
        
        return weighted_variance
    
    def _compute_intra_cluster_entropy(self, embeddings: np.ndarray, cluster_labels: np.ndarray, 
                                     cluster_centers: np.ndarray, cluster_sizes: np.ndarray) -> float:
        """
        Compute intra-cluster entropy using weighted average of cluster variances.
        
        Args:
            embeddings: Array of shape (n_videos, embedding_dim)
            cluster_labels: Array of shape (n_videos,) containing cluster assignments
            cluster_centers: Array of shape (n_clusters, embedding_dim)
            cluster_sizes: Array of shape (n_clusters,) containing cluster sizes
            
        Returns:
            Intra-cluster entropy value
        """
        n_clusters = len(cluster_centers)
        total_videos = len(embeddings)
        
        if total_videos == 0:
            return 0.0
        
        # Compute variance for each cluster
        cluster_variances = []
        
        for i in range(n_clusters):
            if cluster_sizes[i] == 0:
                continue
                
            # Get embeddings for this cluster
            cluster_mask = cluster_labels == i
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) == 0:
                continue
            
            # Compute cosine distances to cluster center
            distances = []
            for emb in cluster_embeddings:
                cos_sim = np.dot(emb, cluster_centers[i]) / (
                    np.linalg.norm(emb) * np.linalg.norm(cluster_centers[i])
                )
                distance = 1 - cos_sim
                distances.append(distance)
            
            # Compute variance (mean of squared distances)
            variance = np.mean(np.array(distances) ** 2)
            cluster_variances.append(variance)
        
        if not cluster_variances:
            return 0.0
        
        # Compute weighted average using cluster sizes
        weighted_entropy = np.average(cluster_variances, weights=cluster_sizes[:len(cluster_variances)])
        
        return weighted_entropy
    
    def select_strategy(self, inter_cluster_entropy: float, intra_cluster_entropy: float) -> str:
        """
        Select strategy based on entropy values.
        
        Args:
            inter_cluster_entropy: Inter-cluster entropy value
            intra_cluster_entropy: Intra-cluster entropy value
            
        Returns:
            Strategy: "ASK" or "REFINE"
        """
        if inter_cluster_entropy > self.alpha_threshold:
            strategy = "ASK"
            reason = f"Inter-cluster entropy ({inter_cluster_entropy:.4f}) > α ({self.alpha_threshold})"
        elif intra_cluster_entropy > self.beta_threshold:
            strategy = "ASK"
            reason = f"Intra-cluster entropy ({intra_cluster_entropy:.4f}) > β ({self.beta_threshold})"
        else:
            strategy = "REFINE"
            reason = f"Both entropies below thresholds (inter: {inter_cluster_entropy:.4f}, intra: {intra_cluster_entropy:.4f})"
        
        self.logger.info(f"Strategy selected: {strategy} - {reason}")
        self.logger.info(f"Detailed comparison:")
        self.logger.info(f"  Inter-cluster entropy: {inter_cluster_entropy:.6f} vs Alpha threshold: {self.alpha_threshold:.6f}")
        self.logger.info(f"  Intra-cluster entropy: {intra_cluster_entropy:.6f} vs Beta threshold: {self.beta_threshold:.6f}")
        return strategy
    
    def determine_ask_strategy(self, inter_cluster_entropy: float, intra_cluster_entropy: float) -> str:
        """
        Determine the specific ASK strategy (macro-exploration or micro-scrutiny).
        
        Args:
            inter_cluster_entropy: Inter-cluster entropy value
            intra_cluster_entropy: Intra-cluster entropy value
            
        Returns:
            Strategy: "macro-exploration" or "micro-scrutiny"
        """
        if inter_cluster_entropy > self.alpha_threshold:
            return "macro-exploration"
        else:
            return "micro-scrutiny"
    
    def generate_question(self, 
                     video_captions: str,
                     embeddings: Optional[np.ndarray] = None,
                     conversation_history: Optional[List[Dict[str, str]]] = None,
                     top_k_videos: Optional[List[str]] = None,
                     max_tokens: int = 500,
                     temperature: Optional[float] = None) -> Dict[str, Any]:
        """
        Generate a question about the video based on captions, embeddings, and conversation history.
        Now supports entropy-driven strategy selection (ASK vs REFINE).
        
        Args:
            video_captions: Captions describing the video content
            embeddings: Optional array of shape (n_videos, embedding_dim) for entropy analysis
            conversation_history: Optional conversation history to consider
            max_tokens: Maximum number of tokens for the response
            temperature: Temperature parameter for generation (uses default if None)
            
        Returns:
            Dictionary containing the generated question and metadata
        """
        # Store the current video captions and top-k videos
        self.current_video_captions = video_captions
        self.current_top_k_videos = top_k_videos
        
        # Use provided conversation history or the internal one
        if conversation_history is not None:
            # Reset the conversation if a new history is provided
            self.messages = []
            
        # Initialize messages with system prompt if empty
        if not self.messages:
            self.messages = [self.system_prompt]
        
        # Perform entropy analysis if embeddings are provided
        strategy = "ASK"  # Default strategy
        entropy_info = {}
        
        self.logger.info(f"Embeddings provided: {embeddings is not None}, Shape: {embeddings.shape if embeddings is not None else 'None'}")
        
        if embeddings is not None and len(embeddings) > 0:
            self.current_embeddings = embeddings
            
            # Compute entropy metrics
            inter_cluster_entropy, intra_cluster_entropy, cluster_info = self.compute_entropy_metrics(embeddings)
            
            # Select strategy based on entropy values
            strategy = self.select_strategy(inter_cluster_entropy, intra_cluster_entropy)
            
            entropy_info = {
                'inter_cluster_entropy': inter_cluster_entropy,
                'intra_cluster_entropy': intra_cluster_entropy,
                'cluster_info': cluster_info,
                'strategy': strategy
            }
            
            self.logger.info(f"Entropy analysis - Inter: {inter_cluster_entropy:.6f}, Intra: {intra_cluster_entropy:.6f}, Strategy: {strategy}")
            self.logger.info(f"Thresholds - Alpha: {self.alpha_threshold:.6f}, Beta: {self.beta_threshold:.6f}")
            self.logger.info(f"Comparison - Inter > Alpha: {inter_cluster_entropy > self.alpha_threshold}, Intra > Beta: {intra_cluster_entropy > self.beta_threshold}")
        else:
            self.logger.warning("No embeddings provided for entropy analysis, using default ASK strategy")
            self.logger.info(f"Thresholds - Alpha: {self.alpha_threshold:.6f}, Beta: {self.beta_threshold:.6f}")
        
        # Generate question based on selected strategy
        if strategy == "ASK":
            question = self._generate_ask_question(video_captions, entropy_info, max_tokens, temperature)
        else:  # REFINE
            question = self._generate_refine_question(video_captions, max_tokens, temperature)
        
        # Add the assistant's response to the conversation
        self.add_to_conversation("assistant", question)
        
        # Add to conversation log
        self.conversation_log.append({
            "question": question,
            "question_timestamp": time.time()
        })
        
        # Return the question and metadata
        result = {
            "question": question,
            "model": "Qwen2.5-VL",
            "temperature": temperature or self.default_temperature,
            "max_tokens": max_tokens,
            "strategy": strategy,
            "entropy_info": entropy_info
        }
        
        return result
    
    def _generate_ask_question(self, video_captions: str, entropy_info: Dict[str, Any], 
                              max_tokens: int, temperature: Optional[float]) -> str:
        """
        Generate a question using ASK strategy (macro-exploration or micro-scrutiny).
        
        Args:
            video_captions: Captions describing the video content
            entropy_info: Dictionary containing entropy analysis results
            max_tokens: Maximum number of tokens for the response
            temperature: Temperature parameter for generation
            
        Returns:
            Generated question string
        """
        if not entropy_info:
            # Fallback to original question generation
            return self._generate_basic_question(video_captions, max_tokens, temperature)
        
        # Determine specific ASK strategy
        ask_strategy = self.determine_ask_strategy(
            entropy_info['inter_cluster_entropy'], 
            entropy_info['intra_cluster_entropy']
        )
        
        if ask_strategy == "macro-exploration":
            return self._generate_macro_exploration_question(video_captions, entropy_info, max_tokens, temperature)
        else:  # micro-scrutiny
            return self._generate_micro_scrutiny_question(video_captions, entropy_info, max_tokens, temperature)
    
    def _generate_macro_exploration_question(self, video_captions: str, entropy_info: Dict[str, Any], 
                                           max_tokens: int, temperature: Optional[float]) -> str:
        """
        Generate a macro-exploration question based on cluster representatives.
        Now supports specialized emotion-focused questioning for emotion datasets.
        
        Args:
            video_captions: Captions describing the video content
            entropy_info: Dictionary containing entropy analysis results
            max_tokens: Maximum number of tokens for the response
            temperature: Temperature parameter for generation
            
        Returns:
            Generated question string
        """
        # Check if this is an emotion-focused dataset
        if self._is_emotion_dataset(video_captions):
            return self._generate_emotion_focused_question(video_captions, entropy_info, max_tokens, temperature)
        
        # For non-emotion datasets, use general macro-exploration strategy
        return self._generate_general_macro_question(video_captions, entropy_info, max_tokens, temperature)
    
    def _generate_general_macro_question(self, video_captions: str, entropy_info: Dict[str, Any], 
                                       max_tokens: int, temperature: Optional[float]) -> str:
        """
        Generate general macro-exploration questions for non-emotion datasets.
        
        Args:
            video_captions: Captions describing the video content
            entropy_info: Dictionary containing entropy analysis results
            max_tokens: Maximum number of tokens for the response
            temperature: Temperature parameter for generation
            
        Returns:
            Generated question string
        """
        # Determine next question type based on asked_topics
        question_types = ['what', 'who', 'how', 'where']
        next_type = None
        
        for q_type in question_types:
            if q_type not in self.asked_topics:
                next_type = q_type
                break
        
        if next_type is None:
            # If all types have been asked, reset or use a general question
            next_type = 'what'
            self.asked_topics = []
        
        # Add to asked topics
        self.asked_topics.append(next_type)
        
        # Create prompt for macro-exploration
        user_message = f"""
        You are analyzing video candidates that have been clustered into {entropy_info['cluster_info']['n_clusters']} groups.
        The clusters have sizes: {entropy_info['cluster_info']['cluster_sizes']}.
        
        Current video caption: {video_captions}
        
        Based on the clustering analysis, generate a {next_type}-type question to help narrow down the search scope.
        Focus on distinguishing between different semantic categories or high-level concepts.
        
        Question: 
        """
        
        return self._generate_with_prompt(user_message, max_tokens, temperature)
    
    def _is_emotion_dataset(self, video_captions: str) -> bool:
        """
        Detect if the current dataset is emotion-focused based on video captions.
        
        Args:
            video_captions: Captions describing the video content
            
        Returns:
            True if emotion-focused dataset, False otherwise
        """
        # Define emotion-related keywords
        emotion_keywords = [
            'crying', 'laughing', 'smiling', 'frowning', 'angry', 'sad', 'happy', 'surprised',
            'fear', 'disgust', 'contempt', 'embarrassed', 'proud', 'confused', 'worried',
            'excited', 'calm', 'nervous', 'relaxed', 'tense', 'joyful', 'melancholy',
            'tears', 'laughter', 'expression', 'emotion', 'feeling', 'mood'
        ]
        
        caption_lower = video_captions.lower()
        emotion_count = sum(1 for keyword in emotion_keywords if keyword in caption_lower)
        
        # If more than 1 emotion keyword is found, consider it emotion-focused
        return emotion_count >= 1
    
    def _generate_emotion_focused_question(self, video_captions: str, entropy_info: Dict[str, Any], 
                                         max_tokens: int, temperature: Optional[float]) -> str:
        """
        Generate emotion-focused questions for macro-exploration in emotion datasets.
        Follows the priority order: intensity -> physical_details -> social_context -> inferred_cause
        
        Args:
            video_captions: Captions describing the video content
            entropy_info: Dictionary containing entropy analysis results
            max_tokens: Maximum number of tokens for the response
            temperature: Temperature parameter for generation
            
        Returns:
            Generated emotion-focused question string
        """
        # Extract emotion from caption
        emotion = self._extract_emotion_from_caption(video_captions)
        
        # Define emotion-focused question types with priority order
        # 1. ask_emotion_intensity: 问情绪的强度
        # 2. ask_physical_manifestation: 问表情相关的具体物理特征
        # 3. ask_social_context: 问社交环境（是否独处）
        # 4. ask_inferred_cause: 问推断出的情绪原因
        emotion_question_types = ['intensity', 'physical_details', 'social_context', 'inferred_cause']
        
        # Find next question type to ask
        next_type = None
        for q_type in emotion_question_types:
            if q_type not in self.asked_topics:
                next_type = q_type
                break
        
        if next_type is None:
            # If all types have been asked, reset to intensity (most important)
            next_type = 'intensity'
            self.asked_topics = []
        
        # Add to asked topics
        self.asked_topics.append(next_type)
        
        # Generate specific emotion-focused question based on priority order
        if next_type == 'intensity':
            question = f"How would you describe the intensity of the person's {emotion}? For example: is it subtle, moderate, or very intense?"
        elif next_type == 'physical_details':
            question = "Describe the person's key physical actions. For instance, are their eyes open or tightly shut? Is their mouth open? Are they using their hands to touch their face?"
        elif next_type == 'social_context':
            question = "Is the person alone, or are there other people visible in the scene with them?"
        elif next_type == 'inferred_cause':
            question = f"From the visual cues, what might be the underlying reason for the {emotion}? For example, does it seem like sadness, happiness, or pain?"
        else:
            # Fallback to intensity question
            question = f"To be precise, how intense is the {emotion} expression?"
        
        # Create enhanced prompt for emotion-focused question
        user_message = f"""
        You are analyzing emotion-focused video candidates that have been clustered into {entropy_info['cluster_info']['n_clusters']} groups.
        The clusters have sizes: {entropy_info['cluster_info']['cluster_sizes']}.
        
        Current video caption: {video_captions}
        
        Based on the clustering analysis and emotion detection, generate a precise question to help distinguish between similar emotional expressions.
        
        Question: {question}
        """
        
        return self._generate_with_prompt(user_message, max_tokens, temperature)
    
    def _extract_emotion_from_caption(self, video_captions: str) -> str:
        """
        Extract the primary emotion from video caption.
        
        Args:
            video_captions: Captions describing the video content
            
        Returns:
            Extracted emotion string
        """
        # Define emotion mapping
        emotion_mapping = {
            'crying': 'crying',
            'laughing': 'laughing', 
            'smiling': 'smiling',
            'frowning': 'frowning',
            'angry': 'anger',
            'sad': 'sadness',
            'happy': 'happiness',
            'surprised': 'surprise',
            'fear': 'fear',
            'disgust': 'disgust',
            'contempt': 'contempt',
            'embarrassed': 'embarrassment',
            'proud': 'pride',
            'confused': 'confusion',
            'worried': 'worry',
            'excited': 'excitement',
            'calm': 'calmness',
            'nervous': 'nervousness',
            'relaxed': 'relaxation',
            'tense': 'tension',
            'joyful': 'joy',
            'melancholy': 'melancholy'
        }
        
        caption_lower = video_captions.lower()
        
        # Find the first emotion keyword in the caption
        for keyword, emotion in emotion_mapping.items():
            if keyword in caption_lower:
                return emotion
        
        # Fallback to generic emotion
        return 'emotional expression'
    
    def _generate_micro_scrutiny_question(self, video_captions: str, entropy_info: Dict[str, Any], 
                                        max_tokens: int, temperature: Optional[float]) -> str:
        """
        Generate a micro-scrutiny question using optimized key frame selection and difference detection.
        
        Args:
            video_captions: Captions describing the video content
            entropy_info: Dictionary containing entropy analysis results
            max_tokens: Maximum number of tokens for the response
            temperature: Temperature parameter for generation
            
        Returns:
            Generated question string
        """
        # Use fallback to text-based micro-scrutiny since frame data is not available
        return self._generate_micro_scrutiny_fallback(video_captions, entropy_info, max_tokens, temperature)
    
    def _generate_micro_scrutiny_fallback(self, video_captions: str, entropy_info: Dict[str, Any], 
                                        max_tokens: int, temperature: Optional[float]) -> str:
        """
        Fallback strategy for micro-scrutiny when actual video frames are not available.
        
        Args:
            video_captions: Captions describing the video content
            entropy_info: Dictionary containing entropy analysis results
            max_tokens: Maximum number of tokens for the response
            temperature: Temperature parameter for generation
            
        Returns:
            Generated question string
        """
        # Create prompt for micro-scrutiny fallback
        user_message = f"""
        You are analyzing video candidates that are semantically similar but need detailed differentiation.
        The intra-cluster entropy is {entropy_info['intra_cluster_entropy']:.4f}, indicating high internal variation.
        
        Current video caption: {video_captions}
        
        Since we have {entropy_info['cluster_info']['n_clusters']} clusters with sizes {entropy_info['cluster_info']['cluster_sizes']},
        and the videos are very similar, generate a specific, detail-oriented question that focuses on:
        
        1. Fine-grained visual attributes (facial expressions, clothing details, accessories)
        2. Spatial relationships and positioning
        3. Background elements or environmental details
        4. Temporal aspects (if mentioned in caption)
        
        The question should be specific enough to distinguish between highly similar videos.
        
        Question: 
        """
        
        return self._generate_with_prompt(user_message, max_tokens, temperature)
    
    def _generate_refine_question(self, video_captions: str, max_tokens: int, temperature: Optional[float]) -> str:
        """
        Generate a REFINE response by analyzing conversation history and creating a refined description.
        
        This method transforms a conversational dialogue log into a single, dense, and factual 
        descriptive paragraph by integrating all confirmed positive attributes and excluding 
        negated or uncertain information.
        
        Args:
            video_captions: Captions describing the video content
            max_tokens: Maximum number of tokens for the response
            temperature: Temperature parameter for generation
            
        Returns:
            Generated refined description string
        """
        # Format the dialogue log from conversation history
        dialogue_log = self._format_dialogue_log_for_refine(video_captions)
        
        # Create specialized prompt for refinement
        user_message = f"""You are an expert information synthesizer for a high-precision video retrieval system. Your task is to transform a conversational dialogue log into a single, dense, and factual descriptive paragraph.

Carefully analyze the entire dialogue history provided below. Follow these rules strictly:
1. **Integrate ALL confirmed positive attributes**: Combine every piece of confirmed information (e.g., "crying", "covering her face") into a coherent description.
2. **Explicitly EXCLUDE all negated or uncertain information**: If the answer is "no" or "uncertain", do not include that attribute in the final description (e.g., ignore the part about glasses).
3. **Eliminate all conversational elements**: Remove all questions, "yes/no" responses, and filler words.
4. **Produce a single, well-written paragraph**: The output should be a clean, objective description, as if written by a professional cataloger.

---
**Dialogue Log to Refine:**
{dialogue_log}
---

**Refined Description:**"""
        
        return self._generate_with_prompt(user_message, max_tokens, temperature)
    
    def _format_dialogue_log_for_refine(self, video_captions: str) -> str:
        """
        Format the dialogue log for refinement processing.
        
        Args:
            video_captions: Initial video captions
            
        Returns:
            Formatted dialogue log string
        """
        # Start with initial description
        dialogue_log = f"Initial Description: {video_captions}\n"
        
        # Add conversation history if available
        if self.conversation_log:
            for i, conv in enumerate(self.conversation_log):
                round_num = i + 1
                question = conv.get("question", "")
                answer = conv.get("answer", "")
                
                if question and answer:
                    dialogue_log += f"[Round {round_num}] Question: {question}\n"
                    dialogue_log += f"[Round {round_num}] Answer: {answer}\n"
        
        return dialogue_log.strip()
    
    def _generate_basic_question(self, video_captions: str, max_tokens: int, temperature: Optional[float]) -> str:
        """
        Generate a basic question (fallback method).
        
        Args:
            video_captions: Captions describing the video content
            max_tokens: Maximum number of tokens for the response
            temperature: Temperature parameter for generation
            
        Returns:
            Generated question string
        """
        user_message = f"""
        This is caption of retrieved video. Read the video captions and ask some question to gain more information to help find out exact video.
        Some video may not have caption due to API error saying sorry I can't provide blah blah.
        Captions for video: {video_captions}

        Question: 
        """
        
        return self._generate_with_prompt(user_message, max_tokens, temperature)
    
    def _generate_with_prompt(self, user_message: str, max_tokens: int, temperature: Optional[float]) -> str:
        """
        Generate response using the provided prompt.
        
        Args:
            user_message: The user message/prompt
            max_tokens: Maximum number of tokens for the response
            temperature: Temperature parameter for generation
            
        Returns:
            Generated response string
        """
        # Add the user message to conversation
        self.add_to_conversation("user", user_message)
        
        try:
            # Process inputs
            text = self.processor.apply_chat_template(
                self.messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature or self.default_temperature
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "Could not generate a response due to an error."
    
    def record_answer(self, answer: str, reranked_caption: str, target_rank: Optional[int] = None, reranked_topk: Optional[List[str]] = None, reformatted_description: Optional[str] = None):
        """
        Record an answer to the most recent question in the conversation log.
        
        Args:
            answer: The answer to the question
            reranked_caption: Caption of the reranked top video
            target_rank: The rank of the target video after reranking (optional)
            reranked_topk: List of top-k reranked video IDs (optional)
            reformatted_description: The reformatted description after this Q&A pair (optional)
        """
        if self.conversation_log:
            # Update the most recent entry in the conversation log
            self.conversation_log[-1]["answer"] = answer
            self.conversation_log[-1]["reranked_caption"] = reranked_caption
            self.conversation_log[-1]["answer_timestamp"] = time.time()
            
            if target_rank is not None:
                self.conversation_log[-1]["target_rank"] = target_rank
            
            if reranked_topk is not None:
                self.conversation_log[-1]["reranked_topk"] = reranked_topk
            
            if reformatted_description is not None:
                self.conversation_log[-1]["reformatted_description"] = reformatted_description
            
            # Add the answer to the conversation history
            formatted_answer = self.format_answer_prompt(answer, reranked_caption)
            self.add_to_conversation("user", formatted_answer)
            
            self.logger.debug(f"Recorded answer, target rank: {target_rank}, reformatted description: {reformatted_description}")
        else:
            self.logger.warning("Attempted to record answer but no questions exist in conversation log")
    
    def format_answer_prompt(self, answer: str, reranked_caption: str) -> str:
        """
        Format the prompt for the next question based on the answer and reranked video caption.
        
        Args:
            answer: The answer to the previous question
            reranked_caption: Caption of the reranked top video
            
        Returns:
            Formatted prompt string
        """
        return f"""answer: {answer}
        Based on your answer, here's caption of reranked video.
        caption: {reranked_caption}
        Keep asking.
        Question: 
        """
    
    def export_conversation_log(self) -> Dict[str, Any]:
        """
        Export the full conversation log in a structured format.
        
        Returns:
            Dictionary containing the full conversation history and metadata
        """
        return {
            "target_video_id": self.target_video_id,
            "conversations": self.conversation_log,
            "total_conversations": len(self.conversation_log),
            "system_prompt": self.system_prompt["content"],
            "timestamp": time.time(),
            "entropy_analysis": {
                "n_clusters": self.n_clusters,
                "alpha_threshold": self.alpha_threshold,
                "beta_threshold": self.beta_threshold,
                "asked_topics": self.asked_topics.copy() if self.asked_topics else [],
                "current_embeddings_shape": self.current_embeddings.shape if self.current_embeddings is not None else None,
                "cluster_labels": self.cluster_labels.tolist() if self.cluster_labels is not None else None,
                "cluster_centers_shape": self.cluster_centers.shape if self.cluster_centers is not None else None
            }
        } 