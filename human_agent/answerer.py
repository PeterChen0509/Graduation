import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class Answerer(torch.nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        super(Answerer, self).__init__()
        # Load VQA model-> Qwen 2.5 VL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Qwen 2.5 VL model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            use_fast=True  # 使用快速处理器避免警告
        )
        
        self.video_path = None

    def load_video(self, video_path):
        """Load video path for processing"""
        self.video_path = video_path

    def ask(self, question):
        """Ask question about the video using Qwen 2.5 VL"""
        print("Asking..")
        
        if not self.video_path:
            raise ValueError("No video loaded. Call load_video() first.")
        
        # Prepare messages for Qwen 2.5 VL with video
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": f"{self.video_path}",
                        "fps": 1.0,  # Extract 1 frame per second
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # Process inputs
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        # Remove fps from video_kwargs if it exists to avoid duplicate argument
        if 'fps' in video_kwargs:
            del video_kwargs['fps']
            
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=1.0,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.device)

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=100,
                do_sample=True,
                temperature=0.3
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # For compatibility, return answer as both aggregated and individual
        return answer, [answer]
    
    async def async_ask(self, question):
        """Async version of ask - for compatibility, but runs synchronously for local model"""
        # For local models, async doesn't provide performance benefits
        # We run synchronously but maintain async interface for compatibility
        answer, _ = self.ask(question)
        return answer, None  # 返回 None 作为 before_aggr，因为我们不再需要按帧处理