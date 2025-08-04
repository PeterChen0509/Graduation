import av
import torch
import numpy as np
import os
import time
import pandas as pd
from tqdm import tqdm
from transformers import BitsAndBytesConfig,LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import traceback

pd.set_option("display.max_colwidth", None)

def load_video_model():
    """加载模型，使用自动设备映射让模型分布在所有可用GPU上"""
    model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    
    # 创建卸载目录
    offload_folder = "/tmp/offload_folder"
    os.makedirs(offload_folder, exist_ok=True)
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True  # 允许CPU卸载
    )
    
    # 自定义device_map，将某些层分配到CPU
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="balanced",  # 使用balanced而不是auto
        offload_folder=offload_folder,
        low_cpu_mem_usage=True,
    )
    processor = LlavaNextVideoProcessor.from_pretrained(model_id, use_fast=True)
    return model, processor

def read_video_pyav(video_path, num_frames=4):
    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.arange(0, total_frames, total_frames / num_frames).astype(int)

        frames = []
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > indices[-1]:
                break
            if i >= indices[0] and i in indices:
                frames.append(frame)

        if frames:
            return np.stack([x.to_ndarray(format="rgb24") for x in frames])
        return None
    except Exception as e:
        print(f"[{video_path}] 解码失败: {e}")
        return None

def generate_video_caption(video_path, model, processor):
    start_time = time.time()
    # 使用提示
    detailed_prompt = """
        You are a professional video captioner for an interactive retrieval system. Watch the video clip and generate one long, highly detailed sentence that captures as many observable visual details as possible.

        Include all the following:
        - The subject's gender, visible identity or appearance cues (e.g., man, woman, girl).
        - Their physical actions over time (e.g., hand gestures, body posture, head movements).
        - Facial expressions, including subtle elements and their **changes over time** if visible.
        - Visible gaze direction or engagement with other people or objects.
        - Any interaction with the **scene or background** that is visually noticeable.
        - The general environment or setting if it provides meaningful context (e.g., "in a bathtub", "at a table", "in front of a crowd").
        - Temporal or rhythm-related information (e.g., rapidly, slowly, repeatedly).
        - State changes, such as changing position or expression.

        Output format:
        Write **one long descriptive sentence** that includes all of this information in natural language. Use specific visual terms and be rich in detail. Do not infer emotions or motivations. Describe only what is observable.

        Example:
        A man stands in a dimly lit bathroom, raising his right arm and then lowering it slowly while keeping his gaze upward. His eyebrows gradually press together, his eyes remain half-open, and his mouth opens wide and then closes, with his neck slightly tilted back and shoulders raised as if in tension.
    """
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": detailed_prompt},
                    {"type": "video"},
                ],
            },
        ]
        prompt_text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # 读取视频帧
        clip = read_video_pyav(video_path)
        if clip is None:
            return f"读取失败: {video_path}", 0
        # 处理视频帧
        inputs_video = processor(text=prompt_text, videos=clip, padding=True, return_tensors="pt").to(model.device)
        # 生成标注
        generate_kwargs = {
            "max_new_tokens": 150,
            "do_sample": False,
            "return_dict_in_generate": True,
            "output_scores": False
        }
        with torch.no_grad():
            output = model.generate(**inputs_video, **generate_kwargs)
        
        # 解码输出
        generated_tokens = output.sequences[:, inputs_video["input_ids"].shape[1]:]
        caption = processor.decode(generated_tokens[0], skip_special_tokens=True)
        processing_time = time.time() - start_time
        return caption, processing_time
    except Exception as e:
        print(f"生成标注时出错: {e}")
        traceback.print_exc()
        return f"生成失败: {str(e)}", time.time() - start_time

def main():
    # 路径设置
    excel_path = "/home/peterchen/M2/MER2024/description.csv"
    video_dir = "/home/peterchen/M2/MER2024/video-selected"
    output_excel_path = "/home/peterchen/M2/MER2024/llava_next_video_caption.xlsx"

    # 读取数据
    print(f"读取 Excel: {excel_path}")
    df = pd.read_csv(excel_path)
    
    # 添加视频路径和新列
    df['video_path'] = df['name'].apply(lambda x: os.path.join(video_dir, x+".mp4"))
    df['model_caption'] = ""  # 使用空字符串而不是np.nan，确保与原始代码行为一致
    df['processing_time'] = 0.0
    
    # 加载模型（只加载一次，使用所有可用GPU）
    print("加载LLaVA-NeXT-Video模型...")
    model, processor = load_video_model()
    
    # 处理每个视频
    print(f"开始处理 {len(df)} 个视频...")
    for idx in tqdm(range(len(df))):
        video_name = df.loc[idx, 'name']
        video_path = df.loc[idx, 'video_path']
        
        print(f"正在处理第 {idx+1}/{len(df)} 个视频: {video_name}")
        caption, proc_time = generate_video_caption(video_path, model, processor)
        
        df.loc[idx, 'model_caption'] = caption
        df.loc[idx, 'processing_time'] = proc_time
        
        print(f"完成: {video_name}，耗时: {proc_time:.2f}秒")
        
        # 每处理10个视频保存一次结果文件，避免中途中断导致全部丢失
        if (idx + 1) % 10 == 0:
            df.to_excel(output_excel_path, index=False)
            print(f"已处理 {idx+1}/{len(df)} 个视频，已保存当前结果")
    
    # 保存最终结果
    df.to_excel(output_excel_path, index=False)
    print(f"\n所有视频处理完成，结果已保存至：{output_excel_path}")

if __name__ == '__main__':
    main()      