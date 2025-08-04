import os
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from PIL import Image
import cv2
from tqdm import tqdm
from transformers import CLIPVisionModelWithProjection, CLIPTokenizer, CLIPTextModelWithProjection

# --- 1. 设置与参数定义 (Setup and Parameters) ---

# 请根据您的实际环境修改这些路径
VIDEO_BASE_PATH = '/home/peterchen/M2/ADEPT/data/mafw/videos'
LABEL_FILE_PATH = '/home/peterchen/M2/ADEPT/data/mafw/labels/sampled_850.xlsx'
MODEL_NAME = "Searchium-ai/clip4clip-webvid150k"

# 检查是否有可用的GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的设备 (Using device): {DEVICE}")

# --- 2. 视频与文本编码函数 (Video and Text Encoding Functions) ---

# 视频编码函数 (与您提供的代码一致)
def video2image(video_path, frame_rate=1.0, size=224):
    def preprocess(size, n_px):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(n_px)

    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在 (ERROR: Video file not found): {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    images = []
    if fps > 0:
        total_duration = (frameCount + fps - 1) // fps
        interval = fps / frame_rate
        frames_idx = np.floor(np.arange(0, total_duration * fps, interval))
        
        for idx in frames_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(preprocess(size, Image.fromarray(frame).convert("RGB")))
    else:
        print(f"警告: 无法读取视频FPS (WARNING: Could not read FPS for video): {video_path}")

    cap.release()

    if not images:
        print(f"错误: 无法从视频中提取帧 (ERROR: Could not extract frames from video): {video_path}")
        return None
        
    video_frames = torch.stack(images)
    return video_frames

# --- 3. 评估指标计算函数 (Evaluation Metrics Function) ---

# 指标计算函数 (与教程中的代码一致)
def compute_metrics(sim_matrix):
    # sim_matrix 的形状应为 (N_texts, N_videos)
    # 假设文本和视频是一一对应的, 所以理想情况下对角线元素值最大
    nn_idx = np.argsort(-sim_matrix, axis=1)
    
    # 创建一个单位矩阵作为理想的排序结果
    y = np.eye(nn_idx.shape[0])
    
    # 找到每个文本的正确视频匹配项在排序列表中的位置
    ind = np.where(np.take_along_axis(y, nn_idx, axis=1) == 1)[1]
    
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1

    print('--- 评估结果 (Evaluation Results) ---')
    print(f'相似度矩阵形状 (Similarity-matrix shape): {nn_idx.shape}')
    result_str = (
        f"文本->视频检索 (Text-to-Video): "
        f"R@1: {metrics['R1']:.2f}% - "
        f"R@5: {metrics['R5']:.2f}% - "
        f"R@10: {metrics['R10']:.2f}% - "
        f"Median R: {metrics['MR']:.2f} - "
        f"Mean R: {metrics['MeanR']:.2f}"
    )
    print(result_str)

    # 保存到txt文件
    with open("/home/peterchen/M2/Clip4Clip/metrics_mafw_results.txt", "w", encoding="utf-8") as f:
        f.write('--- 评估结果 (Evaluation Results) ---\n')
        f.write(f'相似度矩阵形状 (Similarity-matrix shape): {nn_idx.shape}\n')
        f.write(result_str + "\n")

    return metrics

# --- 4. 主执行流程 (Main Execution Workflow) ---

def main():
    # --- 步骤 1: 加载模型 ---
    print("正在加载模型 (Loading models)...")
    video_model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(DEVICE)
    text_model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    video_model.eval()
    text_model.eval()
    print("模型加载完成 (Models loaded).")

    # --- 步骤 2: 加载数据标签 ---
    print(f"正在读取标签文件 (Reading label file): {LABEL_FILE_PATH}")
    df = pd.read_excel(LABEL_FILE_PATH)
    # 为确保一一对应, 移除缺少视频名或标题的行
    df.dropna(subset=['video_name', 'eng_caption'], inplace=True)
    video_names = df['video_name'].tolist()
    captions = df['eng_caption'].tolist()
    print(f"找到了 {len(video_names)} 个有效的视频-文本对 (Found {len(video_names)} valid video-text pairs).")

    # --- 步骤 3: 提取所有视频和文本的特征 ---
    all_video_embeds = []
    all_text_embeds = []
    
    print("开始提取特征 (Starting feature extraction)...")
    with torch.no_grad():
        # 提取视频特征
        for video_name in tqdm(video_names, desc="处理视频中 (Processing Videos)"):
            video_path = os.path.join(VIDEO_BASE_PATH, video_name)
            video_frames = video2image(video_path)
            
            if video_frames is None:
                # 如果视频处理失败, 添加一个零向量作为占位符
                # 在后续分析中可以考虑如何处理这些失败案例
                all_video_embeds.append(torch.zeros(512).to(DEVICE))
                continue
            
            video_frames = video_frames.to(DEVICE)
            visual_output = video_model(video_frames)
            
            # 归一化并取均值
            visual_embeds = visual_output["image_embeds"]
            visual_embeds = visual_embeds / visual_embeds.norm(dim=-1, keepdim=True)
            visual_embeds = torch.mean(visual_embeds, dim=0)
            visual_embeds = visual_embeds / visual_embeds.norm(dim=-1, keepdim=True)
            
            all_video_embeds.append(visual_embeds)
            
        # 提取文本特征
        for caption in tqdm(captions, desc="处理文本中 (Processing Texts)"):
            inputs = tokenizer(text=caption, return_tensors="pt").to(DEVICE)
            text_output = text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            
            # 归一化
            text_embed = text_output[0] / text_output[0].norm(dim=-1, keepdim=True)
            all_text_embeds.append(text_embed.squeeze(0))

    # --- 步骤 4: 计算相似度并评估 ---
    print("特征提取完成，开始计算评估指标 (Feature extraction complete. Calculating metrics)...")
    
    # 将列表转换为张量
    video_embeddings_tensor = torch.stack(all_video_embeds).cpu().numpy()
    text_embeddings_tensor = torch.stack(all_text_embeds).cpu().numpy()

    # 计算相似度矩阵 (文本 x 视频)
    similarity_matrix = np.matmul(text_embeddings_tensor, video_embeddings_tensor.T)

    # 计算并打印评估指标
    compute_metrics(similarity_matrix)

if __name__ == '__main__':
    main()