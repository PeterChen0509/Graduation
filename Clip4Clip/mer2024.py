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

VIDEO_BASE_PATH = '/home/peterchen/M2/MER2024/video-selected'
LABEL_FILE_PATH = '/home/peterchen/M2/MER2024/llava_next_video_caption.xlsx'
MODEL_NAME = "Searchium-ai/clip4clip-webvid150k"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的设备 (Using device): {DEVICE}")

# --- 2. 视频与文本编码函数 (Video and Text Encoding Functions) ---
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
def compute_metrics(sim_matrix):
    nn_idx = np.argsort(-sim_matrix, axis=1)
    y = np.eye(nn_idx.shape[0])
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

    with open("/home/peterchen/M2/Clip4Clip/metrics_mer2024_results.txt", "w", encoding="utf-8") as f:
        f.write('--- 评估结果 (Evaluation Results) ---\n')
        f.write(f'相似度矩阵形状 (Similarity-matrix shape): {nn_idx.shape}\n')
        f.write(result_str + "\n")

    return metrics

# --- 4. 主执行流程 (Main Execution Workflow) ---
def main():
    print("正在加载模型 (Loading models)...")
    video_model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(DEVICE)
    text_model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    video_model.eval()
    text_model.eval()
    print("模型加载完成 (Models loaded).")

    print(f"正在读取标签文件 (Reading label file): {LABEL_FILE_PATH}")
    df = pd.read_excel(LABEL_FILE_PATH)
    df.dropna(subset=['name', 'eng_caption'], inplace=True)
    # 拼接.mp4后缀
    video_names = [str(name) + '.mp4' for name in df['name'].tolist()]
    captions = df['eng_caption'].tolist()
    print(f"找到了 {len(video_names)} 个有效的视频-文本对 (Found {len(video_names)} valid video-text pairs).")

    all_video_embeds = []
    all_text_embeds = []
    print("开始提取特征 (Starting feature extraction)...")
    with torch.no_grad():
        for video_name in tqdm(video_names, desc="处理视频中 (Processing Videos)"):
            video_path = os.path.join(VIDEO_BASE_PATH, video_name)
            video_frames = video2image(video_path)
            if video_frames is None:
                all_video_embeds.append(torch.zeros(512).to(DEVICE))
                continue
            video_frames = video_frames.to(DEVICE)
            visual_output = video_model(video_frames)
            visual_embeds = visual_output["image_embeds"]
            visual_embeds = visual_embeds / visual_embeds.norm(dim=-1, keepdim=True)
            visual_embeds = torch.mean(visual_embeds, dim=0)
            visual_embeds = visual_embeds / visual_embeds.norm(dim=-1, keepdim=True)
            all_video_embeds.append(visual_embeds)
        for caption in tqdm(captions, desc="处理文本中 (Processing Texts)"):
            inputs = tokenizer(text=caption, return_tensors="pt").to(DEVICE)
            text_output = text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            text_embed = text_output[0] / text_output[0].norm(dim=-1, keepdim=True)
            all_text_embeds.append(text_embed.squeeze(0))
    print("特征提取完成，开始计算评估指标 (Feature extraction complete. Calculating metrics)...")
    video_embeddings_tensor = torch.stack(all_video_embeds).cpu().numpy()
    text_embeddings_tensor = torch.stack(all_text_embeds).cpu().numpy()
    similarity_matrix = np.matmul(text_embeddings_tensor, video_embeddings_tensor.T)
    compute_metrics(similarity_matrix)

if __name__ == '__main__':
    main()
