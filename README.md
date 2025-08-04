# Graduation Project: Multimodal Video Retrieval and Emotion Recognition

This repository contains my Master's thesis research at Tokyo Institute of Technology (Tokyo Tech), focusing on multimodal video retrieval and emotion recognition. It includes a comprehensive collection of multimodal video retrieval and emotion recognition models, along with their evaluation results and comparative analysis.

## 📁 Project Structure

```
Graduation/
├── ADEPT/                    # Our proposed model - Adaptive Dialogue-Enhanced Parameter Tuning
├── MERLIN/                   # MERLIN baseline - Multimodal Embedding Refinement via LLM-based Iterative Navigation
├── Clip4Clip/               # Clip4Clip baseline for video-text retrieval
├── Emotion-LLaMA/           # Emotion-LLaMA for multimodal emotion recognition
├── IVR-QA-baselines/        # Interactive Video Retrieval with Questions and Answers baseline
├── llava_next_video_deepseek/ # LLaVA-NeXT video understanding model
└── output_analysis/         # Comprehensive analysis of all model outputs
```

## 🚀 Models Overview

### 1. ADEPT (Our Proposed Model)
**Paper**: [Adaptive Dialogue-Enhanced Parameter Tuning for Multimodal Video Retrieval](https://arxiv.org/abs/2407.12508)

**Description**: Our proposed model that extends MERLIN with adaptive parameter tuning mechanisms. It introduces entropy analysis strategies to automatically select ASK vs REFINE decisions for affective video retrieval tasks.

**Key Features**:
- Adaptive parameter tuning for optimal (m, α, β) combination
- Entropy-based strategy selection (ASK vs REFINE)
- Specialized optimization for affective video datasets (MAFW, MER2024)
- Two-phase approach: parameter tuning and best parameter testing

**Best Parameters**:
- MAFW: m=12, α=0.0075, β=0.062
- MER2024: m=4, α=0.006, β=0.007

### 2. MERLIN (Baseline)
**Paper**: [Multimodal Embedding Refinement via LLM-based Iterative Navigation](https://arxiv.org/abs/2407.12508)

**Description**: The original MERLIN framework that ADEPT is based on. It uses LLM-based iterative navigation for text-video retrieval-rerank pipeline.

**Components**:
- Questioner: Generates questions about video content
- Reranker: Reorders candidate videos using Google Vertex AI
- Answerer: Simulates human agent interactions

### 3. Clip4Clip (Baseline)
**Paper**: [CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval](https://arxiv.org/abs/2104.08860)

**Description**: A baseline model for video-text retrieval using CLIP architecture. It extends CLIP to handle video sequences by processing multiple frames.

**Implementation**: 
- `mafw.py`: MAFW dataset evaluation
- `mer2024.py`: MER2024 dataset evaluation
- Results stored in `metrics_*.txt` files

### 4. Emotion-LLaMA (Baseline)
**Paper**: [Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning](https://arxiv.org/abs/2406.11161)

**Description**: A multimodal emotion recognition model that integrates audio, visual, and textual inputs through emotion-specific encoders.

**Key Features**:
- MERR dataset with 28,618 coarse-grained and 4,487 fine-grained samples
- Emotion-specific encoders for audio, visual, and textual inputs
- Instruction tuning with modified LLaMA model
- Top performance on EMER, MER2023, and DFEW datasets

**Performance**:
- Clue Overlap: 7.83, Label Overlap: 6.25 on EMER
- F1 score: 0.9036 on MER2023 challenge
- UAR: 45.59, WAR: 59.37 on DFEW dataset

### 5. IVR-QA-baselines (Baseline)
**Paper**: [Simple Baselines for Interactive Video Retrieval with Questions and Answers](https://arxiv.org/abs/2308.10402)

**Description**: ICCV'2023 paper presenting simple yet effective baselines for interactive video retrieval via question-answering.

**Key Features**:
- Interactive video retrieval through question-answering
- VideoQA model to simulate user interactions
- Simple but effective baselines for interactive retrieval
- Evaluated on MSR-VTT, MSVD, and AVSD datasets

**Methods**:
- Heuristic approach
- Auto-text generation
- Auto-text-video combination

### 6. LLaVA-NeXT Video (Baseline)
**Repository**: [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)

**Description**: Advanced video understanding model based on LLaVA-NeXT architecture for multimodal video analysis.

**Implementation**:
- `deepseek_interaction.py`: DeepSeek-based video interaction
- `ichise_interaction.py`: Ichise-based video interaction
- `llava_next_video_caption.py`: Video caption generation

## 📊 Output Analysis

The `output_analysis/` directory contains comprehensive evaluation results and comparative analysis:

### Structure
```
output_analysis/
├── analysis.ipynb              # Main analysis notebook
├── MAFW/                       # MAFW dataset results
├── MER2024/                    # MER2024 dataset results
└── entropy_analysis_outputs/   # Entropy analysis results
```

### Key Metrics Analyzed
- **Recall@1/5/10**: Retrieval accuracy at different ranks
- **Entropy Analysis**: Inter-cluster and intra-cluster entropy
- **Parameter Optimization**: Best parameter combinations for each model
- **Comparative Performance**: Cross-model performance comparison

## 🎯 Datasets

### MAFW (Multi-modal Affective Video Dataset)
- **Type**: Affective video dataset
- **Focus**: Emotional video content analysis
- **Usage**: Primary dataset for affective video retrieval evaluation

### MER2024 (Multimodal Emotion Recognition 2024)
- **Type**: Multimodal emotion recognition dataset
- **Focus**: Emotion recognition across multiple modalities
- **Usage**: Secondary dataset for comprehensive evaluation

## 🛠️ Setup and Usage

### Environment Requirements
- Python 3.8+
- PyTorch
- Transformers
- Google Cloud Vertex AI API
- OpenAI API

### Quick Start
1. **ADEPT Model**:
   ```bash
   cd ADEPT
   python run_adept.py --dataset mafw --data_path /path/to/data --num_rounds 5
   ```

2. **Parameter Tuning**:
   ```bash
   cd ADEPT/tuning
   python parameter_tuning.py --dataset mafw --data_path /path/to/data
   ```

3. **Baseline Models**:
   - Clip4Clip: Run `mafw.py` or `mer2024.py`
   - Emotion-LLaMA: Follow the original repository setup
   - IVR-QA: Use `eval_interactive.py` with appropriate configs
   - LLaVA-NeXT: Use the provided interaction scripts

## 📈 Performance Comparison

### ADEPT vs Baselines
- **ADEPT**: Adaptive parameter tuning with entropy analysis
- **MERLIN**: Original framework without parameter optimization
- **Clip4Clip**: Standard video-text retrieval baseline
- **Emotion-LLaMA**: Specialized for emotion recognition
- **IVR-QA**: Interactive retrieval with question-answering
- **LLaVA-NeXT**: Advanced multimodal video understanding

### Key Findings
- ADEPT shows improved performance through adaptive parameter tuning
- Entropy analysis helps in better strategy selection
- Affective video datasets benefit from specialized optimization
- Interactive approaches generally outperform single-shot retrieval

## 🔬 Research Contributions

1. **Adaptive Parameter Tuning**: Novel approach to automatically find optimal parameters for different datasets
2. **Entropy Analysis**: Systematic analysis of cluster entropy for strategy selection
3. **Affective Video Retrieval**: Specialized optimization for emotional content
4. **Comprehensive Evaluation**: Multi-model comparison on affective video datasets

## 📚 Citation

If you use this code or find it helpful, please cite the relevant papers:

```bibtex
@article{merlin2024,
  title={Multimodal Embedding Refinement via LLM-based Iterative Navigation},
  author={Original Authors},
  journal={arXiv preprint},
  year={2024}
}

@inproceedings{clip4clip2021,
  title={CLIP4Clip: An Empirical Study of CLIP for End to End Video Clip Retrieval},
  author={Original Authors},
  booktitle={arXiv preprint},
  year={2021}
}

@article{emotionllama2024,
  title={Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning},
  author={Original Authors},
  journal={arXiv preprint},
  year={2024}
}

@inproceedings{ivrqa2023,
  title={Simple Baselines for Interactive Video Retrieval with Questions and Answers},
  author={Liang, Kaiqu and Albanie, Samuel},
  booktitle={ICCV},
  year={2023}
}
``` 