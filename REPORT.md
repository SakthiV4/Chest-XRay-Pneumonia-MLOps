# Project Report: End-to-End AI Application for Chest X-Ray Pneumonia Detection

## 1. Abstract
This report details the development and deployment of **PneumoScan AI**, a full-stack, end-to-end artificial intelligence application designed to detect pneumonia from PA (Posteroanterior) chest radiographs. Built as a software paper project, the system demonstrates the practical integration of deep learning (ResNet-50) with modern scalable deployment architectures, including NVIDIA Triton Inference Server, FastAPI, and a React frontend. The model achieved a notable **93.11% accuracy** and an **AUC-ROC of 0.973** on the widely recognized Kaggle Chest X-Ray dataset. Additionally, the system features LLM-assisted diagnostic reporting, powered by Groq's fast Llama3 inference, to bridge the gap between raw classification and clinical utility.

## 2. Problem Statement
Pneumonia remains a leading cause of mortality worldwide, particularly among children and the elderly. While chest X-rays (CXR) are the standard modality for diagnosing pneumonia, interpreting them requires expert radiologists, who are often in short supply in resource-constrained environments. AI-driven automated analysis can serve as a vital triage tool, reducing diagnostic delay. 

However, many AI models in medical imaging remain confined to Jupyter notebooks. The challenge addressed by this project is bridging the "deployment gap"—taking a raw PyTorch model and building a robust, full-stack application capable of serving predictions with low latency via hardware-accelerated inference, wrapped in a user-friendly clinical interface.

## 3. System Architecture
The application leverages a microservices-based architecture, orchestrated via Docker Compose:

1. **Frontend (React + Vite)**: A responsive, dark-mode medical UI allowing users to drag-and-drop X-ray images. It provides immediate visual feedback, including confidence bars and micro-animations, communicating with the backend via REST API.
2. **Backend API (FastAPI)**: A high-performance Python backend responsible for routing, image preprocessing (resizing, normalization), orchestrating inference, and interacting with the Anthropic API for LLM report generation.
3. **Inference Engine (NVIDIA Triton / ONNXRuntime)**: The core AI serving layer. The PyTorch model is exported to ONNX format to leverage optimizations like dynamic batching and GPU acceleration, ensuring high-throughput inference (~60ms per request on an RTX 4060).
4. **LLM Diagnostic Assistant (Groq / Llama3)**: An integrated Large Language Model that takes the classification label and confidence score to generate a structured, radiologist-style assistant note.

## 4. Methodology & Training
### 4.1 Dataset
The project utilizes the Kaggle **Chest X-Ray Images (Pneumonia)** dataset, containing 5,856 validated X-ray images. The data exhibits significant class imbalance (~3:1 Pneumonia to Normal).

### 4.2 Model Architecture & Fine-Tuning (Phase 2)
A **ResNet-50** architecture, pretrained on ImageNet, was selected for its strong feature extraction capabilities. The training was conducted in two phases:
1. **Initial Fine-Tuning**: The backbone was frozen, and a custom multi-layer perceptron (MLP) classification head (2048 → 256 → Dropout(0.4) → 1) was trained for 10 epochs. This achieved a baseline accuracy of 87.5%.
2. **Phase 2 Polish**: The model was further refined by unfreezing the deeper convolutional layers (`layer3` and `layer4`). To prevent catastrophic forgetting, **Differential Learning Rates** were employed (Backbone: $1 \times 10^{-5}$, Head: $1 \times 10^{-4}$), alongside a Cosine Annealing LR scheduler. Furthermore, strict class weighting was applied to the Binary Cross-Entropy loss to counteract the dataset imbalance.

### 4.3 Data Augmentation
To prevent overfitting, robust dynamic data augmentation was applied during training using PyTorch's `transforms`, including:
- Random Cropping to $224 \times 224$ (from $256 \times 256$ resized images)
- Random Horizontal Flipping (p=0.5)
- Color Jitter (Brightness, Contrast, Saturation adjustments)
- Random Affine Transformations (Rotation up to 10°, slight translations)

## 5. Results & Evaluation
Following the Phase 2 Polish, the model was evaluated on the strict unseen test split (624 images). The AI achieved exceptional discriminative ability, particularly prioritizing the reduction of false negatives (missing a pneumonia diagnosis).

| Metric | Performance |
| :--- | :--- |
| **Accuracy** | 93.11% |
| **AUC-ROC** | 0.9732 |
| **Sensitivity (Recall - Pneumonia)** | 98.50% |
| **Specificity (Recall - Normal)** | 84.20% |
| **F1-Score** | ~0.94 |

The sensitivity of **98.5%** ensures the model acts as an excellent screening tool, missing extremely few actual positive pneumonia cases.

## 6. Deployment & LLM Integration (Phase 3-5)
To transition from a trained `.pth` file to a deployable artifact:
1. **ONNX Export**: The PyTorch model was exported to ONNX format (`opset_version=17`) with dynamic batch axes enabled, creating a highly portable inference graph.
2. **Triton Integration**: The ONNX model was structured into a Triton model repository (`config.pbtxt`), configuring GPU execution (`KIND_GPU`) and dynamic batching (`max_queue_delay_microseconds: 50000`).
3. **LLM Generation**: Instead of simply returning a binary "1 or 0", the FastAPI backend queries the Groq API (using the high-speed Llama3 model) with a dynamic prompt. The LLM translates the prediction into a plain-English, 3-4 sentence clinical note, explicitly noting its AI-assisted nature to encourage responsible "human-in-the-loop" usage.

## 7. Conclusion
**PneumoScan AI** successfully demonstrates the complete lifecycle of a modern medical AI application. By combining transfer learning, rigorous evaluation, hardware-accelerated serving via NVIDIA Triton, and Generative AI for interpretability, this project bridges the gap between machine learning research and practical software engineering. The resulting system is scalable, clinically aligned (prioritizing high sensitivity), and ready for further peer-review or clinical validation.
