# Hierarchical Federated Learning and XAI Framework for Oral Cancer Detection

This repository contains a Python implementation of a hierarchical federated learning (FL) and explainable AI (XAI) framework for detecting oral squamous cell carcinoma (SCC) using histopathological images, as described in the research paper. The code simulates a three-tier architecture (clients, edge servers, global server) and integrates transfer learning with pre-trained CNN models and XAI techniques.

## Features
- **Models:** Implements DenseNet121, EfficientNetB4, InceptionV3, and VGG19 for binary classification.
- **Federated Learning:** Hierarchical FL with client training, edge aggregation, and global synchronization.
- **XAI:** Visualizes Grad-CAM and Guided Backprop heatmaps for interpretability.
- **Data Handling:** Supports data augmentation and non-IID partitioning.

## Requirements
- Python 3.8+
- Libraries: `torch`, `torchvision`, `flower`, `numpy`, `matplotlib`, `pytorch_grad_cam`, `lime`
- Install via: `pip install torch torchvision flower numpy matplotlib pytorch-grad-cam lime`

## Installation
1. Clone the repository: `git clone <repository_url>`
2. Navigate to the directory: `cd <repository_name>`
3. Install dependencies: `pip install -r requirements.txt` (create `requirements.txt` with listed libraries)
4. Download CIFAR-10 dataset (proxy) or replace with Kaggle histopathological images.

## Usage
1. Run the main script: `python main.py`
2. The code will:
   - Train the FL model for 10 rounds.
   - Display global accuracy per round.
   - Generate and show XAI heatmaps.

## File Structure
- `main.py`: Orchestrates the FL and XAI execution.
- `models.py`: Defines CNN models.
- `data_utils.py`: Loads and partitions the dataset.
- `fl_client.py`: Implements client-side training.
- `fl_edge.py`: Handles edge server aggregation.
- `fl_server.py`: Manages global server logic.
- `xai_utils.py`: Provides XAI visualization.

## Customization
- **Dataset:** Replace CIFAR-10 with your 10,000-image Kaggle dataset in `data_utils.py` by updating `CustomDataset`.
- **Hyperparameters:** Adjust epochs, learning rate, and batch size in `fl_client.py` and `fl_server.py`.
- **Models:** Fine-tune pre-trained models in `models.py` for better performance (e.g., 98.6% local accuracy).

## Results
- Simulated accuracy ranges from 40-50% with CIFAR-10. With your dataset and fine-tuning, expect ~97-98% local accuracy (EfficientNetB4) and ~97.1% global accuracy, as per the paper.
- XAI heatmaps highlight features like nuclear atypia.

## Contributing
Fork the repository, make changes, and submit a pull request. Ensure code aligns with the methodology.

## License
MIT License (specify if different).

## Last Updated
10:34 PM PKT, Friday, August 08, 2025
