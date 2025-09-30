Plant Leaf Disease Detection

A deep learning project for identifying plant diseases from leaf images using transfer learning and custom CNN architectures.

 Live Demo

**Try it now:** [Plant Disease Detector on Hugging Face](https://huggingface.co/spaces/yenushka/plant-disease-detector)

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/yenushka/plant-disease-detector)

 Overview

This project implements two neural network models to classify plant leaf diseases:
- **Model A**: MobileNetV2-based transfer learning model
- **Model B**: Ultra-lightweight custom CNN

The models can distinguish between healthy and diseased leaves across multiple plant species.

 Features

- Transfer learning using MobileNetV2 with ImageNet weights
- Custom lightweight CNN for resource-constrained environments
- Data augmentation for improved generalization
- Automatic model checkpointing and early stopping
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- Simple console-based prediction interface

Project Structure

```
plant-disease-detection/
├── PlantLeafDiseaseGradio1.ipynb    # Main Jupyter notebook
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── models/                           # Saved model files (created during training)
```

 Requirements

- Python 3.7+
- TensorFlow 2.x
- Google Colab (recommended) or local environment with GPU support

See `requirements.txt` for complete dependencies.

 Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Google Colab usage (recommended):
   - Upload the notebook to Google Colab
   - Mount your Google Drive
   - Upload dataset to Google Drive at `/MyDrive/DataSetPlantDiseases.zip`

 Dataset Structure

The dataset should be organized as follows:

```
DataSetPlantDiseases/
├── train/
│   ├── Apple___Black_rot/
│   ├── Apple___Healthy/
│   ├── Corn___Common_rust/
│   └── ...
├── valid/
│   ├── Apple___Black_rot/
│   └── ...
└── test/
    ├── Apple___Black_rot/
    └── ...
```
 Usage

### Option 1: Use the Live Demo (Recommended)

Visit the [Hugging Face Space](https://huggingface.co/spaces/yenushka/plant-disease-detector) to:
- Upload leaf images directly through the web interface
- Get instant disease predictions
- No installation or setup required

### Option 2: Train Models Locally

1. Mount Google Drive and Extract Dataset** (Cell 2)
   - Automatically extracts dataset from zip file

2. Create Smaller Dataset** (Cell 2a, Optional)
   - Limits images per class to 100 for faster experimentation
   - Removes hidden files and cleans data

3. Train Model A (MobileNetV2)** (Cell 6)
   - Transfer learning approach
   - Image size: 224x224
   - Training epochs: 5 (adjustable)

4. Train Model B (Lightweight CNN)** (Cell 8)
   - Custom architecture
   - Image size: 128x128 (configurable)
   - Faster training, smaller model size

 Option 3: Local Predictions

Run the console UI (last cell):

```python
The interface will prompt for image paths
Enter path to leaf image (or 'quit' to exit): /path/to/leaf.jpg
```

Output example:
```
Prediction Result:
Plant Type / Disease Class: Apple___Black_rot
Leaf Status: Diseased
Confidence: 0.9234
```

  Model Performance

Both models are evaluated on:
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted average precision
- **Recall**: Weighted average recall
- **F1-Score**: Weighted average F1-score

Comparison is automatically generated after training both models.

 Configuration

Key hyperparameters can be adjusted:

`python
IMAGE_SIZE = (224, 224)        # Input image dimensions
BATCH_SIZE = 32                # Batch size for training
MAX_IMAGES_PER_CLASS = 100     # Dataset size limit (optional)
```

Model A (MobileNetV2):
- Learning rate: 1e-3
- Dropout: 0.4
- Dense layer: 256 units

Model B (Custom CNN):
- Learning rate: 2e-3
- Dropout: 0.2
- Dense layer: 64 units

 Model Saving

Models are automatically saved to Google Drive with timestamps:
- `model_a_final_YYYYMMDD_HHMMSS.h5` - Final trained Model A
- `model_a_best_YYYYMMDD_HHMMSS.h5` - Best checkpoint Model A
- `model_b_final_YYYYMMDD_HHMMSS.h5` - Final trained Model B
- `model_b_best_YYYYMMDD_HHMMSS.h5` - Best checkpoint Model B

 Resume Training

To continue training an existing model:

```python
additional_epochs = 3
# Model will resume from current state
```
 Known Issues & Solutions

1. **Runtime disconnection**: Models auto-save to Google Drive
2. **Memory issues**: Use the smaller dataset option (Cell 2a)
3. **Class imbalance**: Augmentation helps but consider class weights

Contributing

Contributions are welcome! Areas for improvement:
- Add more data augmentation techniques
- Implement ensemble methods
- Add Gradio/Streamlit web interface
- Support for additional plant species
- Model quantization for mobile deployment

 Deployment

The model is deployed on Hugging Face Spaces using Gradio interface:
- **Space URL**: https://huggingface.co/spaces/yenushka/plant-disease-detector
- **Framework**: Gradio
- **Hosting**: Hugging Face Spaces (CPU)

To deploy your own version:
1. Fork the Hugging Face Space
2. Update the model files
3. Modify the Gradio interface as needed

 License

This project is provided as-is for educational and research purposes.

👥 Authors

Yenushka - [Hugging Face Profile](https://huggingface.co/yenushka)

Acknowledgments

- MobileNetV2 architecture from TensorFlow/Keras
- Plant Village dataset (or specify your dataset source)
- Google Colab for providing GPU resources
- Hugging Face for hosting the demo space

 📧 Contact

For questions or suggestions:
- Open an issue in the repository
- Visit the [Hugging Face Space](https://huggingface.co/spaces/yenushka/plant-disease-detector)
- https://huggingface.co/spaces/yenushka/plant-disease-detector

