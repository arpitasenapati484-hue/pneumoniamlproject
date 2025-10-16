# Chest X-Ray Classification Project

## Overview
This project implements a machine learning model for classifying chest X-ray images into three categories: Normal, Pneumonia, and Tuberculosis. The model assists medical professionals in faster and more accurate diagnoses.

## 🎯 Project Goals
- Develop a robust deep learning model for medical image classification
- Implement comprehensive data preprocessing and augmentation
- Achieve reliable performance with detailed evaluation metrics
- Ensure model interpretability for clinical acceptance
- Address ethical considerations in medical AI applications

## 📋 Dataset
**Source**: Kaggle "Chest X-Ray (Pneumonia, Covid-19, Tuberculosis)" Dataset

**Classes**:
- **Normal**: Healthy lung X-rays with no visible pathology
- **Pneumonia**: X-rays showing lung infection and inflammation
- **Tuberculosis**: X-rays indicating bacterial lung infection

**Structure**: The dataset is organized into train/validation/test folders with subfolders for each class.

## 🛠️ Technical Approach

### Model Architecture
- **Base Model**: EfficientNetB0 (pretrained on ImageNet)
- **Transfer Learning**: Fine-tuned for medical image classification
- **Custom Head**: Dense layers with Dropout and BatchNormalization
- **Regularization**: Dropout (0.3-0.5), Weight Decay, Early Stopping

### Data Processing
- **Image Size**: 224×224 pixels
- **Normalization**: Pixel values scaled to [0,1]
- **Augmentation**: 
  - Random rotation (±10°)
  - Width/height shifts (10%)
  - Random zoom (10%)
  - Horizontal flip
  - Brightness adjustment (90-110%)

### Training Strategy
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Categorical cross-entropy
- **Class Balancing**: Weighted loss to handle data imbalance
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing
- **Fine-tuning**: Gradual unfreezing of pretrained layers

## 📊 Evaluation Metrics
- **Accuracy**: Overall classification performance
- **Precision**: Class-specific precision scores
- **Recall**: Class-specific sensitivity scores
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under receiver operating characteristic curves
- **Confusion Matrix**: Detailed classification results

## 🔍 Model Interpretability
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **Attention Visualization**: Highlighting regions influencing predictions
- **Confidence Scoring**: Uncertainty quantification for predictions
- **Feature Analysis**: Understanding model focus areas per class

## ⚖️ Ethical Considerations

### Bias Assessment
- **Data Bias**: Geographic and demographic representation limitations
- **Equipment Bias**: Variations in X-ray machine characteristics  
- **Population Bias**: Potential underrepresentation of certain groups
- **Labeling Bias**: Inter-radiologist diagnostic variability

### Mitigation Strategies
- Diverse dataset collection across populations and equipment
- Regular bias testing and fairness evaluation
- Continuous monitoring of performance across patient groups
- Clear guidelines for clinical use and limitations

### Clinical Safety
- Model should **supplement**, not replace, expert diagnosis
- Designed as screening/triage tool for initial assessment
- Requires validation in real clinical environments
- Emphasizes interpretability for clinical trust and adoption

## 🚀 Performance Expectations
Based on similar medical imaging studies, the model targets:
- **Overall Accuracy**: >90%
- **Per-class F1-scores**: >85% for all classes
- **Clinical Utility**: Effective screening and triage support
- **Interpretability**: Clear visualization of decision reasoning

## 📁 Repository Structure
```
chest-xray-classification/
├── notebook/
│   └── chest_xray_classification.ipynb    # Main implementation
├── data/
│   ├── train/                            # Training images
│   ├── val/                              # Validation images
│   └── test/                             # Test images
├── models/
│   └── saved_models/                     # Trained model files
├── results/
│   ├── plots/                            # Training curves, confusion matrices
│   └── reports/                          # Performance reports
├── requirements.txt                      # Python dependencies
└── README.md                            # This file
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ free disk space

### Dependencies
```bash
pip install -r requirements.txt
```

**Key Libraries**:
- TensorFlow 2.8+
- Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn
- OpenCV
- Pillow

### Dataset Setup
1. Download dataset from Kaggle
2. Extract to `data/` directory
3. Ensure proper folder structure:
   ```
   data/
   ├── train/
   │   ├── Normal/
   │   ├── Pneumonia/
   │   └── Tuberculosis/
   ├── val/
   │   └── [same structure]
   └── test/
       └── [same structure]
   ```

## 🏃 Usage

### Training the Model
1. Open `chest_xray_classification.ipynb`
2. Update dataset paths in the configuration
3. Run all cells sequentially
4. Monitor training progress and validation metrics

### Key Steps
1. **Data Loading**: Load and explore dataset distribution
2. **Preprocessing**: Apply normalization and augmentation
3. **Model Building**: Create EfficientNet-based architecture
4. **Training**: Train with callbacks and class balancing
5. **Evaluation**: Comprehensive performance assessment
6. **Interpretability**: Generate Grad-CAM visualizations
7. **Analysis**: Review ethical considerations and bias

### Model Inference
```python
# Load trained model
model = keras.models.load_model('chest_xray_model_best.h5')

# Predict on new image
prediction = model.predict(preprocessed_image)
class_probabilities = prediction[0]
predicted_class = CLASS_NAMES[np.argmax(class_probabilities)]
confidence = np.max(class_probabilities)
```

## 📈 Expected Results

### Performance Targets
- **Multi-class Accuracy**: 90-95%
- **Precision per class**: 85-95%
- **Recall per class**: 85-95%
- **F1-scores**: Balanced across all classes
- **AUC-ROC**: >0.90 for each class

### Clinical Validation
- Interpretable predictions with Grad-CAM
- Appropriate confidence calibration
- Robust performance across different image qualities
- Suitable for screening and triage applications

## 🔬 Future Enhancements

### Technical Improvements
- **Ensemble Methods**: Combine multiple architectures
- **Attention Mechanisms**: Advanced spatial attention
- **Self-supervised Pretraining**: Medical image-specific pretraining
- **Multi-label Classification**: Handle co-existing conditions

### Clinical Integration
- **DICOM Compatibility**: Direct medical imaging format support
- **Real-time Inference**: Optimized for clinical workflows
- **User Interface**: Radiologist-friendly visualization tools
- **Continuous Learning**: Adaptation to new data

### Validation & Deployment
- **Multi-site Validation**: Testing across different hospitals
- **Regulatory Approval**: FDA/CE marking pathways
- **Clinical Trials**: Prospective validation studies
- **Performance Monitoring**: Real-world performance tracking

## ⚠️ Important Notes

### Limitations
- Model trained on specific dataset characteristics
- Performance may vary with different imaging equipment
- Requires validation in target clinical environments
- Should not be used as sole diagnostic tool

### Disclaimer
This model is for **research and educational purposes only**. It has not been validated for clinical use and should not be used for actual medical diagnosis without proper regulatory approval and clinical validation.

### Clinical Responsibility
- Always consult qualified medical professionals
- Model outputs should supplement, not replace, expert judgment
- Continuous monitoring and validation required in clinical settings
- Adherence to local medical regulations and guidelines essential

## 📚 References

1. **EfficientNet**: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.

2. **Medical Imaging AI**: Rajpurkar, P., et al. (2017). CheXNet: Radiologist-level pneumonia detection on chest X-rays with deep learning.

3. **Transfer Learning**: Raghu, M., et al. (2019). Transfusion: Understanding transfer learning for medical imaging.

4. **Grad-CAM**: Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization.

5. **Medical AI Ethics**: Char, D. S., et al. (2018). Implementing machine learning in health care—addressing ethical challenges.

## 📞 Support & Contribution

For questions, issues, or contributions to this project:

- **Issues**: Open GitHub issues for bugs or feature requests
- **Contributions**: Submit pull requests with improvements
- **Documentation**: Help improve documentation and examples
- **Validation**: Contribute to clinical validation efforts

---

**Note**: This project demonstrates deep learning techniques for medical image analysis with proper consideration of ethical implications and clinical requirements. It serves as an educational resource for understanding AI applications in healthcare while emphasizing the importance of responsible AI development and deployment.
