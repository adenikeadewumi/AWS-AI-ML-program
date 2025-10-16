# 🏙️ Landmark Classification and Tagging with Deep Learning

This project applies **Convolutional Neural Networks (CNNs)** to automatically classify world landmarks in photos. It is a task inspired by real-world problems in **photo storage and sharing platforms**. When images lack GPS or metadata, this model predicts the landmark location based solely on the visual features in the image.

---

## 📖 Project Overview

Photo sharing and storage platforms often rely on metadata (like GPS coordinates) to organize and tag photos. However, many images lack this information due to device limitations or privacy settings.  
To address this, the project aims to build a **Landmark Classifier** that can:

- Identify landmarks in images using deep learning.
- Predict the most likely location or landmark class.
- Automatically tag or organize photos based on predictions.

The project represents a real-world application of **computer vision** and **machine learning deployment**, forming part of the AWS AI & ML Scholarship learning path.

---

## 🧠 Skills and Concepts Applied

- **Convolutional Neural Networks (CNNs)**
- **Data preprocessing and augmentation**
- **Transfer learning with pre-trained models (e.g., VGG, ResNet, or Inception)**
- **Model evaluation and performance tuning**
- **Deployment of CNN model in an application environment**
- **Image tagging and metadata simulation**

---

## ⚙️ Project Workflow

1. **Data Preprocessing**
   - Loaded and preprocessed landmark image dataset.
   - Applied normalization, resizing, and augmentation (rotation, flip, crop).

2. **Model Development**
   - Built CNNs from scratch and fine-tuned pre-trained architectures.
   - Experimented with different architectures to compare accuracy and loss.

3. **Model Evaluation**
   - Measured performance using accuracy, F1-score, and confusion matrix.
   - Selected the best-performing model for deployment.

4. **Deployment**
   - Integrated the model into a simple tagging app to predict and label landmarks.
   - Visualized top-K predictions for interpretability.

---

## 🧩 Tools and Frameworks

| Tool | Purpose |
|------|----------|
| **Python** | Programming language |
| **PyTorch / TensorFlow** | Deep learning framework |
| **NumPy & Pandas** | Data manipulation |
| **Matplotlib & Seaborn** | Visualization |
| **OpenCV / PIL** | Image processing |
| **Jupyter / Google Colab** | Development environment |

---

## 📊 Results and Insights

- Achieved strong classification performance on a diverse landmark dataset.
- Demonstrated the effectiveness of **transfer learning** for image recognition tasks.
- Showed how deep learning can replace manual tagging for scalability and automation.


