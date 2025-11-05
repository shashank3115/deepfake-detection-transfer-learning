
# 🧠 Deepfake Detection using Transfer Learning (ResNet18, PyTorch)

[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Hackathon](https://img.shields.io/badge/Event-Synergy’25%20Hackathon-blue)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-90%25-success)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen)]()

> 🚀 **AI-driven Deepfake Detection Pipeline** using Transfer Learning on **ResNet18**, designed to predict proprietary deepfake detector outputs.  
> Developed for **Synergy’25 – Deepfake ML Model Hackathon**.

---

## 🔍 Overview

Deepfakes are rapidly evolving threats to digital authenticity.  
This project builds a **Deepfake Detection Model** that predicts whether an image is *real* or *AI-generated (fake)*, replicating the output distribution of a proprietary model.  

Leveraging **Transfer Learning** on **ResNet18**, the model achieves **~90% validation accuracy** while remaining lightweight and scalable.

---

## 🎯 Problem Statement

> **Goal:** Develop a predictive model that maps image features to deepfake detection scores.  
> Use provided training images and proprietary model outputs (JSON format) to generalize effectively to unseen test data.  
>  
> **Expected Output:** A JSON file (`teamname_prediction.json`) containing detection probabilities for all test images.

---

## 🗂️ Dataset Description

### Folder Structure

```
data/
├── real_cifake_images/        # Real training images
├── fake_cifake_images/        # Fake training images
├── test_images/               # Unlabeled test images
├── real_cifake_preds.json     # Proprietary outputs (real images)
├── fake_cifake_preds.json     # Proprietary outputs (fake images)
└── train_meta.json            # Combined metadata file
```
### JSON Format Example

```json
{
  "index": 1,
  "label": 1,
  "prediction": "real"
}
```

* `label`: 1 → Real, 0 → Fake
* `index`: Corresponds to the image filename (`1.png`, `2.png`, …)

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing

* Resized all images to `224×224` pixels
* Normalized using ImageNet mean & std deviation
* Implemented a custom **PyTorch Dataset** class
* Automatically handles missing image indices gracefully

### 2️⃣ Model Architecture

* **Base Model:** ResNet18 pretrained on ImageNet
* **Modified Layers:**

  ```
  Linear(512 → 128) → ReLU → Dropout(0.3) → Linear(128 → 1) → Sigmoid
  ```
* **Loss Function:** Binary Cross Entropy (BCE)
* **Optimizer:** Adam (`lr = 1e-4`)
* **Batch Size:** 16
* **Epochs:** 5

### 3️⃣ Training

* Dataset: 2000 images (1000 real + 1000 fake)
* Validation Split: 10%
* Achieved **~90% validation accuracy** by Epoch 3
---

### ⚡ Efficiency & Robustness Notes

- **Inference Speed:** ~0.02 seconds per image on CPU  
- **Model Size:** ~44 MB (lightweight and deployable)  
- **Hardware Compatibility:** Runs efficiently on both CPU and NVIDIA RTX GPUs  
- **Data Augmentation:**  
  - Applied random horizontal flips, slight rotations, and normalization for better generalization  
  - Helped reduce overfitting and improve validation stability  

These enhancements improve both **model efficiency** and **robustness**, ensuring that the pipeline performs consistently across unseen data while maintaining fast inference times.

---


### 4️⃣ Inference

* Loads the saved model (`outputs/model.pth`)
* Generates probability scores (0–1) for all test images
* Produces a structured JSON output identical to proprietary format

---

## 📈 Results

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | ~96%  |
| Validation Accuracy | ~90%  |
| Validation Loss     | 0.23  |
| Best Epoch          | 3     |

**Observations:**

* Model effectively distinguishes real vs fake faces
* Slight overfitting after Epoch 4 (expected due to small dataset)

**Performance Graphs:**

* `outputs/loss_curve.png`
* `outputs/accuracy_curve.png`

---

## 🧮 Output Example

**File:** `outputs/shashank_prediction.json`

```json
[
  {"image_id": "1.png", "prediction": 0.00025},
  {"image_id": "10.png", "prediction": 0.99988},
  {"image_id": "20.png", "prediction": 0.00213}
]
```

---

## 💡 Key Features

✅ Transfer learning with pretrained CNNs
✅ Automatic dataset merging and JSON handling
✅ Robust dataloader with missing file tolerance
✅ Compatible with both CPU and GPU
✅ Visualization of loss and accuracy metrics

---

## 🧰 Tech Stack

| Component       | Technology                   |
| --------------- | ---------------------------- |
| Language        | Python                       |
| Framework       | PyTorch                      |
| Model           | ResNet18 (Transfer Learning) |
| Data Handling   | Pillow, NumPy                |
| Visualization   | Matplotlib                   |
| IDE             | VS Code                      |
| Version Control | Git, Git Bash                |

---

## 📦 Project Structure

```bash
deepfake-hackathon/
├── data/                        # Dataset folders & JSONs
├── scripts/
│   ├── dataset.py               # Custom Dataset class
│   ├── merge_jsons.py           # Combines real & fake JSONs
│   ├── train.py                 # Training script
│   ├── infer.py                 # Inference script
│   ├── make_submission.py       # Automates ZIP creation
│   └── plot_training_graphs.py  # Generates accuracy/loss plots
├── outputs/
│   ├── model.pth                # Saved trained model
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── shashank_prediction.json # Final JSON output
├── requirements.txt
└── README.md
```

---

## 🧪 How to Run Locally

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Merge JSON Files

```bash
python scripts/merge_jsons.py
```

### 4️⃣ Train the Model

```bash
python scripts/train.py
```

### 5️⃣ Generate Predictions

```bash
python scripts/infer.py
```

### 6️⃣ Plot Training Graphs

```bash
python scripts/plot_training_graphs.py
```

---

## 🔮 Future Improvements

🚀 Upgrade to ResNet50 / EfficientNet / Vision Transformer
🧠 Integrate facial embeddings or landmarks for improved accuracy
🧩 Expand dataset with diverse lighting and ethnic variations
🌐 Deploy real-time deepfake detection web app (Flask/Streamlit)

---

## 🏁 Conclusion

This project demonstrates how **Transfer Learning** can effectively detect deepfakes using limited data.
The resulting model achieves **90% accuracy**, with robust generalization and reproducible results.

It serves as a strong foundation for future research in **AI-based content verification and digital forensics**.

---

## 👤 Author

**Shashank Padmasale**
💻 B.Tech – Computer Science & Engineering
📧 [shashankpadmasale@gmail.com](mailto:shashankpadmasale@gmail.com)
🌐 [GitHub: shashank3115](https://github.com/shashank3115)

---

## 🏆 Acknowledgements

Special thanks to **Synergy’25 Hackathon Organizers** for providing the dataset and framework.
Built with 💻, ☕, and a passion for exploring **AI innovation**.

---


