## 🧠 **README.md**

```markdown
# 🧩 Deepfake Detection Using Transfer Learning

> A deep learning pipeline to detect AI-generated (deepfake) images using transfer learning on ResNet18.  
> Built for **Synergy’25 – Deepfake ML Model Hackathon**.

---

## 🚀 Project Overview

Deepfakes pose a serious challenge to digital authenticity.  
This project develops a **deepfake detection model** that predicts whether an image is *real* or *fake*, simulating the output distribution of a proprietary deepfake detector.

The model learns from a dataset of labeled real and fake images and outputs prediction scores for unseen test images.  
It uses **transfer learning (ResNet18)** to achieve high accuracy with limited training data.

---

## 🧾 Problem Statement

> Build a predictive model that maps image features to deepfake detection scores.  
> Use the provided training images and proprietary model outputs (in JSON) to generalize on unseen test data.

The output should be a JSON file (`teamname_prediction.json`) with predicted detection scores for all test images.

---

## 📂 Dataset Description

**Folders:**
```

data/
├── real_cifake_images/        # Real training images
├── fake_cifake_images/        # Fake training images
├── test_images/               # Unlabeled test images
├── real_cifake_preds.json     # Proprietary model outputs for real images
├── fake_cifake_preds.json     # Proprietary model outputs for fake images
└── train_meta.json            # Combined training metadata (created by merge_jsons.py)

````

**train_meta.json Format:**
```json
{
  "index": 1,
  "label": 1,
  "prediction": "real"
}
````

* `label`: 1 → Real, 0 → Fake
* `index`: Corresponds to image filename (e.g., `1.png`, `2.png`, …)

---

## ⚙️ Methodology

### 1. **Data Preprocessing**

* Resized all images to `224×224` pixels.
* Normalized using ImageNet mean and standard deviation.
* Custom PyTorch `Dataset` dynamically loads labeled images.
* Missing indices are auto-handled gracefully.

### 2. **Model Architecture**

* **Base Model:** `ResNet18` pretrained on ImageNet.
* **Modified Layers:**

  ```
  Linear(512 → 128) → ReLU → Dropout(0.3) → Linear(128 → 1) → Sigmoid
  ```
* **Loss:** Binary Cross Entropy (BCE)
* **Optimizer:** Adam (learning rate = 1e-4)
* **Batch Size:** 16
* **Epochs:** 5

### 3. **Training**

* Trained on 2000 images (1000 real + 1000 fake).
* Validation split: 10%.
* Achieved **~90% validation accuracy** within 5 epochs.

### 4. **Inference**

* Loads saved model (`outputs/model.pth`)
* Predicts probability scores (0–1) for all test images.
* Outputs results in JSON format identical to the proprietary model.

---

## 📈 Results

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | ~96%  |
| Validation Accuracy | ~90%  |
| Validation Loss     | 0.23  |
| Best Epoch          | 3     |

**Observation:**

* Model effectively distinguishes real vs fake faces.
* Slight overfitting observed after Epoch 4 (expected due to small dataset).

**Performance Graphs:**

* `outputs/loss_curve.png`
* `outputs/accuracy_curve.png`

---

## 🧮 Output Example

**File:** `outputs/teamname_prediction.json`

```json
[
  {"image_id": "1.png", "prediction": 0.00025},
  {"image_id": "10.png", "prediction": 0.99988},
  {"image_id": "20.png", "prediction": 0.00213}
]
```

---

## 💡 Key Features

* Transfer learning with pretrained CNNs
* Automatic dataset merging and JSON handling
* Robust dataloader with missing file tolerance
* CPU and GPU compatible
* Visualization of training performance

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

```
deepfake-hackathon/
├── data/                        # Dataset folders & JSONs
├── scripts/
│   ├── dataset.py               # Custom PyTorch Dataset class
│   ├── merge_jsons.py           # Combines real & fake JSONs
│   ├── train.py                 # Model training script
│   ├── infer.py                 # Inference script (prediction generation)
│   └── plot_training_graphs.py  # Generates accuracy/loss plots
├── outputs/
│   ├── model.pth                # Saved trained model
│   ├── loss_curve.png
│   ├── accuracy_curve.png
│   └── teamname_prediction.json # Final submission file
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

## 🧠 Future Improvements

* Use deeper networks (ResNet50 / EfficientNet / Vision Transformer).
* Integrate facial landmarks or embeddings for higher accuracy.
* Expand dataset with multi-ethnic and multi-lighting images.
* Deploy real-time deepfake detection web app (Flask/Streamlit).

---

## 🏁 Conclusion

This project demonstrates how **transfer learning** can effectively detect deepfakes using limited data.
The resulting model achieves **90% accuracy**, with stable performance and a scalable, reproducible pipeline.

It’s a strong foundation for future research and real-world applications in **AI-based content verification**.

---

## 👤 Author

**Shashank Padmasale**
💻 B.Tech – Computer Science
📧 shashankpadmasale@gmail.com

---



