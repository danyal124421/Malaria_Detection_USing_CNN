# Malaria-Cell-Classification-with-CNN

A concise TensorFlow 2.x notebook that trains, evaluates and saves a Convolutional Neural Network to detect **malaria-infected** vs. **uninfected** human red-blood cells.  
Dataset: [NIH Malaria Dataset](https://lhncbc.nlm.nih.gov/publication/pub9932) – automatically downloaded through TensorFlow-Datasets (`tfds`).

---

## 📌 Highlights
- Binary image-classification (Infected = 1, Uninfected = 0)  
- 27 558 cell images (100×100 px) → resized to 224×224 px  
- 80 % train / 10 % validation / 10 % test split  
- Two model versions compared:
  1. Baseline CNN with *Sigmoid* activations
  2. Improved CNN with *ReLU* + *Batch-Normalisation*  
- Training & validation curves visualised  
- Inference on test images + Google-Drive model persistence  
- Ready-to-run in **Google Colab** (GPU runtime recommended)

---

## 🚀 Quick Start (Colab)
1. Open notebook in Colab  
2. Runtime → Change runtime type → GPU  
3. Run all cells top-to-bottom  
4. Authorise Drive mount when prompted – model is saved to  
   `/content/drive/MyDrive/model_relu.keras`

---

## 📁 Project Structure
```
malaria_cell_classification.ipynb   # this notebook
└── model_relu.keras               # exported model (after training)
```

---

## 🔧 Dependencies
| Package        | Version |
|----------------|---------|
| Python         | ≥3.8    |
| TensorFlow     | ≥2.10   |
| tensorflow-datasets | ≥4.6 |
| NumPy          | ≥1.21   |
| Matplotlib     | ≥3.5    |

Install once:
```bash
pip install -q tensorflow tensorflow-datasets numpy matplotlib
```

---

## 🧪 Data Pipeline
1. Load `tfds.load('malaria', …)`  
2. Split deterministically into train/val/test  
3. `tf.image.resize` → 224×224  
4. Normalise pixels to [0,1]  
5. Batch (32) → prefetch (`tf.data.AUTOTUNE`)

---

## 🏗️ Model Architecture (final)
| Layer                 | Config                                 |
|-----------------------|----------------------------------------|
| Input                 | (224, 224, 3)                          |
| Conv2D                | 6 filters, 3×3, ReLU                   |
| BatchNorm             | —                                      |
| MaxPool2D             | 2×2                                    |
| Conv2D                | 16 filters, 3×3, ReLU                  |
| BatchNorm             | —                                      |
| MaxPool2D             | 2×2                                    |
| Flatten               | —                                      |
| Dense                 | 100 units, ReLU                        |
| BatchNorm             | —                                      |
| Dense                 | 10 units, ReLU                         |
| BatchNorm             | —                                      |
| Dense                 | 1 unit, Sigmoid (probability)          |

---

## 📊 Metrics Tracked
- `BinaryAccuracy`
- `Precision`, `Recall`
- `True/False Positives & Negatives`

---

## 🎯 Results (example, 15 epochs – GPU)
```
val_accuracy ≈ 0.9343 
val_loss     ≈ 0.2806
```
*Exact numbers vary slightly per run.*

---

## 🔮 Inference Helper
```python
def predict_image(prob):
    return 'Uninfected' if prob > 0.5 else 'Infected'

pred = model.predict(my_image)
print(predict_image(pred[0][0]))
```

---

## 💾 Save / Load
```python
model.save('model_relu.keras')
reloaded = tf.keras.models.load_model('model_relu.keras')
```

---

## 📄 License
Public domain – built on open NIH data and TFDS.

---

## 🤝 Contributing
Feel free to open issues or PRs if you spot bugs or have improvement ideas.
