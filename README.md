<div align="center">

# 🧠 CNN Image Classification Labs

**Hands-on Convolutional Neural Network labs built with TensorFlow & Keras**

From handwritten digits to a custom flower dataset, one lab at a time.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)
![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📖 Table of Contents

- [About](#-about)
- [Labs Overview](#-labs-overview)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Lab Details](#-lab-details)
- [Results](#-results)
- [Concepts Covered](#-concepts-covered)
- [Technologies Used](#-technologies-used)
- [Troubleshooting](#-troubleshooting)
- [Author](#-author)
- [License](#-license)

---

## 🎯 About

This repository contains a series of exercises for the **Artificial Neural Networks**.

Each builds on the previous one. You start with a simple CNN on handwritten digits and finish by training a classifier on your own folder of images. The code has been cleaned up and corrected so that it runs on current versions of TensorFlow/Keras in Google Colab.

---

## 🧪 Labs Overview

| # | Lab | Dataset | Image Size | Classes | Approx. Test Accuracy |
|:-:|-----|---------|:----------:|:-------:|:---------------------:|
| 1 | CNN with CIFAR10 | CIFAR10 | 32×32 RGB | 10 | ~71% |
| 2 | CNN with Fashion-MNIST | Fashion-MNIST | 28×28 grey | 10 | ~92% |
| 3A | CNN with MNIST (pooling + dropout) | MNIST | 28×28 grey | 10 | ~99% |
| 3B | CNN with MNIST (basic) | MNIST | 28×28 grey | 10 | ~98% |
| 4 | CNN with a Custom Dataset | TensorFlow Flowers | 64×64 RGB | 5 | ~65–75% |

> 📝 Accuracy values are approximate. They vary by a point or two between runs because of random weight initialization.

---

## 📁 Project Structure

Adjust the file names below to match your own folder.

```
cnn-image-classification-labs/
│
├── 01_cnn_cifar10.py             # Lab 1: CIFAR10
├── 02_cnn_fashion_mnist.py       # Lab 2: Fashion-MNIST
├── 03a_cnn_mnist_pooling.py      # Lab 3A: MNIST with pooling and dropout
├── 03b_cnn_mnist_basic.py        # Lab 3B: MNIST basic
├── 04_cnn_custom_dataset.py      # Lab 4: custom dataset (Flowers)
│

```

> 💡 Datasets are **not** stored in this repository. Each script downloads its data automatically on first run.

---

## 🚀 Getting Started

### Option 1: Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com) and create a new notebook.
2. Enable the GPU: **Runtime → Change runtime type → T4 GPU**.
3. Copy the code of a lab into the notebook cells and run them in order.
4. Use a **fresh notebook (or restart the session)** for each lab so variables don't carry over.

### Option 2: Run locally

```bash
# 1. Clone the repository
git clone https://github.com/mabdullahbuzdar/CNN-Image-Classification.git
cd cnn-image-classification-labs

# 2. (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install tensorflow numpy matplotlib opencv-python scikit-learn imutils

# 4. Run a lab
python 01_cnn_cifar10.py
```

---

## 🔬 Lab Details

### Lab 1: CNN with CIFAR10
Classifies 60,000 small colour images into 10 mutually exclusive classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck.

- **Architecture:** 3 × Conv2D (32, 64, 64) with 2 × MaxPooling → Flatten → Dense(64) → Dense(10)
- **Loss / Optimizer:** Sparse categorical cross-entropy (`from_logits=True`) / Adam
- **Training:** 10 epochs
- **Observation:** validation accuracy flattens near 71% while training accuracy keeps rising, which is a sign of mild overfitting.

### Lab 2: CNN with Fashion-MNIST
Classifies 70,000 greyscale images of clothing into 10 categories (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

- **Architecture:** 3 × Conv2D (32, 64, 128) → MaxPooling → Dropout(0.25) → Flatten → Dense(128) → Dropout(0.5) → Softmax(10)
- **Loss / Optimizer:** Sparse categorical cross-entropy / Adam
- **Training:** 12 epochs, batch size 128
- **Extras:** accuracy and loss curves, saving weights, and predicting a single image.

### Lab 3: CNN with MNIST
Recognises handwritten digits (0–9) from 70,000 greyscale images. Two versions are included.

| Version | Architecture | Epochs |
|---------|--------------|:------:|
| **3A: Pooling + Dropout** | Conv(32) → Pool → Conv(64) → Pool → Flatten → Dropout(0.5) → Softmax | 15 |
| **3B: Basic** | Conv(64) → Conv(32) → Flatten → Softmax | 10 |

- **Loss / Optimizer:** Categorical cross-entropy / Adam, with one-hot encoded labels.
- **Observation:** adding pooling and dropout in 3A gives higher accuracy with fewer parameters than the basic model in 3B.

### Lab 4: CNN with a Custom Dataset
Shows the full workflow for training on **your own images** stored in one folder per class. This lab uses the **TensorFlow Flowers** dataset (3,670 photos).

- **Classes:** daisy, dandelion, roses, sunflowers, tulips
- **Pipeline:** load with OpenCV → resize to 64×64 → scale to [0, 1] → 75/25 stratified split → `LabelBinarizer` → CNN → classification report
- **Architecture:** 3 × (Conv2D + ReLU + MaxPooling) → Flatten → Dense(128) → Dropout(0.5) → Softmax(5)
- **Optional:** data augmentation (rotation, shift, flip, zoom) to reduce overfitting.

**To use your own images,** arrange them like this and point `data_dir` at the parent folder:

```
my_dataset/
├── class_a/   img1.jpg  img2.jpg ...
└── class_b/   img1.jpg  img2.jpg ...
```

Labels are taken from the folder names, so no other changes are needed apart from the `classes` count.

---

## 📊 Results

Typical results with the provided settings. Add your own screenshots to a `images/` folder to show them here.

| Lab | Train Acc. | Test Acc. | Notes |
|-----|:----------:|:---------:|-------|
| CIFAR10 | ~79% | ~71% | Overfits slightly after epoch 5 |
| Fashion-MNIST | ~95% | ~92% | Shirt and T-shirt are the hardest classes |
| MNIST (pooling) | ~99% | ~99% | Test loss ≈ 0.025 |
| MNIST (basic) | ~99% | ~98% | More parameters, slightly less accurate |
| Flowers | ~95%+ | ~65–75% | Roses and tulips are most often confused |

<!-- Uncomment and add your own images:
![Accuracy curve](images/accuracy_curve.png)
![Sample predictions](images/predictions.png)
-->

---

## 📚 Concepts Covered

- Convolutional layers and feature extraction
- Max pooling and spatial down-sampling
- Dropout and overfitting
- Data normalization and reshaping
- One-hot encoding of labels
- Train/test splitting and validation
- Loss functions: sparse vs. categorical cross-entropy
- Optimizers: Adam and SGD
- Model evaluation: accuracy, loss curves, classification report
- Data augmentation
- Training on a custom image dataset

---

## 🛠 Technologies Used

| Tool | Purpose |
|------|---------|
| **Python** | Programming language |
| **TensorFlow / Keras** | Building and training the CNNs |
| **OpenCV** | Loading and resizing custom images |
| **scikit-learn** | Data splitting, label encoding and evaluation reports |
| **NumPy** | Array operations |
| **Matplotlib** | Plots and image visualization |
| **Google Colab** | Free GPU for training |

---

## 🩺 Troubleshooting

| Problem | Solution |
|---------|----------|
| `KeyError: 'acc'` when plotting | Newer Keras uses `accuracy` and `val_accuracy` as the key names. |
| `ModuleNotFoundError: imutils` | Run `pip install imutils` (in Colab: `!pip install imutils`). |
| `img_to_array` import error | Use `from tensorflow.keras.utils import img_to_array`. |
| Training is very slow | Switch to a GPU runtime in Colab. |
| `NameError: X_train` | Variable names are case-sensitive. Use `X_train`, not `x_train`. |
| Weights file error when saving | Name the file `model.weights.h5`. |

---

## 👤 Author

**Muhammad Abdullah**
Computer Science and IT, University of Balochistan
Course: Artificial Neural Networks
Instructor: Dr. Abdul Basit

🔗 GitHub: https://github.com/mabdullahbuzdar/CNN-Image-Classification.git

---

## 📄 License

This project is licensed under the **MIT License**. You are free to use, modify and share it with attribution.

Lab concepts and the original exercises are based on course material by **Dr. Abdul Basit**, University of Balochistan.

---

<div align="center">

⭐ If you found this repository helpful, consider giving it a star! ⭐

</div>
