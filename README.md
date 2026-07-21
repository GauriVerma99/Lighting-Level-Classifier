# Lighting-Level-Classifier

This project is a simple deep learning model built using **PyTorch** to classify images based on their lighting conditions. The model predicts whether an image belongs to one of three categories:

- Bright
- Normal
- Dark

I built this project to learn the complete image classification workflow, from preparing the dataset to training a model and making predictions on new images.

## Tech Stack

- Python
- PyTorch
- Torchvision
- Pillow

## Project Structure

```
Lighting-Level-Classifier/
│── data/
│── train.py
│── predict.py
│── submit.py
│── best_model.pth
│── requirements.txt
```

## Running the Project

Train the model:

```bash
python3 train.py
```

Predict on a new image:

```bash
python3 predict.py image.jpg
```

> The original dataset is not included in this repository. A small sample dataset is provided for demonstration.

---

Made by **Gauri Verma**
