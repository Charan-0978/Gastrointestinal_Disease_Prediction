# Gastrointestinal Disease Prediction

This project aims to build a deep learning model for **Gastrointestinal (GI) Disease Detection** using image data.  
The model leverages Convolutional Neural Networks (CNNs) to classify medical images into different GI disease categories.

---

## 📂 Dataset

The dataset is too large (~1.5 GB) to upload to GitHub.  
You can download it from the following link:  

👉 [Download Dataset](https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset)  

After downloading, place the dataset inside the project folder:

project-root/
│-- Gastrointenstinal disease detection/
| |__ kvasir - dataset/
│ ├── data_preprocessing.py/
│ ├── train_model.py/
│ └── model_cnn.py/
| |__ streamlit_app.py
     

---

## 🚀 Features

- Image preprocessing and augmentation  
- CNN-based deep learning model for disease classification  
- Training, validation, and evaluation scripts  
- Streamlit app for interactive disease prediction  

---

## 🛠️ Installation

Clone this repository:

```bash
git clone https://github.com/your-username/gastrointestinal-disease-prediction.git
cd gastrointestinal-disease-prediction
Install dependencies:

bash
Copy code
pip install -r requirements.txt
▶️ Usage
1. Preprocessing the data
bash
python data_preprocessing.py
2. Train the Model
bash
python train_model.py
3. Run the Streamlit App
bash
streamlit run app.py
📊 Results
Achieved 91% accuracy on the validation dataset

Model performance evaluated using accuracy, precision, recall, and F1-score

📌 Acknowledgments
Dataset source: [(https://www.kaggle.com/datasets/meetnagadia/kvasir-dataset)]

Libraries: TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Streamlit

📧 Contact
For queries or collaborations, feel free to reach out:
Charan Muppidi – [muppidicharan019@gmail.com]
