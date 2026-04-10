# 🛒 E-Commerce Recommender System

A hybrid recommendation system that combines **Collaborative Filtering (CF)** and **Content-Based Filtering (CB)** to provide personalized product suggestions.

---

## 📌 Features

* Hybrid recommendation model (CF + CB)
* Product similarity using TF-IDF
* Personalized recommendations
* RMSE evaluation support
* Flask-based web interface

---

## 📁 Project Structure

```
ecommerce-recommender/
│
├── app/                        
│   ├── routes.py              
│   ├── recommender.py         
│   ├── preprocessing.py       
│   ├── model.py               
│   └── utils.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
│
├── templates/                 
├── static/                    
├── notebooks/                 
├── config.py                  
├── run.py                     
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/your-username/ecommerce-recommender.git
cd ecommerce-recommender
```

---

### 2. Create virtual environment

```
python -m venv venv
```

---

### 3. Activate environment

**Windows**

```
venv\Scripts\activate
```

**Mac/Linux**

```
source venv/bin/activate
```

---

### 4. Install dependencies

```
pip install -r requirements.txt
```

---

### 5. Prepare Data

* Place your dataset inside:

```
data/raw/
```

* Run preprocessing (if required):

```
python -m app.preprocessing
```

---

### 6. Run the Application

```
python run.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## 📊 Model Evaluation (RMSE)

Run:

```
python -m app.evaluate
```

---

## 🧠 Algorithms Used

* Collaborative Filtering
* Content-Based Filtering (TF-IDF)
* Hybrid Recommendation System

---

## 📌 Notes

* Dataset is not included due to size.
* Add your dataset in `data/raw/`
* Model files (`.pkl`) are ignored using `.gitignore`

---

## 🚀 Future Improvements

* Precision@K and Recall@K
* Deep Learning-based recommendations
* Real-time recommendation API

---

## 👩‍💻 Author

Your Name
