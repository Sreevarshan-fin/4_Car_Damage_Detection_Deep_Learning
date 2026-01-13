# Car_Damage_Detection_Deep_Learning


[![Launch App](https://img.shields.io/badge/Launch%20App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://cardamagedetectiondeeplearning-bbjntpd5cibrxdkwbl3p46.streamlit.app/)
&nbsp;&nbsp;

---

## **Project Overview**

In the automotive resale and inspection space, assessing car damage is still largely dependent on **manual checks**, which can be slow, inconsistent, and subjective. To explore a smarter alternative, this project was developed as a **Proof of Concept (POC) for VROOM Cars**.

The idea was simple but powerful:
*Can a deep learning model look at a car image and reliably tell whether the car is damaged — and how badly?*

To answer this, I built a **Car Damage Detection system** that analyzes images of a car’s **front and rear sections**, identifies the type of damage, and presents the prediction through an easy-to-use **Streamlit web application**.
This POC serves as a foundation to evaluate whether automated damage detection can realistically support vehicle inspections at scale.

---

```
Car_Damage_Detection_Deep_Learning/
│
├── app.py                  # Streamlit app
├── model_helper.py         # Model loading + prediction logic
├── requirements.txt
├── README.md
├── model/
│   └── saved_model_sree_varshan.pth   # Stored via Git LFS
```


## **Problem Statement**

In real-world car inspections, damage assessment often varies from one inspector to another. Minor breakages can be missed, while damage severity can be overestimated — leading to inconsistent decisions and delays.

The goal of this project was to **reduce subjectivity** by building an **image-based classification model** that can automatically detect car damage and classify it into predefined categories with **at least 75% accuracy**.

From a user’s perspective, the system is simple:

* Upload a car image
* Let the model analyze it
* Instantly receive a prediction showing the **location and severity of damage**

---

## **Classes**

To make predictions meaningful and actionable, damage was categorized based on **vehicle position** and **severity**.
The model was trained to classify images into **six clear categories**:

1. Front Normal
2. Front Breakage
3. Front Crushed
4. Rear Normal
5. Rear Breakage
6. Rear Crushed

This structure allows the system not just to say *“damaged”*, but to explain **where** the damage is and **how serious** it is — which is critical for inspection and pricing decisions.

---

## **Models Used**

Instead of relying on a single approach, I experimented with multiple deep learning models to understand their strengths and limitations for this problem.

---

### **1. Convolutional Neural Network (CNN)**

I began by building a **custom CNN from scratch**.
This helped establish a baseline and allowed the model to learn basic visual patterns such as dents, cracks, and crushed surfaces directly from the dataset.

However, this approach showed signs of **overfitting** and struggled to generalize well on unseen images.

---

### **2. CNN with Regularization**

To improve performance, I enhanced the CNN using **regularization techniques**, including:

* Dropout
* Batch normalization

These changes improved stability and reduced overfitting, but the model still had limitations when distinguishing between visually similar damage types.

---

### **3. Pre-Trained Model – EfficientNet**

Next, I applied **transfer learning using EfficientNet**.
Because EfficientNet is trained on large-scale image datasets, it already understands complex visual features like textures, shapes, and edges.

This significantly improved accuracy and training efficiency compared to the custom CNN models.

---

### **4. Pre-Trained Model – ResNet (Selected Best Model)**

Finally, I implemented a **ResNet-based model**, which emerged as the **best-performing solution**.

ResNet consistently delivered **higher validation accuracy** and more stable results compared to the other models. Its residual connections allowed the network to learn deeper and more meaningful image features without losing important information.

Most importantly, **ResNet met and exceeded the required accuracy threshold**, especially in challenging cases where damage types like **breakage and crushed** look visually similar.
Because of this strong and consistent performance, **ResNet was selected as the final model for deployment**.

---

#  **Final Outcome**

* Designed and evaluated multiple deep learning models for car damage detection
* Compared baseline CNNs with advanced transfer learning approaches
* Selected **ResNet as the final model because it met the accuracy requirement and performed most consistently**
* Integrated the model into a **Streamlit application** for real-time image upload and prediction
* Delivered a **scalable, production-ready POC** demonstrating clear business value


## **Demo**

**Front Crushed**

<img width="1033" height="757" alt="image" src="https://github.com/user-attachments/assets/0c24b8b2-6951-4f8d-af03-d0c7e8429d3d" />

-----------------------

**Rear Crushed**

<img width="1017" height="707" alt="image" src="https://github.com/user-attachments/assets/acc26a78-fb94-4b67-a69a-1fc68090c5a4" />

--------------


