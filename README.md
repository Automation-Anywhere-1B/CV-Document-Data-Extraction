# CalVision: __CV-Document-Data-Extraction__

Welcome to **CalVision**! This was made from the Automation Anywhere 1-B Team to help automate the parsing of financial checks. 

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                                                                                  |
|------------------|---------------|-------------------------------------------------------------------------------------------------------------------------------|
| Margaret Galvez  | @margoglvz    | Data exploration, overall project coordination, Streamlit interface, EfficientDet, connecting YOLO Stacked to interface       |
| Thy Tran         | @thyatran     | Data exploration, Detectron2 training and optimization, results interpretation and analysis                                    |
| Gurnoor Bola     | @GurnoorBola  | Data exploration, YOLO Model & YOLO Stacked training and optimization, results interpretation and analysis                  |
| Ricardo Tellez   | @RVARELAT     | Data exploration, Streamlit interface, performance analysis                                                                   |
| Meera Vyas       | @meeraa5      | Detectron model testing, performance analysis, results interpretation                                                         |
| Aaryan Hakim     | @syaaryan     | RT-DETR training, connecting YOLO to interface, Streamlit interface                                                           |


---

## 🎯 **Project Highlights**

**Example:**

- Developed 3 Computer Vision models using YOLO, Resnet, and Detectron to address the manual and tedious task of scanning and identifying text fields from checks.
- YOLO Model
  - Achieved 95% accuracy on forged signature classifications and 99% accuracy on genuine signature classifications, demonstrating high impact for Automation Anywhere.
  - Achieved 100% accuracy on identifying amount, amount_words, date and payee
- YOLO Stacked (YOLO + ResNet)
  - Achieved 100% accuracy on identifying amount, amount_words, date and payee
  - Achieve 95% accuracy on classifying forged or genuine signatures
- Detectron
  - 68.3% accuracy on identifying amount, 75.8% accuracy on classifying genuine signatures
  - Most values fall below 0.5, meaning the model is often unsure between neighboring classes
- Generated actionable insights on text fields to improve scanning and inform business decisions at Automation Anywhere.

---

## 👩🏽‍💻 **Setup and Installation**

### Intallation Guide
1. In order to run this app locally, please run 

    ```pip install -r requirements.txt```

    to install at the necessary libraries. 

2. After installing requirements.txt and Detectron2 Weights, in your **root** directory, run

    ```streamlit run streamlit/app.py```

    to open the app on your local device.

3. You should see this on your screen once it is fully loaded!
   <img width="757" height="383" alt="image" src="https://github.com/user-attachments/assets/836e52d6-154c-437a-a0ba-8123e4a69bcd" />


### Detectron2 Weights

The trained Detectron2 model weights (`model_final.pth`) are not stored in the repo
due to GitHub's 100MB file limit.

Please download them from: [Link](https://drive.google.com/file/d/1jWxcURBcVml9_GHqsgy8Gv0I-keas6m1/view?usp=drive_link)

Then place the file at:

`Detectron2/output/model_final.pth`

### User Guide
In order to submit your photo, please make sure to follow these simple instructions:

1. Position your check in the center of your camera

2. Make sure you have good lighting and the **_whole_** check is visible

3. Confirm that your check is in **_focus_** and there are no other objects in the photo

4. You are ready to submit!

5. Upload your document, then click **_Run All Models_** to compare YOLO, YOLO Stacked, and Detectron2.


---

## 🏗️ **Project Overview**

**Describe:**

- This project allowed us to implement the machine learning life cycle into a real problem and perform to industry standards
- Automation Anywhere wanted a project that could automate the identification of text fields of texts as manual identification can result in errors and is tedious
- This will help identify fields from checks faster and create an easier time to input text fields 

Throughout this project, we focused on creating an application that makes the tedious task of parsing checks faster and simpler. This is why we created **CalVision**, which uses multiple ML models (YOLO, YOLO Stacked, and Detectron) to detect the fields of your checks faster.

---

## 📊 **Data Exploration**

**You might consider describing the following (as applicable):**

* The dataset(s) used: [SSBI Dataset](https://github.com/dfki-av/bank-check-security): Contains 4,360 annotated check images of 19 different signers
* We used **pandas** and **numpy** to make dataframes and explore what might be missing or alarming in our dataset
* The biggest thing we noticed was how nested the information was (multiple annotations for one image) and the imbalance of genuine and forged signatures
* We decided to first see how well the models perform with the imbalance which surprisingly classified and identified text fields with high precision, recall, and accuracy. We also converted the COCO format of the dataset to YAML for the YOLO model to fit its expected format. The dataset was overall very clean which require minimal to no preprocessing

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images
* Nested Information
  * <img width="180" height="116" alt="image" src="https://github.com/user-attachments/assets/0078ce02-a20c-40f8-ac57-1a3e159c8f6e" />
* YOLO and YOLO Stacked Prediction Example
  * <img width="320" height="237" alt="image" src="https://github.com/user-attachments/assets/a2bddcdd-bc4d-44d1-a2fa-1a66a422391b" />
* Detectron Prediction Example
  * <img width="320" height="242" alt="image" src="https://github.com/user-attachments/assets/64d52d05-4cfd-4eab-bb9b-74b5135434ff" />
  
---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

- YOLO Model
  - Metrics:
      - Achieved 95% accuracy on forged signature classifications and 99% accuracy on genuine signature classifications, demonstrating high impact for Automation Anywhere.
      - Achieved 100% accuracy on identifying amount, amount_words, date and payee
      - Achieved 99.6% Precision and 99.3% Recall
  - Training setup:
    - Epochs: 200
    - Early Stopping ~ 136 epochs
    - Image Size: 640px 
    - Batch: 16
    - Learning Rate: 0.01

- YOLO Stacked (YOLO + ResNet)
  - Metrics:
      - Achieved 100% accuracy on identifying amount, amount_words, date and payee
      - Achieved 95% accuracy on classifying forged or genuine signatures
      - Achieved 99.6% Precision and 99.3% Recall
- Detectron
  - Metrics:
      - 68.3% accuracy on identifying amount, 75.8% accuracy on classifying genuine signatures
      - Most values fall below 0.5, meaning the model is often unsure between neighboring classes
      - AP (mAP): 52.9%
      - AP50: 76.8%
      - AP75: 64.9%
  - Training setup:
    - Training iterations: 2000
    - Learning rate: 0.00025
    - Batch size: 4


**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

* YOLO Model (applies to YOLO part of YOLO Stacked)
  * Loss
    * <img width="555" height="385" alt="image" src="https://github.com/user-attachments/assets/a1361d88-cfad-4a27-8d21-054ca9e1e2e2" />
  * Confusion Matrix
    * <img width="488" height="427" alt="image" src="https://github.com/user-attachments/assets/2c43d11b-c921-4177-aebf-e1c5fc0ce05d" />


* Detectron
  * Loss
    * <img width="488" height="294" alt="image" src="https://github.com/user-attachments/assets/456e2c41-6038-4868-9672-6578d92f4a9c" />
  * Precision-Recall Curve
    * <img width="488" height="297" alt="image" src="https://github.com/user-attachments/assets/e2f11902-ded7-460f-997c-1555009e4304" />
 
---

## 🚀 **Next Steps**

* If this were to be applied in a real world fast-paced setting, we would like to incorporate blurrier and overall more diversity in the photos as they might not be as clear as the ones in the dataset
* Include an option where users can interact with the model’s output to correct any mistakes
* Add security and compliance to make sure no sensitive information is being leaked

---

## ⭐ **Discussion and Reflection**

* Even though the YOLO model performed really well for identifying fields, it still needs a littel help for classification. We tried to help with this by having the YOLO Stacked but it didn't change too much. Some things that could help classification is tuning more hyperparameters and possibly more training time
* For Detectron2, it was able to perform strongly but could possibly be improved with more training iterations (possibly 50,000) but we couldn't because of resource limitations
* Learning how to train, evaluate and connect these models to an interface was a great learning experience on how to build end-to-end ML pipelines!


---

## 📄 **References** 

@inproceedings{khan2024enhanced,
  title={Enhanced Bank Check Security: Introducing a Novel Dataset and Transformer-Based Approach for Detection and Verification},
  author={Khan, Muhammad Saif Ullah and Shehzadi, Tahira and Noor, Rabeya and Stricker, Didier and Afzal, Muhammad Zeshan},
  booktitle={International Workshop on Document Analysis Systems},
  pages={37--54},
  year={2024},
  organization={Springer}
}

---

## 🙏 **Acknowledgements** 

**Thank you** to our AI Coach, **Swagath Babu** and our Challenge Advisors, **Vibhas Gejji** and **Joseph Lam**. And, a thank you to the Break Through Tech AI Program for this opportunity! 

The team learned so much about applying techniques to real-world projects and couldn't be more grateful!

The students who were a part of this project were Gurnoor Bola, Thy Tran, Margaret Galvez, Aaryan Hakim, Ricardo Varela Tellez, and Meera Vyas.
