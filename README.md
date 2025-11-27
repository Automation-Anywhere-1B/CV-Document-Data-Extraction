# CalVision: __CV-Document-Data-Extraction__

Welcome to CalVision! This was made from the Automation Anywhere 1-B Team to help automate the parsing of financial checks. 

### Our Goal
Throughout this project, we focused on creating an application that makes the tedious task of parsing checks faster and simpler. This is why we created **CalVision**, which uses multiple ML models (YOLO, YOLO Stacked, and Detectron) to detect the fields of your checks faster. 

### Intallation Guide
1. In order to run this app locally, please run 

    ```pip install requirements.txt```

    to install at the necessary libraries. 

2. After installing the libraries, in your **root** directory, run

    ```streamlit run streamlit/app.py```

    to open the app on your local device. 

### User Guide
In order to submit your photo, please make sure to follow these simple instructions:

1. Position your check in the center of your camera

2. Make sure you have good lighting and the **_whole_** check is visible

3. Confirm that your check is in **_focus_** and there are no other objects in the photo

4. You are ready to submit!

5. Upload your document, then click **_Run All Models_** to compare YOLO, YOLO Stacked, and Detectron2.


### Credits
**Special Thanks!**:

**Thank you** to our AI Coach, **Swagath Babu** and our Challenge Advisors, **Vibhas Gejji** and **Joseph Lam**. And, a thank you to the Break Through Tech AI Program for this opportunity.

The students who were a part of this project were Gurnoor Bola, Margaret Galvez, Aaryan Hakim, Ricardo Tellez, Thy Tran, Meera Vyas, and Priya Gadhe


