# 🧁 Build a Machine Learning Workflow for Scones Unlimited on Amazon SageMaker

This project is part of the **Udacity AWS Machine Learning Engineer Nanodegree Program**.

---

## 🧠 Project Overview

In this project, I built and deployed an **image classification model** using **Amazon SageMaker** to help a fictional logistics company, **Scones Unlimited**, identify the type of vehicle their delivery drivers use, specifically, whether it’s a **bicycle** or a **motorcycle**.

Image classification models are an essential part of **computer vision**, widely used in industries ranging from **autonomous vehicles** and **augmented reality** to **eCommerce** and **diagnostic medicine**. For Scones Unlimited, this model enables smarter logistics management and improved operational efficiency.

By automatically identifying drivers’ vehicles, the company can:
- Assign nearby orders to **bicyclists** for quicker delivery.
- Assign longer routes to **motorcyclists** for better efficiency.
- Optimize routing, safety, and delivery speed.
- Scale operations with automated ML-driven decision-making.

---

## ⚙️ Technologies Used

- **Amazon SageMaker** — Model training, deployment, and hosting
- **AWS Lambda** — Serverless backend services for invoking model endpoints
- **AWS Step Functions** — Workflow automation and orchestration
- **AWS S3** — Data storage and staging
- **AWS CloudWatch** — Model monitoring and logging
- **Python** — Data processing and model scripting
- **Jupyter Notebook** — Experiment tracking and visualization

---

## 🚀 Project Steps Overview

### **Step 1: Data Staging**
- Collected and prepared the dataset of vehicle images.
- Uploaded data to **Amazon S3** for accessibility by SageMaker.
- Ensured images were properly labeled as *bicycle* or *motorcycle*.

### **Step 2: Model Training and Deployment**
- Built and trained a **Convolutional Neural Network (CNN)** for binary classification.
- Used **SageMaker built-in training jobs** for model experimentation.
- Deployed the trained model as a **real-time inference endpoint**.

### **Step 3: Lambda Functions and Step Function Workflow**
- Created **AWS Lambda functions** to:
  - Invoke the SageMaker model endpoint.
  - Process inference results and trigger workflow steps.
- Designed an **AWS Step Functions workflow** to coordinate:
  - Image input → Model prediction → Result processing → Output.

### **Step 4: Testing and Evaluation**
- Validated endpoint predictions with a held-out test set.
- Evaluated model accuracy and response latency.
- Confirmed the end-to-end pipeline functioned correctly.

### **Step 5: Optional Challenge**
- Explored additional optimizations for scalability and monitoring.
- Implemented optional CloudWatch metrics for endpoint health and performance.

### **Step 6: Cleanup Cloud Resources**
- Deleted deployed endpoints, Step Functions, and S3 objects to avoid unnecessary costs.

---

## 📈 Results and Insights

- Achieved a **high accuracy** in distinguishing between bicycles and motorcycles.
- Successfully deployed an **event-driven ML pipeline** on AWS.
- The workflow can easily be extended to handle multi-class classification or integrated into a larger logistics platform.

---

## 🧩 Key Learnings

- Building **end-to-end ML pipelines** using **AWS cloud services**.
- Deploying and integrating ML models in **serverless environments**.
- Using **Step Functions** for automated, scalable ML workflows.
- Applying **best practices** for model monitoring, versioning, and cleanup.

---

## 🏁 Conclusion

The **Scones Unlimited** project demonstrates how to:
- Build, train, and deploy scalable image classification models on AWS.
- Connect multiple AWS services (SageMaker, Lambda, Step Functions) to create a fully automated ML workflow.
- Deliver production-ready ML solutions that can adapt to real-world business needs.

