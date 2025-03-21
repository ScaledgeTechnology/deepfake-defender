# **Deepfake Defender**

**Deepfake Defender** is revolutionizing the way media is created, using artificial intelligence to generate hyper-realistic videos, images, and audio that can be nearly impossible to distinguish from reality. While this innovation brings incredible creative opportunities, it also introduces serious risks—misinformation, identity fraud, and privacy breaches are just a few of the challenges. At Detect AI, we are committed to staying ahead of this rapidly evolving technology, providing powerful tools that help users detect and prevent malicious deepfake content. Our goal is to ensure digital authenticity, giving you the confidence to trust what you see and hear in an age where manipulation is just a click away.

---
### **Demo Video**

<a href="https://youtu.be/d5mOhsKQorM">
    <img src="https://github.com/ScaledgeTechnology/deepfake-defender/blob/main/demo_gif_deepfake.gif" alt="Deepfake Defender Demo" />
</a>

*Watch the full video on [YouTube](https://youtu.be/d5mOhsKQorM)* 
<a href="https://youtu.be/d5mOhsKQorM">
    <img src="https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png" alt="YouTube Logo" width="17" style="vertical-align: middle;" />
</a>

or

*Watch the full video on [Google Drive](https://drive.google.com/file/d/1PtNili2XQftArzNvCV1ZNlb5GTTSsRaI/view?usp=sharing)* 
<a href="https://drive.google.com/file/d/1PtNili2XQftArzNvCV1ZNlb5GTTSsRaI/view?usp=sharing">
    <img src="https://ssl.gstatic.com/images/branding/product/1x/drive_2020q4_48dp.png" alt="Google Drive Logo" width="17" style="vertical-align: middle;" />
</a>








---

## **Installation Steps**

Follow these steps to set up and run the project on your local machine.

### **1. Clone the Repository**
1. Open any directory of your choice on your local system.  
2. Launch **Git Bash** or **Terminal** in that directory.  
3. Run the following command to clone the repository:  
   ```bash
   git clone git@github.com:ScaledgeTechnology/deepfake-defender.git
   ```
   or if shows error do using https-
   ```bash
   git clone https://github.com/ScaledgeTechnology/deepfake-defender.git
   ```
   You have successfully cloned the project repository.
---


### **2. Set Up the Django Project**
- `Note:` Your system should have **any Python version between 3.10 and 3.12.6** installed.
If your Python version is not in this range, please install it first from the official [Python website](https://www.python.org/downloads/).  

- You can set up the project using **either** of the following methods:
---

### **I. Direct Setup Using the .exe File**
- Navigate to the **deepfake-defender** folder that you have cloned on your system.  
- Inside this folder, you will find a file named **django_app.exe**.  
- **Double-click** on `django_app.exe` — this will automatically handle the entire setup process.  
- After a short while, you will be redirected to the project's webpage.

---

### **II. Manual Setup Using the Terminal**
#### **Step 1: Make sure you are inside the root folder**
If not, open the terminal and navigate to the deepfake-defender folder:  
```bash
cd your_path_to_the_project/deepfake-defender
```
   
#### **Step 2: Create a Virtual Environment**
Inside the root folder, create a virtual environment by running:  
```bash
python -m venv .venv
```

#### **Step 3: Activate the Virtual Environment**
Activate the virtual environment using the following command:  

- For **Windows**:  
  ```bash
  .venv\Scripts\activate
  ```  
  _(You’ll see the terminal prefix change to `(.venv)`.)_

- For **Mac/Linux**:  
  ```bash
  source .venv/bin/activate
  ```

#### **Step 4: Upgrade pip**
Upgrade `pip` to the latest version:  
```bash
python.exe -m pip install --upgrade pip
```

#### **Step 5: Install Project Dependencies**
Install all the required dependencies using the `requirements.txt` file:  
```bash
pip install -r requirements.txt
```

---


### **3. Run the Project**

#### **Step 1: Go inside project directory from root directoryr**
- `deepfake-defender` is the root directory
- `deepfake` is the project directory

   ```bash
   cd deepfake
   ```

   So make sure you are inside-
   ```bash
   your_path/deepfake-defender/deepfake
   ```
   
- Ensure you are in the **root `deepfake` folder** (the first `deepfake`, not the nested one).  

#### **Step 2: In terminal start the Django Development Server**
  
```bash
python manage.py runserver
```

---

### **5. Access the Application**
Once the development server is running, a link will be displayed in the terminal (something like `http://127.0.0.1:8000`).  

- **To view the application**, either:  
  - Hold **Ctrl** and click the link, or  
  - Copy and paste the link into your browser.

---

### **Additional Notes**
- Ensure your virtual environment is always activated when working on the project (`(.venv)` will appear in your terminal).  
- Follow all steps carefully to avoid configuration issues.


---
