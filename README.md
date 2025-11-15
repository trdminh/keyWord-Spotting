# Key word Spotting

### 1. Record voice use esp32xiao seeed s3 studio

Set up your label and use this repo to record voice with sd card and esp32s3 xiao seeed [click here](https://github.com/Mjrovai/XIAO-ESP32S3-Sense/blob/main/Wav_Record_dataset/Wav_Record_dataset.ino).

![image](https://camo.githubusercontent.com/4f4e51f715e6817f3ae106cf7bc9cd1c465e09bfc855ce7cef3f12dc9063ef61/68747470733a2f2f66696c65732e736565656473747564696f2e636f6d2f77696b692f536565656453747564696f2d5849414f2d455350333253332f696d672f36362e6a7067)

### 2. Training model with edge impulse

![image](https://camo.githubusercontent.com/0f624eb8fdcd49e0cd967565bd4ff1e5088fad0a26182d403b457613f190411f/68747470733a2f2f66696c65732e736565656473747564696f2e636f6d2f77696b692f7869616f657370333273335f6b77732f312e706e67)

##### Step 1:
Upload dataset and set up label for this data from your device to Edge Impulse.

<img width="1830" height="924" alt="image" src="https://github.com/user-attachments/assets/9b917013-839b-4c8e-848f-dba4d8ed0a6f" />

Set up this for training dataset.

##### Step 2:
Create impulse for our model

<img width="1828" height="921" alt="image" src="https://github.com/user-attachments/assets/4e28e025-ffca-4c19-9edd-1fbd6b866152" />

##### Step 3:

Feature extraction with MFCCs
1. Pre-emphasis
    A pre-emphasis filter is applied to boost high frequencies:
        𝑦[𝑛]=𝑥[𝑛]−0.98𝑥[𝑛−1]
2. Framing
    With a frame length of 20 ms and a sampling rate of 16 kHz:
        N=0.02×16000=320 samples

    The frame stride is also 20 ms, so the hop size is:
        H=320

    Each frame is defined as:
        yk(n)=y(n+kH)
3. Windowing (Hamming Window)
        w(n)=0.54−0.46cos(2πn/319)
    Windowed frame:
        xw(n)=yk(n)⋅w(n)

<img width="705" height="879" alt="image" src="https://github.com/user-attachments/assets/dc235e60-2e57-484f-8b92-4c3f84ed3bdf" />

### Step 3: Training model
Set up neural netword, batch size and learning rate for model in Edge Impulse and see the accurancy of our model.
<img width="1836" height="921" alt="image" src="https://github.com/user-attachments/assets/88375bec-d9a9-4dec-85eb-58489cf7c48c" />

### Step 4: Deploy model

Build and deploy this model with Arduino Library. This is true for esp and arduino when you use Arduino IDE or platform.io .
If you use different platform, you can select. When don't have platform you need, You build to C/C++ library and write CMake and Makefile to link this library with your project.
<img width="1831" height="920" alt="image" src="https://github.com/user-attachments/assets/da7fd0e9-ed40-4149-84ba-5da7735af306" />


