# natural-image-classifier-flask
<p>This project is an image classifier web application  developed in Python, Flask web framework using natural images dataset from kaggle. It uses Convolution Neural Networks to train and classify the given images based on dataset given. </p>
# Pre-Requisities: <br/>
1.Python installed <br/>
2.Pycharm IDE <br/>
3.Install Flask <br/>
&emsp; pip install -U Flask <br/>
4.Download Dataset from <a href="https://www.kaggle.com/datasets/prasunroy/natural-images">Kaggle Natural Images Dataset</a>.

# Requirements:
certifi==2021.10.8 <br/>
charset-normalizer==2.0.12 <br/>
gunicorn==20.1.0 <br/>
idna==3.3 <br/>
numpy==1.22.3 <br/>
Pillow==9.0.1 <br/>
requests==2.27.1 <br/>
torch==1.11.0+cu113 <br/>
torchaudio==0.11.0+cu113 <br/>
torchvision==0.12.0+cu113 <br/>
typing_extensions==4.1.1 <br/>
urllib3==1.26.9 <br/>

# Deployment(Local):

![image](https://user-images.githubusercontent.com/39313346/166704278-4979488b-2133-44d0-a2b4-67feba2347f8.png)

# Steps to Deploy the Application:

1. Create a folder named data as shown above and move the downloaded dataset to that folder <br/>
2. Run the program with Pycharm(May take a longer time for the page to load as the model is training for the first request) <br/>
3. The app will be deployed at your local host at assigned port as shown in your pycharm console. <br/>

# Run Application:
1. Upload Image and click predict to see model's prediction for uploaded image.





