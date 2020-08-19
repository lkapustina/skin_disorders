# README #

This skin application description

### Requirements ###

* python 3.6
* GPU with CUDA (preferably)
* installed Tensorflow


### Tensorflow installation guide ###

* https://alliseesolutions.wordpress.com/2016/09/08/install-gpu-tensorflow-from-sources-w-ubuntu-16-04-and-cuda-8-0/
* http://simonboehm.com/tech/2017/06/23/installingTensorFlow.html


### How to start ###

* apt-get install -y python3-pip python3-dev build-essential python3-venv
* python3 -m venv
* source venv/bin/activate
* pip install -r requirements.txt


### Scrapping data ###

* python skin_diseases_scraper.py
* some scrapped data to run notebook: https://drive.google.com/open?id=1kKk7NKF1PaxFwNVPugU589-UOOJmHb6J


### Training model ###

* python train_cnn.py


### Notebook for demo ###

* skin recognition.ipynb


### data ###

data.zip - unzip to folder skin_disorders/data

https://drive.google.com/open?id=1kKk7NKF1PaxFwNVPugU589-UOOJmHb6J