# flower_cnn
this project comes with a pretrained model

so after installing 
you can either train or test the model in which ever order you like

## Installing 
```bash
sudo -H pip install -U pipenv
git clone https://github.com/grantozz/flower_cnn
cd flower_cnn
pipenv install 
```

## training the model
```bash
#activates environment 
pipenv shell
#run train script 
python src/flower_cnn.py
```

## testing the model
```bash
#activates environment 
pipenv shell
#run test script 
python src/test.py flower_model_v4.0.6.h5 flower_model_v4.0.6_labels.txt
```
