# flower_cnn
## Installing 
```bash
sudo pip install pipenv
git clone https://github.com/grantozz/flower_cnn
cd flower_cnn
pipenv install 
```

## training the model
```bash
#activates enviorment 
pipenv shell
#run train script 
python src/flower_cnn.py
```

## testing the model
```bash
#activates enviorment 
pipenv shell
#run test script 
python src/flower_cnn.py flower_model_v4.0.6.h5 flower_model_v4.0.6_labels.txt
```
