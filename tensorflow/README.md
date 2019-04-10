# Autoregressive Energy Machine - Tensorflow
Tested on Tensorflow 1.12

### Training a model
To train a model on the UCI or BSDS300 datasets run
```
python train_AEM_UCI_data.py --<args>
```
To train on 2D synthetic datasets run:
```
python train_AEM_2D_data.py --<args>
```

Required arguments are 
- ```--model_name``` - used to name log directories
- ```--dataset``` - one of ```{'power', 'gas', 'hepmass', 'miniboone', 'bsds300'}``` for ```train_AEM_UCI_data.py``` or ```{'gaussian_grid', 'two_spirals', 'checkerboard', 'einstein'}``` for ```train_AEM_2D_data.py```

For instance  ```python train_AEM_2D_data.py --model_name=my_model --dataset=power``` will train a model with default settings on the Power dataset, and will save logs and checkpoints using the name ```my_model```. Additional arguments with hyperparameters and training options are explained in ```train_AEM_UCI_data.py``` and ```train_AEM_2D_data.py```.

Log directories with tensorboard summaries will be created automatically in the training script. For 2D datasets, plots will be created in the log directory. 

### Evaluating a model trained on UCI or BSDS data
Evaluation of a trained model with large numbers of importance samples can be performed with
```
python eval_AEM_UCI_data.py --<args>
```
Required arguments are again
- ```--model_name``` - name of the trained model
- ```--dataset``` - dataset the model was trained on

To evaluate the model trained above with default settings use ```python eval_AEM_2D_data.py --model_name=my_model --dataset=power```. This will write a results file to the model log directory.
