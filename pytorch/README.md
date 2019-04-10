# Autoregressive Energy Machines - PyTorch
```cd``` to the ```pytorch``` directory and run
```
python <experiment>.py --<args>
```
where ```<experiment>``` is one of 
- ```plane```
- ```face```
- ```uci```
to run the corresponding experiment with default settings. The ```plane``` script includes the spirals, checkerboard, and diamond experiments, the ```face``` script corresponds to the Einstein experiment, and the ```uci``` script includes all UCI experiments, as well as BSDS300. All experimental details are outlined in the paper's supplementary material, and can be specified using ```<args>```.

Training metrics and model checkpoints can also be saved at specified intervals. Intermediate visualization is also available for the ```plane``` and ```face``` tasks.
