# GreenABR-MMSys22
GreenABR is an energy aware adaptive bitrate streaming model designed with deep reinforcement learning. 

### Prerequisites
```
python                 3.7.3
Keras                  2.3.1
numpy                  1.16.4
matplotlib             3.1.0
pandas                 0.24.2
joblib                 1.0.1
tensorflow             1.14.0
scikit-learn           0.21.2
```

### Power Model 
Streaming power measurements are stored under /power_model/dataset folder. The measurements are aggregated for each second. Galaxy S4 dataset is used for training the model and XCover Pro device is used for testing. To train the power model and evaluate use ``python train.py`` under **power_model** folder.
```
python train.py
```

### GreenABR Training 
To train GreenABR, first you need to train the power model and copy the saved model to **"training"** folder. "training" folder includes network traces(cooked_traces) and "power_attributes.csv" file. Use "rep_6" folder to train for 6 representations case and "rep_10" for 10 representations. Each folder has the training script and resource files for training. 
```
python GreenABR.py
```
The default script is set to have 30000 iterations and we found 9000 iterations to be optimal to avoid overfitting. Training script saves the model at every 1000 iterations to "savedModels" folder. 

### GreenABR Evaluation
To evaluate GreenABR, first copy the saved model for each representation set to the corresponding folder. For rep_10_exp2, you can use the same model with rep_10. All source files for evaluations are given under "powerMeasurementFiles", "evaluationFiles", "test_sim_traces" folders. Power model and power attributes files should also be given for power model use. 
To train for any representation set, run ``python evaluate.py`` under the corresponding folder. 
```
python evaluate.py
```
#### Plotting Graphs
GreenABR is compared several SOTA works for evaluations. Bola, Bolae, Dynamic-Dash, Dynamic ABR, Throughput Rule from the [sabre](https://github.com/umass-lids/sabre) environment and [Pensieve](https://github.com/hongzimao/pensieve) with its own environment. The results from these environments should include VMAF values for every selected chunk along with other streaming values such as rebuffering time. The logs for these algorithms are included in test_results folder. 
To plot the graphs for each representation set, run the following under the corresponding folder:
```
python create_summary_results.py
python plot_graphs.py
```
All figures are saved to the "plots" folder in the same directory. 

### QoE Model 
To train and find the coefficients of our proposed QoE model, [THE WATERLOO STREAMING QUALITY-OF-EXPERIENCE DATABASE-III](https://ieee-dataport.org/open-access/waterloo-streaming-quality-experience-database-iii) is used. The required files for training are given in "data.csv" file and "streaming_logs" folder.
To train and find the coefficients:
```
python train.py
```
To compare with other QoE models:
```
python compare_models.py
```
