# GreenABR-MMSys22 - Docker image instructions
The image includes all libraries for training and evaluations. Thus, you do not need to run setup.py scripts at any repository. 

### To import the image 
```
docker import \                                        
--change 'CMD ["bash"]' \
greenabr.tar greenabr:local
```

### To run the container 
```
docker run -it --name gabr greenabr:local 
```
After the new container is created and started, source the "bash_profile"
```
source ~/.bash_profile
```
### Code repository
The code repository is already included in the image under the home folder, "GreenABR-MMSys22". You can follow the instructions in the main repository for the power model, qoe model, and GreenABR training.  
