
# MODEL'S ARCHITECTURE FINETUNING LIFECYCLE

The Challange is to build a efficient architecture which adhires to the condition below:

    1) Parameter should be less than 10k
    2) Minimum test Accuracy 99.4 in the last epochs
    3) Has to complete training under >=15 epoch
    4) Explain the whole lifecycle and steps you choose to get the accuracy.





 - [Notebook Link](https://github.com/darshanvjani/Extensive-Vision-AI-Program-EVAI6-/blob/main/4_Basics_of_Architecture/MNIST_custom_architecture.ipynb)


## Model Building Lifecycle

| Model | Parameters | Best Train Acc. | Best Test Acc. | Brief Steps                                                           |
|-------|------------|-----------------|----------------|-----------------------------------------------------------------------|
| 1     | 6.3M       | 99.99           | 99.24          | Code Stepup, No emphasis on Architecture!                             |
| 2     | 51k        | 99.45           | 98.90          | Set the basic skeleton, No Fancy Stuff!                               |
| 3     | 12k        | 98.96           | 98.45          | Testing the skeleton's capacity by reducing the params.               |
| 4     | 12.2k      | 99.98           | 99.35          | Added BatchNorm(regularization tech.)                                 |
| 5     | 12.2k      | 98.98           | 99.36          | Added Dropout(regularization tech.)                                   |
| 6     | 9.8k       | 96.7            | 99.36          | Added Global Average Pooling(GAP)                                     |
| 7     | 12.5k      | 98.97           | 99.45          | Changed RF structure from 7 to 5 in the first conv block              |
| 8     | 12.5k      | 99.25           | 99.35          | Added Data Augmentation                                               |
| 9     | 9.2k       | 97.76           | 99.41          | Added Further Augmentation, Reduced parameters, added Cyclic Learning |


## Analysis

## 1.) CODE SETUP:

TARGET:

1. Get the set-up right
2. Set Transforms
3. Set Data Loader
4. Set Basic Working Code
5. Set Basic Training & Test Loop
6. Results:
    1. Parameters: 6.3M
    2. Best Training Accuracy: 99.99
    3. Best Test Accuracy: 99.24
7. Analysis:
    1. Extremely Heavy Model for such a problem
    2. Model is over-fitting, but we are changing our model in the next step

## 2.) BASIC SKELETON

TARGET:

1. Get the basic skelton right. No fancy stuff.
2. Result:
    1. Parameters: 51k
    2. Best training accuracy: 99.45
    3. Best test accuracy: 98.89
3. Analysis:
    1. The structure of the model is good.
    2. However the model is still overfitting
    3. the model is still very large

## 3.) LIGHTER MODEL

TARGET:

1. lets test the skeleton of model by making it lighter.
2. Will reduce the number of parameters and see if it can able to learn with the same ability.
3. Result:
    1. Parameters: 12k
    2. Best training accuracy: 98.96
    3. Best test accuracy: 98.45
4. Analysis:
    1. The structure of the model is great.
    2. The difference between Train and test accuracy is great
    3. The architecture is capable if pushed further.
    
    ## 4.) ADDING BATCH-NORMALIZATION
    
    TARGET:
    
    1. We need to add something which can sorta change the way the model is behaving without changing the parameters.
    2. Will introduce batch normalization
    3. Result:
        1. Parameters: 12.2k
        2. Best training accuracy: 99.98
        3. Best test accuracy: 99.35
    4. Analysis:
        1. Improved Training and Testing performance.
        2. However after some epoch overfitting kicked in
        3. Need to add the right kind of regularization.
    
    ## 5.) ADDING (THE RIGHT) amount of REGULARIZATION
    
    TARGET:
    
    1. Add regularization, dropout
    2. Result:
        1. Parameters: 12.2k
        2. Best training accuracy: 98.98
        3. Best test accuracy: 99.38
    3. Analysis:
        1. Improved the gap between Training and Testing accuracy.
        2. However its still not able to hit the desired accuracy.
        3. Need to add some more capability to the model while keeping the parameters low.
    
    ## 7.) ADDING Global Average Pooling(GAP)
    
    TARGET:
    
    1. Add GAP so that total number of params decreases and then add more parameters in the above layers.
    2. Result - 1(After Adding GAP):
        1. Parameters: 9.8k
        2. Best training accuracy: 96.7
        3. Best test accuracy: 99.36
    3. Result - 2(After Adding GAP and increasing the accuracy)
        1. Parameters: 11.2k
        2. Best training accuracy: 96.7
        3. Best test accuracy: 99.28
    4. Analysis:
        1. Not a lot of difference, I think i need to play with the receptive field of the first block, instead of putting maxpooling at 7RF, let me try putting MP at 5RF.
        2. No Overfitting
        3. Need to add some more capability to the model while keeping the parameters low.
    
    ## 8.) CHANGE THE RF STRUCTURE
    
    TARGET: 
    
    1. To change the RF structure of the first block.
    2. Results:
        1. Parameters: 12.5k
        2. Best training accuracy: 98.97
        3. Best test accuracy: 99.25
    3. Analysis:
        1. The model is performing great the skelaton is great, the regularization is just awesome.
    
    ## 9.) INCREASING THE CAPABILITY EVEN FURTHER BY ADDING IMG AUGMENTATION
    
    TARGET:
    
    1. See the DS we can say, that the dataset as some images which are tilted in one day or the other by a little degree. So that we will add some rotation to the images.
    2. Results:
        1. Parameters: 12.5k
        2. Best Training accuracy:  99.45
        3. Best Testing accuracy: 99.35
    3. Analysis:
        1. We were able to reach the optimum accuracy needed!
        2. However we need to reduce the parameter even more and bring it down to 10k
    
    ## 10.) MORE ADVANCE IMAGE AUGMENTATION
    
    TARGET:
    
    1. Add more image augmentation techniques like random affine, color jitter and rotation
    2. Reducing the number of parameters
    3. Results:
        1. Parameters: 9.2k
        2. Best Training accuracy:  97.76
        3. Best Testing accuracy: 99.41
    4. Analysis:
        1. Success!

### Final Model Architecture: 

![App Screenshot](https://github.com/darshanvjani/Extensive-Vision-AI-Program-EVAI6-/blob/main/5_Architecture_Finetuning(CodingDrillDown)/images/best_model_architecture.PNG?raw=true)

 - [FINAL MODEL](https://github.com/darshanvjani/Extensive-Vision-AI-Program-EVAI6-/blob/main/5_Architecture_Finetuning(CodingDrillDown)/Architecture_Finetuning(Best_Model).ipynb)
