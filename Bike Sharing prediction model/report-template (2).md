# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Adenike Adewumi 

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
TODO: Add your explanation
I noticed that I had to create new variables from the samplesubmission.csv file each time before replacing the values with my predictions
I also noticed that negative values were not allowed so I had to clip negative values to zero

### What was the top ranked model that performed?
TODO: Add your explanation
kNeighborsDist

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
TODO: Add your explanation
I went to check the meaning of each feature on kaggle and also created the correlation of all the features and to see how they are related to each other
I created a new feature of the difference between the felt temp(atemp) and the actual temp.
I created a categorical feature of is_windy for when windspeed is greater than 30. This is because at this windspeed it isn't ideal to bike in this condition
I created a ratio of temp and humidity because they are inversely related
Changed all categorical feature to type "category"
Extracted hour, dayofweek,month, and isweekend from datatime feature. This is to se whether work, or school periods, nighttime will affect classification

### How much better did your model preform after adding additional features and why do you think that is?
TODO: Add your explanation
It did mich better. I think this is because it had more data to work with and these are features added after checking the conditions that affect the target feature. Using both atural knowledge and data driven insights
## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
TODO: Add your explanation
Not so better actually. I wanted to try it again but my colab GPU ran out of use

### If you were given more time with this dataset, where do you think you would spend more time?
TODO: Add your explanation
I would spend some time in the EDA then majorly in hyperparameter tuning. This is because I want to fully understand the consequences of each tuning done

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|hpo1|hpo2|hpo3|score|
|--|--||--|--|
|initial|default vals|default vals|default vals|1.86412|
|add_features|default vals|default vals||0.50577|
|hpo|XGB n_estimators [100], max_depth [10], learning_rate [0.1], subsample [0.8], colsample_bytree [0.9]|GBM num_boost_round [default], extra_trees [False] (default model)
GBM num_boost_round [300], extra_trees [True] (extra trees variant with more boosting rounds)|CAT Default|0.49587|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

TODO: Replace the image below with your own.

![model_train_score.png](https://drive.google.com/file/d/1jnc9kp60mVCLwC8eSvGYJ4JaiJ0q9Dqw/view?usp=drive_link)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

TODO: Replace the image below with your own.

![model_test_score.png](https://drive.google.com/file/d/1zwVbzmkLttCfZQfmoFdkl93ef9eDir-i/view?usp=drive_link)

## Summary
TODO: Add your explanation
In summary, adding additional features improved the Kaggle score because it allowed the model to better capture important patterns in the data. This led to lower errors and higher accuracy, which directly reflected as an improved Kaggle score.
Different hyperparameter values can affect how well the model fits the training data and generalizes to new data. In the context of Kaggle competitions, tuning hyperparameters often leads to a direct improvement in the leaderboard score because the model becomes more optimized for the specific dataset and prediction task.
Percentage Increase= (1.86412-0.49587)/(1.86412)* 100= 72.23%