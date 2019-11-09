# WI_PROJECT
First Web Intelligence project of team MIRA studying in IG5 at Polytech Montpellier


##Set up
Begin by cloning the project :

```
git clone https://github.com/Remigestin/WI_PROJECT.git
```

Then you need to put your testing data on the root project. 
Be careful your data should be named : "testData.json"


###Build a new model
A model already exists on the project (it's the "spark-lr-model folder"). 
When the program is running, this model is load and so used.

If you want to change this model, you need to :
- remove the "spark-lr-model" folder
- put your training data on the root of the project. Those data should be named : "trainingData.json"
And so, when you will run the program it will first build the new model and then run predictions.


##Run
After everything is set up, just run on shell : 
```
sbt run
```


##Result
After running the program, on the root of the project you will find a folder named "result". 
Inside, the result csv is contained in a folder named : "predictions_[date]_[time]". 
This allowed you to keep a track of every run.

The csv represent your testing data with a new column on the beginning named "predictedLabel". 
As you may guess this column gathers the predictions made on your test data (true or false). 

