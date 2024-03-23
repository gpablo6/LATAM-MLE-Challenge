## Model Selection
We do not see any significant difference between the best models of each type. As of right now the dataset is quite small and not many work has been put in place, so, from a DS perspective the LogisticRegression could be the best choice, however, in a real world enviroment, the task of a MLE would be to also consider  other factors that might affect the operationalization and maintanability of the pipeline down the road, which could include things like the following:
1. Even though some EDA was executed, in the future there could be more external
factors computed into the equation, so the model should be able to handle more complex relationships in the future.
2. The tuning for the models is quite simple, but as the data could become more complex also the tunning might require more parameters to try out, which the booster might provide an upper hand in that sense.
3. Right now we are only working on the premise of one airport, this model could also be extended to other airports, which might introduce additional uncertainty and variables, so for easing up the development of those said models we should consider something more standard instead of handling different types of models.

Thus with the three previous statements, the best choice would be kickstarting at once with the XGBoost as will allow us for handling a more wide set of scenarios and arragemenets without spending that much time in the development and/or retraining of the model(s). In addition, for leaving open room for further improvements and experiments, we will make use of parametrization for values like the data imbalacing so we include it in the current pipeline but further changes might not disrupt the CI/CD process of future releases.

## Model Deployment
I decided to attemp SageMaker endpoints as I'm more familiared with it.
The steps would be the following:
1. Having a training script, we will trigger that one every time we detect a change in the `model.py` module. Or a flag is given for retraining.
    * This script should handle the creation of the new training job.
2. Having a deployment script which will push the new image and attach the training job to the new endpoint.

All this steps are meant to be orchestrated trough the github actions workflow, we might need to terraform the ECR repository for keeping track of the containers, and the remaining piece of the code can be automated with boto3 and sagemaker sdk using python.
