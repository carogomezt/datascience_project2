# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation<a name="installation"></a>
1. Clone the repository.
2. Create a virtual environment.
```
$ virtualenv --python=python3  ds-project2 --no-site-packages 
$ source ds-project2/bin/activate  
```
3. Go to the project folder (datascience_project2) and run the following command to install all the dependencies:
```
$ pip install -r requirements.txt  
```
## Project Motivation <a name="motivation"></a>
For this project was used data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.
With this information I was able to put into practice ETL skills, and the creation of ML Pipelines. 

## File Descriptions<a name="files"></a>

1. [app](https://github.com/carogomezt/datascience_project2/tree/main/app): Folder with html files and code to run the API.
2. [data](https://github.com/carogomezt/datascience_project2/tree/main/data): Folder with the file to make the preprocessing of the data.
3. [data_analysis](https://github.com/carogomezt/datascience_project2/tree/main/data_analysis): Folder with the Jupyter Notebooks used to make the initial exploration of the data and models.
4. [img](https://github.com/carogomezt/datascience_project2/tree/main/img): Folder with images of the results.
4. [models](https://github.com/carogomezt/datascience_project2/tree/main/models): Folder with the file to make, train and evaluate the model.
5. [README.md](https://github.com/carogomezt/datascience_project2/blob/main/README.md): File with repository information.
6. [requirements.txt](https://github.com/carogomezt/datascience_project2/blob/main/requirements.txt): File with requirements of the project.

## Instructions<a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results<a name="results"></a>

In the step to build the model I tried different models, and they give me the following results:
- RandomForest score: 81%
- DecisionTrees score: 79%
- KNeighborsClassifier score: 26%

![models accuracy](https://github.com/carogomezt/datascience_project2/blob/main/img/models_accuracy.png "Models accuracy")
  
You could see more detailed information on the [jupyter notebook](https://github.com/carogomezt/datascience_project2/blob/main/data_analysis/ML%20Pipeline%20Preparation.ipynb).

I choose the RandomForest model because it had the highest score, and I tried to optimize this model with the GridSearch. It took more than 10 hours to train and the score descended to 21%.
For that reason I decided to choose the model before applying the GridSearch.
I couldn't upload the model because the size of the file was around 1Gb.

Some target classes didn't have different values (all values were 0) and that could make that the model couldn't generalize in a better way.


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Stack Overflow for the data.  You can find the Licensing for the data and other descriptive information [here](https://insights.stackoverflow.com/survey).  Otherwise, feel free to use the code here as you would like! 