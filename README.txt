# Machine-Learning-Dengue-Case-Prediction-in-Driven-Data-competition
Prediction of average Dengue cases in a week of a year using ensemble models.
Project: DengAI: Predicting Disease Spread
---------------------------------------------------------------------------

Dataset : https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/page/80/) 

---------------------------------------------------------------------------

Code is developed using Python 3.6
Libraries used: Scikit-Learn, Keras
Development Environment & IDE: Anaconda Navigator, Spyder

---------------------------------------------------------------------------

Script name: Preprocessing_Code.py
Input argument format : <Raw Data set> <Processed Data set> 

Example: 
"training_data_features.csv" "ImputedV_Dataset.csv"

This code is used for Pre-processing the raw data set.

---------------------------------------------------------------------------

Script name: Final_code.py            
Input argument format : <Training Data set> <Test Data set> <Output file> 

Example:
"ImputedV_Dataset.csv" "Prep_Test.csv" "Output_file.csv"

This code contains python code for Gradient Boosting, Random Forest and K-NN

---------------------------------------------------------------------------

Script name: ANN_Keras.py
Input argument format : <Training Data set> <Test Data set>

This code contains python code for Artificial Neural Network using Keras.
