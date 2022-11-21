# NAP
Next Argument Prediction

Step 1: download main.ipynb and open the file in google Colab

step 2: Clone the project to the colab working directory

step 3: main.ipynb
	This file contains the training, validation and testing phases step by step in colab notebook.
	
# Files:
1. kialo_GeneralDF.py
   This python code takes source path as input and provides a General dataFrame in ".pkl" and excel file format.
	INPUT: Accepts from user
      	>> "file Path" str: if there is only 1 input file/
		   "Folder path" str: if there are more input files in the folder.
      	>> "User prefered Dataset name" str: To differenciate the dataset generated from the given input.
	OUTPUT: Generates Argument pool DataFrame and save it in .pkl and excel file 
		  with the user specified Dataset name in the "/dataframe" directory.
      	>> Panda DF with ["index", "title", "position", "opinion", "argument"] Futures

2. GeneralDF_SentPairDF.py
   This python file takes as input the ".pkl" file generated by Kialo_GeneralDF.py in order to 
   create a proper dataframe format used for [NLP] Next Sentence Prediction.
	Input: accept from user
		>> "dataframe_path" str: The ".pkl" dataset file path that contains all the sentence pool.
		>> "dataFrame_name" str: The name user chooses to give to the paired sentence dataset.
	Output: Generates sentence pair DataFrame and save it in .pkl and excel file 
		>> DataFrame: The DataFrame that contains each Debate argument pair.
		   with Futures ["index", "label", "position1", "position2", "argument1", "argument2"]
		>> The position1 and position2 are included for inspecting the correctness of the pairing process/
		   they can be droped during the network training process.
3. output.py  I defined the classes in this file to use the new loaded model to predict the output independently.
4. Dataset folder contains files downloaded from kialo.com, [they are the raw data].
5. dataframe folder contains .pkl and excel files generated by kialo_GeneralDF.py and GeneralDF_SentPairDF.py
    Notice: the data frames without a transposed agrument alignmnet are in the without_Transpose folder.
6. models folder contains .pth files [the weight of the best model status].
7. results folder contains the prediction output of the model. 
8. Config.py file, this file is used to set the basic parameters of the model under the training or prediction process.
 
	
