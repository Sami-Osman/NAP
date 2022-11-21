import os
import sys
import re, string, nltk
import pandas as pd


# This class is used to manage the data files downloaded from "kialo.com" contained in a .txt dataset file in order to create a usable dataframe
class load_data:
  def __init__(self, dataset_name = 'Arg_dataset'):

    '''
      A class to manage the text file downloaded from "Kialo.com" website which contains the Debates dataset.

      Attributes
      ----------

      dataset_name: str
        User defined Name for the dataset based on the content of the Debates

      index: int
        index for every Debate or argument extracted from the file.

    '''

    self.dataset_name = dataset_name
    self.idx = 0

  @classmethod
  def get_files_name(self, folder_path):

    '''
      Static method which gets all the file names from the specified folder

        Parameters
        ----------
        folder_path: str
          The path of the folder containing the Debate files downloaded from "Kialo.com"

        Returns
        -------
        files_name: list
          all avilable file names in the specified folder path.

    '''  

    files_name = []
    folder = os.path.join(os.getcwd(), folder_path)
    for filename in os.listdir(folder):
      files_name.append(os.path.join(folder, filename))
    return files_name

  def get_file_content(self, file_path):

    '''

      Public method which gets all the Debates or arguments from a single file.

        Parameters
        ----------
        file_path: str
          The path of the file containing the Debates downloaded from "Kialo.com"

        Returns
        -------
        dataframe_rows: list of a dictionary
          all the debates avilable inside the file indexed and formated, in such a way that it is easier to store in Dataframe.

    '''  
    dataframe_rows = []
    # open the file
    with open(file_path, mode='r', encoding='utf-8') as text_file:
      # read it and extract informations
      title = text_file.readline().replace('Discussion Title: ', "").replace("Discussion Name: ", "")
      while True:
        # Get next line from file
        argument = text_file.readline()
        srt = str(argument)
        if (srt[0:2] == "1." and not re.search(": -> See 1.", srt)):
          position= argument.split(' ', 2)
          dataframe_row = {
          "index": self.idx,
          "title": title,
          "position": [int(s) for s in position[0].split('.') if s.isdigit()],
          "opinion": 'Farg' if [int(s) for s in position[0].split('.') if s.isdigit()] == [1] else ''.join([str(elem) for elem in position[1:2]]),
          "argument": ' '.join([str(elem) for elem in position[2:]])
          }
          self.idx +=1
          dataframe_rows.append(dataframe_row)
        if not argument:
          break
    return dataframe_rows

  def Creat_Dataframe(self, content):

    '''

      Public method which gets content in the form of list and convert them to Panda DataFrame and store them on a pkl file.

        Parameters
        ----------
        content: list 
          The list of Debates extracted and indexed from a file.

        Returns
        -------
        dataframe: DataFrame
          > The DataFrame that contains each Debate argument with proper Futures ["index", "title", "position", "opinion", "argument"]
          > Saves the DataFrame in /dataframe/ dataset name.pkl file for further use.

    '''

    folder = os.path.join(os.getcwd() , "dataframe")
    if not os.path.exists(folder):
      os.makedirs(folder)
    # transform the list of rows in a proper dataframe
    dataframe = pd.DataFrame(content)
    dataframe = dataframe[list(dataframe.iloc[0].keys())]
    dataframe_path = os.path.join(folder + '/' + self.dataset_name + ".pkl")
    dataframe.to_pickle(dataframe_path)
    dataframe.to_excel(folder + '/' + self.dataset_name + '.xlsx', index = False)  
    return dataframe

  def Upload_folder(self, folder_Path):
    '''

      Public method which gets all the Debates or Arguments from an entire folder. 
      [Taking in to account all the files inside the folder are .txt files downloaded from Kialo.com website]

        Parameters
        ----------
        folder_path: str
          The path to the folder containing the files downloaded from "Kialo.com"

        Returns
        -------
        dataframe: DataFrame
          DataFrame that contains All the debates avilable inside each files that are in the folder specified.

    ''' 
    files_name = self.get_files_name(folder_Path)
    file_content = []
    for file_name in files_name:
      try:
        if os.path.isfile(file_name):
          file_path = os.path.join(os.getcwd(), file_name)
          file_content.extend(self.get_file_content(file_path))
      except Exception as e:
          print('Failed to process %s. Reason: %s' % (file_name, e))
          sys.exit(0)
    return self.Creat_Dataframe(file_content)

def main(*arg):
  '''
    main method for Kialo_to_DF.py script:

    INPUT: Accepts from user
      > file Path or Folder path.
      > Accepts user prefered Dataset name.

    OUTPUT: Generate DataFrame 
      > with ["index", "title", "position", "opinion", "argument"] Futures
      > from the given file or folder path.
  '''
  if(arg):
    path = arg[0]
    datasetName = arg[1]
  else:
    path = input('Inter File or folder path: ')
    datasetName = input("Give your Dataset a Name: ")

  if(re.search('.txt', path)):
    #single file uploader#
    Data = load_data(dataset_name = datasetName)
    print("UPLOADING DATASET FROM FILE.........")
    content = Data.get_file_content(path)
    df = Data.Creat_Dataframe(content)
  else:
    # Folder uploader #  
    Data = load_data(dataset_name = datasetName)
    print("UPLOADING DATASET FROM FOLDER.........")
    df = Data.Upload_folder(path)
  print("\t Argument Pool Count ")
  print("\t", datasetName, ' size: ', len(df))
  print()
  return df


if __name__ == '__main__':
  main()