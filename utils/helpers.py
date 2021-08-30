import os
import shutil

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.mkdir(directory)
    except OSError:
        print("Error creating directory: " + directory)

def clearFolderContents(directory):
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    except OSError:
        print("Error in deleting the contents of the directory: " + directory)
