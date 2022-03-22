import os
import shutil

root = 'F:\\VelocitySpectrum\\MIFN\\2GeneraTest'
Folder = [os.path.join(root, file) for file in os.listdir(root) if file.split('-')[0] == 'Ep']
Folder = [file for file in Folder if int(file.split('-')[1])>=120]
for folder in Folder:
    DelFolder = os.path.join(folder, 'test')
    if os.path.exists(DelFolder):
        shutil.rmtree(DelFolder)