from shutil import rmtree
import os

os.chdir(os.path.split(__file__)[0])
evidence = ['train', 'val', 'test', 'images_tmp']

for clue in evidence:
    rmtree(clue, ignore_errors=True)