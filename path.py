import sys
import os

# List of directories to be added to sys.path
directories = ['components',
               'data',
               'explanation',
               'features',           
               'networks', 
               'notebook',
               'output',
               'train',
               'utils'
               ]

# Add each directory to the sys.path variable
for dir_name in directories:
    sys.path.append(os.path.abspath(
        os.path.join(os.getcwd(), os.pardir, dir_name)
        )
    )