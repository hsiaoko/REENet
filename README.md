# REENet
Torch-based, implementation of similarity classification that identity two tuples with the same semantic similarity. It return true if two tuples are referring as the same real world entity.


Instructions
-------
```
1. Copy your ML models to "./models"
2. 
```
Run
------
```Bash
./per -e paper # Use the experiment name directly if written in code
```
```Bash
./per -m -r "path/to/rules" -n 5 -d "path/to/db" -s "path/to/schema"

-m    #Set to use MQO
-n 5  #Number of rules
-r "path/to/rules"   # Path of the rules file
-d "path/to/db"      # Path of the data file
-s "path/to/schema"  # Path of the schema file
```
