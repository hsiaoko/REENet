# REENet
PER
========
Install
-------
```Bash
Copy your ML model ()
sudo apt install libyaml-cpp-dev
sudo apt install libsqlite3-dev
sudo apt install mpich
mkdir build
cd build
cmake ../
make
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
