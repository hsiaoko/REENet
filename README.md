# REENet
Torch-based, implementation of similarity classification that identity two tuples with the same semantic similarity. It return true if two tuples are referring as the same real world entity.

Build and start Ditto locally
-------
In order to build Ditto, you'll need
```
1.  torch/script.h
2.  Torch >= 1.0
```

Instructions
-------
```
1. Train embeding model by source code provided "./python/train".
2. Train classification model by source code provided by "./pyhton/train_classifier.py"
3. Copy your classification and embeding models to "./models/"
5. Check path of your models in "./examples/main.cc"
4. Bash: mkdir build & cd build
5. Bash: cmake ..
6. Bash: ./exe_reenet
```
