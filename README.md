# REENet
Torch-based, implementation of similarity classification that identity two tuples with the same semantic similarity. It return true if two tuples are referring as the same real world entity.

Build and start REENet locally
-------
In order to build REENet, you'll need
```
1.  libtorch >= 1.7.1
2.  Torch >= 1.0
3.  CUDA >= 11.0
```

Instructions
-------

0. Download Libtorch from https://pytorch.org/cppdocs/installing.html to "./lib"
1. Train embeding model by source code provided "./python/train".
2. Train classification model by source code provided by "./pyhton/train_classifier.py"
3. Copy your classification and embeding models to "./models/"
5. Check path of your models in "./examples/main.cc"
4. Bash: mkdir build & cd build
5. Bash: cmake ..
6. Bash: "./exe_reenet" or mv "./libreenet.so" to you library.
7. Rewrite your similarity function with
```
    reenet::REEModule reemodule = reenet::REEModule("embeding model xx.bin", "ML model xx.pt");

    std::string str_l = "Modeling High-Dimensional Index Structures using Sampling";
    std::string str_r = "On-line reorganization of sparsely-populated B+-trees";
    bool der = reemodule.ML(
        str_l,
        str_r
    );
    cout<<der<<endl;
```

