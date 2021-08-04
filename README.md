# REENet

This is a Torch-based implementation of the similarity classification that identities two tuples with high semantic similarity. 
It returns true if the two tuples refer to the same real-world entity.

Dependencies
-------
In order to build REENet, you also need the following:
```
  1.  libtorch >= 1.7.1
  2.  Torch >= 1.0
  3.  CUDA >= 11.0
```

Build and start REENet locally
-------

0. Download Libtorch from https://pytorch.org/cppdocs/installing.html to "./lib"
1. Train the embeding model by source code provided in "./python/train".
2. Train the classification model by source code provided in "./pyhton/train_classifier.py"
3. Copy the embeding and the classification models to "./models/"
5. Check the path of your models in "./examples/main.cc"
4. Bash: mkdir build & cd build
5. Bash: cmake ..
6. Bash: "./exe_reenet" or mv "./libreenet.so" to you library.
7. Rewrite your similarity function as follows:

```
    #include "../include/cfasttext.h"
    #include "../include/core.h"
    #include <torch/script.h> // One-stop header.
    
    reenet::REEModule reemodule = reenet::REEModule("embeding model xx.bin", "ML model xx.pt");

    std::string str_l = "Modeling High-Dimensional Index Structures using Sampling";
    std::string str_r = "On-line reorganization of sparsely-populated B+-trees";
    bool der = reemodule.ML(
        str_l,
        str_r
    );
    cout<<der<<endl;
```

