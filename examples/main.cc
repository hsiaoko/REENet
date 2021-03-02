#include "../include/cfasttext.h"
#include "../include/core.h"
#include <torch/script.h> // One-stop header.
#include <iostream>
using namespace std;
int main(int argc, const char *argv[])
{
    char *errptr = new char[8];

   // fasttext_t *handle = cft_fasttext_new();
   // cft_fasttext_load_model(handle, "../models/embeding/dblp_acm_title.bin", (char **)&errptr);

    string csv_pth_ = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/train_balance.csv";
    reenet::CSVLoader csv_loader = reenet::CSVLoader(csv_pth_, ' ', false);
    reenet::REEModule reemodule = reenet::REEModule("../models/embeding/dblp_acm_authors.bin", "../models/classifiers/rnn_linear_authors_seq.pt");
    float *words_embeding;
    std::vector<size_t> *embeding_shape = new std::vector<size_t>;
    size_t tp = 0;
    for (int i = 0; i < csv_loader.tuples_->size(); i++)
    {
        std::string str_l = csv_loader.tuples_->at(i)->at(3);
        std::string str_r = csv_loader.tuples_->at(i)->at(7);
        if(str_l.length()==0 || str_r.length() == 0) {
            cout<<"pass"<<endl;
            continue;
        }
        bool der = reemodule.ML(
            str_l,
            str_r
            );
        if (to_string(der) == csv_loader.tuples_->at(i)->at(1))
        {
            tp++;
        } else {
            cout<<str_l<<", "<<str_r<<endl;
        }
    }

    std::string str_l = "Modeling High-Dimensional Index Structures using Sampling";
    std::string str_r = "On-line reorganization of sparsely-populated B+-trees";
    bool der = reemodule.ML(
        str_l,
        str_r
    );
    cout<<der<<endl;


    printf("tp: %d, prec: %f\n", tp, (float)tp/(float)csv_loader.tuples_->size());
}
