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

    string csv_pth_ = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/TPACC/Test2005/processed/train_item_detail_desc.csv";

    mrlnet::CSVLoader csv_loader = mrlnet::CSVLoader(csv_pth_, ' ', false);
    mrlnet::MRLModule mrlmodule = mrlnet::MRLModule("../models/embeding/tfacc/desc.bin", "../models/classifiers/tfacc/tfacc_desc_clas.pt");
    //float *words_embeding;
    //std::vector<size_t> *embeding_shape = new std::vector<size_t>;
    //size_t tp = 0;
    //for (int i = 0; i < csv_loader.tuples_->size(); i++)
    //{
    //    std::string str_l = csv_loader.tuples_->at(i)->at(1);
    //    std::string str_r = csv_loader.tuples_->at(i)->at(2);
    //    if (str_l.length() == 0 || str_r.length() == 0)
    //    {
    //        cout << "pass" << endl;
    //        continue;
    //    }
    //    bool der = reemodule.ML(
    //                   str_l,
    //                   str_r
    //               );
    //    float *rnn_embeding = reemodule.rnn_embeding(str_l, 64);
    //    //for (int i = 0; i < 64; i++)
    //    //{
    //    //    cout << rnn_embeding[i] << ", ";
    //    //}
    //    //cout << "\n##################" << endl;
    //    if (to_string(der) == csv_loader.tuples_->at(i)->at(1))
    //    {
    //        tp++;
    //    }
    //    else
    //    {
    //        cout << "dis match: " << str_l << ", " << str_r << endl;
    //    }
    //}
    std::string str_l = "not able to be operated   normal riding position";
    std::string str_r = "not able be operated from  normal riding position";

    bool der = mrlmodule.ML(
                   str_l,
                   str_r
               );
    cout << der << endl;
    str_l = "does not face the front";
    str_r = "inadequatly repaired significantly reducing the original strength";
    der = mrlmodule.ML(
              str_l,
              str_r
          );
    cout << der << endl;


    //printf("tp: %d, prec: %f\n", tp, (float)tp / (float)csv_loader.tuples_->size());
}
