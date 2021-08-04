#ifndef TOOLS_H_
#define TOOLS_H_
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <string>
#include <fstream>
#include "rapidcsv.h"
#include <torch/script.h>
#include "cfasttext.h"
using namespace std;

void avg_embeding(float *vec_, size_t len_vec_, float *sentence_vec);
void vec_division(float *vec_, size_t len_vec_, float divisor);
void show_buf(float *buf, size_t length_buf);

float get_score(float *sentence_vec_l_, float *sentence_vec_r_, size_t lev_vec_);
vector<vector<std::string> *> *read_csv(bool read_head, char split_symbol, std::string csv_path_);
void split_string(const string &s, vector<string> &v, const string &c);

namespace mrlnet
{
class MRLModule
{
private:
    char *errptr = new char[8];
    fasttext_t *handle = cft_fasttext_new();
    torch::jit::script::Module module;
    std::string fasttext_bin_pth;
    std::string model_bin_pth;
public:
    MRLModule();
    ~MRLModule();
    MRLModule(std::string fasttext_bin_pth, std::string model_bin_pth);

    bool ML(std::string l_sentense, std::string r_sentense);

    void embeding(std::string, float **, std::vector<size_t> *);
    float *rnn_embeding(std::string sentense, size_t embeding_size);
    static void split_string(const string &s, vector<string> &v, const string &c);
};

class CSVLoader
{
private:

public:
    std::string csv_path_;
    size_t data_size_;
    std::vector<std::vector<std::string>*> *tuples_;

    CSVLoader(std::string csv_path, char split_symbol, bool read_head);

    void ShowTuple(std::vector<std::string> *tuple);
    void ShowCSV();
    void ShowCSV(size_t k);
    void ShowCSV(std::vector<std::vector<std::string> *> *tuples, size_t k);
    std::vector<string> *split_str(string s);
    vector<vector<std::string> *> *read_csv(bool read_head, char split_symbol, std::string csv_path_);
    void freeTuples();
};

} //end namespace mrlnet
#endif //TOOLS_H_
