#include "../include/core.h"

void avg_embeding(float *vec_, size_t len_vec_, float *sentence_vec)
{
    for (int i = 0; i < len_vec_; i++)
    {
        sentence_vec[i] += vec_[i];
    }
}

float get_score(float *sentence_vec_l_, float *sentence_vec_r_, size_t len_vec_)
{
    printf("\nget_score()\n");
    float *sim_vec_ = (float *)malloc(sizeof(float) * len_vec_);
    for (int i = 0; i < len_vec_; i++)
    {
        sim_vec_[i] = pow((sentence_vec_l_[i] - sentence_vec_r_[i]), 2);
    }
    float sim_ = 0;
    for (int i = 0; i < len_vec_; i++)
    {
        sim_ += sim_vec_[i];
    }
    std::cout << "sim" << sim_ << std::endl;

    return sim_;
}

void vec_division(float *vec_, size_t len_vec_, float divisor)
{
    for (int i = 0; i < len_vec_; i++)
    {
        vec_[i] /= divisor;
    }
}

void show_buf(float *buf, size_t length_buf)
{
    for (int i = 0; i < length_buf; i++)
    {
        std::cout << buf[i] << ", ";
    }
    std::cout << std::endl;
}
std::vector<string> *split_str(string s)
{
    int n = s.size();
    for (int i = 0; i < n; ++i)
    {
        if (s[i] == ',')
        {
            s[i] = ' ';
        }
    }
    istringstream out(s);
    string str;
    std::vector<string> *str_vec = new std::vector<string>;
    while (out >> str)
    {
        str_vec->push_back(str);
    }
    return str_vec;
}

vector<vector<std::string> *> *read_csv(bool read_head, char split_symbol, std::string csv_path_)
{
    std::ifstream fp(csv_path_);
    vector<vector<std::string> *> *tuples = new vector<vector<std::string> *>;
    vector<vector<std::string> *> *col_vec = new vector<vector<std::string> *>;
    string line;
    getline(fp, line);
    cout << "csv_pth: " << csv_path_ << endl;
    rapidcsv::Document doc(csv_path_);
    std::vector<string> *head_vec = split_str(line);

    for (int i = 0; i < head_vec->size(); i++)
    {
        std::vector<string> *col = new std::vector<string>;
        while (head_vec->at(i).find("\"") != head_vec->at(i).npos)
        {
            head_vec->at(i).replace(head_vec->at(i).find("\""), 1, "");
        }
        *(col) = doc.GetColumn<string>(i);
        col_vec->push_back(col);
    }

    vector<std::string> *tuple;
    for (int i = 0; i < col_vec->at(0)->size(); i++)
    {
        tuple = new vector<std::string>;
        for (int j = 0; j < col_vec->size(); j++)
        {
            tuple->push_back(col_vec->at(j)->at(i));
        }
        tuples->push_back(tuple);
    }
    for (int i = 0; i < col_vec->size(); i++)
    {
        col_vec->at(i)->erase(col_vec->at(i)->begin(), col_vec->at(i)->end());
    }
    free(col_vec);

    return tuples;
}

void split_string(const string &s, vector<string> &v, const string &c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}

namespace mrlnet
{
MRLModule::MRLModule(std::string fasttext_bin_pth, std::string model_bin_pth)
{
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        this->module = torch::jit::load(model_bin_pth.c_str());
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading classifier model\n";
    }
    try
    {
        cft_fasttext_load_model(this->handle, fasttext_bin_pth.c_str(), (char **)&this->errptr);
    }
    catch (const c10::Error &e)
    {
        std::cerr << "error loading embeding model\n";
    }
    printf("load %s | %s done!\n", model_bin_pth.c_str(), fasttext_bin_pth.c_str());
}
MRLModule::~MRLModule()
{
    cft_fasttext_free(this->handle);
}

void MRLModule::split_string(const string &s, vector<string> &v, const string &c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}

void MRLModule::embeding(std::string sentense, float **p_words_embeding, std::vector<size_t> *embeding_shape)
{
    vector<string> words;
    split_string(sentense, words, " ");

    *p_words_embeding = (float *)malloc(sizeof(float *) * words.size() * 100);
    for (int i = 0; i < words.size(); i++)
    {
        cft_fasttext_get_word_vector(this->handle, words.at(i).c_str(), (*p_words_embeding + i * 100));
        //show_buf((*p_words_embeding + i * 100), 100);
    }
    embeding_shape->push_back(1);
    embeding_shape->push_back(words.size());
    embeding_shape->push_back(100);
}

bool MRLModule::ML(std::string l_sentense, std::string r_sentense)
{
    float *l_words_embeding, *r_words_embeding;
    std::vector<size_t> *l_embeding_shape, *r_embeding_shape;
    l_embeding_shape = new std::vector<size_t>;
    r_embeding_shape = new std::vector<size_t>;
    this->embeding(l_sentense, (float **)&l_words_embeding, l_embeding_shape);
    this->embeding(r_sentense, (float **)&r_words_embeding, r_embeding_shape);

    at::Tensor l_tensor = torch::tensor(at::ArrayRef<float>(l_words_embeding, 1 * l_embeding_shape->at(1) * 100)).view({1, l_embeding_shape->at(1), 100});
    at::Tensor r_tensor = torch::tensor(at::ArrayRef<float>(r_words_embeding, 1 * r_embeding_shape->at(1) * 100)).view({1, r_embeding_shape->at(1), 100});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(l_tensor);
    inputs.push_back(r_tensor);

    at::Tensor output, out_l, out_r;
    output = this->module.forward(inputs).toTuple()->elements()[0].toTensor();
    out_l = this->module.forward(inputs).toTuple()->elements()[1].toTensor();
    out_r = this->module.forward(inputs).toTuple()->elements()[2].toTensor();

    std::tuple<at::Tensor, at::Tensor> max_index_val = at::max(output, 1);

    return std::get<1>(max_index_val).item().toFloat();
}

float *MRLModule::rnn_embeding(std::string sentense, size_t embeding_size)
{
    float *words_embeding, *rnn_embeding = (float *)malloc(sizeof(float) * embeding_size);
    std::vector<size_t> *embeding_shape;
    embeding_shape = new std::vector<size_t>;
    this->embeding(sentense, (float **)&words_embeding, embeding_shape);

    at::Tensor tensor = torch::tensor(at::ArrayRef<float>(words_embeding, 1 * embeding_shape->at(1) * 100)).view({1, embeding_shape->at(1), 100});
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor);
    inputs.push_back(tensor);

    at::Tensor output, out_l, out_r;
    output = this->module.forward(inputs).toTuple()->elements()[0].toTensor();
    out_l = this->module.forward(inputs).toTuple()->elements()[1].toTensor();
    out_r = this->module.forward(inputs).toTuple()->elements()[2].toTensor();
    for (int i = 0; i < 64; i++)
    {
        rnn_embeding[i] = out_l[0][i].item<float>();
//           cout<<out_l[0][i].item<double>()<<" ";
    }
    return rnn_embeding;
}

CSVLoader::CSVLoader(std::string csv_path, char split_symbol, bool read_head)
{
    this->csv_path_ = csv_path;
    this->tuples_ = read_csv(false, ' ', csv_path_);
}
std::vector<string> *CSVLoader::split_str(string s)
{
    int n = s.size();
    for (int i = 0; i < n; ++i)
    {
        if (s[i] == ',')
        {
            s[i] = ' ';
        }
    }
    istringstream out(s);
    string str;
    std::vector<string> *str_vec = new std::vector<string>;
    while (out >> str)
    {
        str_vec->push_back(str);
    }
    return str_vec;
}

vector<vector<std::string> *> *CSVLoader::read_csv(bool read_head, char split_symbol, std::string csv_path_)
{
    std::ifstream fp(csv_path_);
    vector<vector<std::string> *> *tuples = new vector<vector<std::string> *>;
    vector<vector<std::string> *> *col_vec = new vector<vector<std::string> *>;
    string line;
    getline(fp, line);
    cout << "csv_pth: " << csv_path_ << endl;
    rapidcsv::Document doc(csv_path_);
    std::vector<string> *head_vec = split_str(line);

    for (int i = 0; i < head_vec->size(); i++)
    {
        std::vector<string> *col = new std::vector<string>;
        while (head_vec->at(i).find("\"") != head_vec->at(i).npos)
        {
            head_vec->at(i).replace(head_vec->at(i).find("\""), 1, "");
        }
        *(col) = doc.GetColumn<string>(i);
        col_vec->push_back(col);
    }

    vector<std::string> *tuple;
    for (int i = 0; i < col_vec->at(0)->size(); i++)
    {
        tuple = new vector<std::string>;
        for (int j = 0; j < col_vec->size(); j++)
        {
            tuple->push_back(col_vec->at(j)->at(i));
        }
        tuples->push_back(tuple);
    }
    for (int i = 0; i < col_vec->size(); i++)
    {
        col_vec->at(i)->erase(col_vec->at(i)->begin(), col_vec->at(i)->end());
    }
    free(col_vec);

    return tuples;
}

void CSVLoader::ShowTuple(std::vector<std::string> *tuple)
{
    for (int i = 0; i < tuple->size(); i++)
    {
        //   cout<<endl;
        //printf("%s ", tuple->at(i).c_str());
        cout << tuple->at(i) << "|";
    }
    printf("\n\n");
}

void CSVLoader::ShowCSV()
{
    for (int i = 0; i < this->tuples_->size(); i++)
    {
        // for(int j = 0; j < this->tuples_->at(i)->size(); j++){
        //     cout<<this->tuples_->at(i)->at(j)<<"|";
        // }
        cout << "\n"
             << this->tuples_->at(i)->at(0) << "|" << this->tuples_->at(i)->at(1) << "|" << this->tuples_->at(i)->at(2) << "|" << this->tuples_->at(i)->at(3) << "|" << this->tuples_->at(i)->at(4) << "|" << this->tuples_->at(i)->at(5) << "|" << this->tuples_->at(i)->at(6) << "|" << this->tuples_->at(i)->at(7) << endl;
    }
    cout << endl;
}

void CSVLoader::ShowCSV(size_t k)
{
    for (int i = 0; i < k; i++)
    {
        ShowTuple(this->tuples_->at(i));
    }
    cout << endl;
}

void CSVLoader::ShowCSV(std::vector<std::vector<std::string> *> *tuples, size_t k)
{
    size_t t = 0;
    t = tuples->size() < k ? tuples->size() : k;

    for (int i = 0; i < t; i++)
    {
        ShowTuple(tuples->at(i));
    }
    cout << endl;
}

void CSVLoader::freeTuples()
{
    for (int i = 0; i < this->tuples_->size(); i++)
    {
        //for(int j = 0; j< this->tuples_->at(i)->size(); j++){
        this->tuples_->at(i)->clear();
        //}
    }
}
} //end namespace reenet
