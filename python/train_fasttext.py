import fasttext
if __name__ == '__main__':
    corpus_pth = "../corpus/dblp_acm/dblp_acm.title.csv"
    model_pth = "../models/embeding/dblp_acm_title.bin"
    model = fasttext.train_unsupervised(corpus_pth)
    print(model.words)
    model.save_model(model_pth)