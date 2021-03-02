import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import fasttext
import torch.nn.functional as F
from torch.utils import data
import modules.classifiers as classifiers
import modules.embeding_models as eb

class REEDataset(data.Dataset):
    def __init__(self, attr_name, train_pth, bin_pth, embeding_style):
        self.train_pth = train_pth
        self.data  = pd.read_csv(self.train_pth)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data_size = len(self.data)
        self.train_label = self.data['label']
        self.label =  torch.tensor(np.array(self.data["label"].values))
        self.label = self.label.view(-1,1)
        self.attr_pair = self.get_attr(attr_name=attr_name)
        self.eb_model = eb.FastTextEmbeding(bin_pth=bin_pth)
        self.embeding_data = self.eb_model.dataset_embeding(self.attr_pair, embeding_style)

    def get_attr(self, attr_name):
        l_attr_name = "left_"+attr_name
        r_attr_name = "right_"+attr_name
        attr_pair = [self.data[l_attr_name].values, self.data[r_attr_name].values]
        attr_pair = np.array(attr_pair)
        return attr_pair.T
    def get_label(self):
        return self.label
    def __getitem__(self, index):
        label = self.label[index]
        data = self.embeding_data[index]
 #       print(self.attr_pair[index])
        data[0] = torch.tensor([data[0]])
        data[1] = torch.tensor([data[1]])
        return data, label
    def init(self):
        self.embeding_data = self.eb_model.dataset_embeding(self.attr_pair, embeding_style)

    def __len__(self):
        return len(self.label)

    

class REENet():
    def __init__(self, train_data_pth, eval_data_pth, test_data_pth, attr_name, embeding_style, bin_pth, classifier, args, model_pth):
        self.train_data_pth = train_data_pth
        self.eval_data_pth = eval_data_pth
        self.test_data_pth = test_data_pth
        self.bin_pth = bin_pth
        self.train_data = REEDataset(train_pth=train_data_pth, bin_pth=bin_pth, attr_name=attr_name,embeding_style=embeding_style)
        self.eval_data = REEDataset(train_pth=train_data_pth, bin_pth=bin_pth, attr_name=attr_name,embeding_style=embeding_style)
        self.test_data = REEDataset(train_pth=test_data_pth, bin_pth=bin_pth, attr_name=attr_name,embeding_style=embeding_style)

        self.args = args
        self.classifier = classifier
        self.loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=self.args['LR'])   # optimize all parameters
        self.model_pth = model_pth
        self.eb_model = eb.FastTextEmbeding(bin_pth=bin_pth)

    def train(self,):
        for epoch in range(self.args['EPOCH']):
            tp = 0
            count =0
            self.train_data.init()
            for step, (x, y) in enumerate(self.train_data):
                left_x = (x[0]) 
                right_x = (x[1])
                output = self.classifier(left_x, right_x)
                loss = self.loss_func(output, y)   # cross entropy loss
                self.optimizer.zero_grad()           # clear gradients for this training step
                pred_y = torch.max(output, 1)[1].data.numpy()
                if(pred_y == y.numpy()):
                    tp += 1
                count+=1    
                if step % 50 == 0:
                    loss.backward()                 # backpropagation, compute gradients
                    self.optimizer.step()                # apply gradients
                if step % 100 == 0:
                    print("epoch: %d, tp: %d, prec: %f" % (epoch, tp, tp/count))
                    tp = 0
                    count = 0
        sm = torch.jit.script(self.classifier)
        print("save: ", self.model_pth)
        sm.save(self.model_pth)
        pass
    
    def test(self):
        tp = 0
        ree_model = torch.jit.load(self.model_pth)
        for index, (x, y) in enumerate(self.test_data):
            left_x = (x[0]) 
            right_x = (x[1])
            #print(left_x, right_x)
            #output = self.classifier(left_x, right_x)
            output = ree_model(left_x, right_x)
            pred_y = torch.max(output, 1)[1].data.numpy()
            if(pred_y == y.numpy()):
                tp += 1
            else:
                pass
              #  print(pred_y, self.test_data.attr_pair[index])
        print("tp: %d, prec: %f"% (tp, tp/ self.test_data.data_size))
        pass  
    def predicate(self, x_l, x_r):
        ree_model = torch.jit.load(self.model_pth)
        x_l = self.eb_model.seq_embeding(x_l)
        x_r = self.eb_model.seq_embeding(x_r)
        x_l = torch.tensor([x_l])
        x_r = torch.tensor([x_r])
        output = ree_model(x_l, x_r)
        pred_y = torch.max(output, 1)[1].data.numpy()
        print(pred_y)
        pass

if __name__ == '__main__':
    train_data_pth = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/train_balance.csv"
    test_data_pth = "/home/LAB/zhuxk/project/data/ER-dataset-benchmark/ER/DBLP-ACM/train_balance.csv"
    bin_pth = "/home/LAB/zhuxk/project/REENet/models/embeding/dblp_acm_authors.bin"
    embeding_style = "seq"
    attr_name = "authors"
    args = {
        "EPOCH":1,
        "BATCH_SIZE":64,
        "TIME_STEP":28,  
        "INPUT_SIEE":28,
        "LR":0.01
    }
    model_pth = "../models/classifiers/rnn_linear_authors_seq.pt"
    classifier = classifiers.REEModel()


    ree_net = REENet(
        train_data_pth=train_data_pth,
        eval_data_pth=train_data_pth,
        test_data_pth=test_data_pth,
        attr_name=attr_name,
        embeding_style=embeding_style,
        bin_pth=bin_pth,
        classifier=classifier,
        args=args,
        model_pth=model_pth
    )

    #ree_net.train()

    ree_net.test()
    
    str_1 = "A. R. Dasgupta"
    str_r = "A. R. Dasgupta"
    str_1 = "Gio Wiederhold, Byung Suk Lee"
    str_r = "Yuh-Ming Shyy, Javier Arroyo, Stanley Y.W. Su, Herman Lam"
    ree_net.predicate(str_1, str_r)
    #print(test_output)
