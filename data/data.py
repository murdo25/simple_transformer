from tqdm import tqdm
import torchtext
from torchtext.data.utils import get_tokenizer
import revtok # Just to make sure you have revtok installed.. pip install revtok

class Data:
    def __init__(self, data_path, device, BPTT=35, BATCH_SIZE=32):
        self.batch_size = BATCH_SIZE
        self.device = device
        self.vocab = {}
        self.max_seq_len = 0 
        self.bptt = BPTT

        self.progress_through_dataset = 0

        # self.train = []
        self.test = [] # split(split_ratio=0.7, stratified=False, strata_field='label', random_state=None)
        self.eval = []
        self.TEXT = torchtext.data.ReversibleField(tokenize=get_tokenizer("basic_english"),
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

        self.data = self.clean(data_path)
        self.max_seq_len = self.find_max_seq_len(self.data)
        self.TEXT.build_vocab(self.data)
        print("vocab", self.TEXT.vocab.stoi)
        self.train = self.batchify(self.data, self.batch_size)


    def clean(self, data_path):
        open_file = open(data_path, 'r').readlines()

        lines = []
        # Strips the newline character 
        for line in tqdm(open_file): 
            line = line.strip()
            lines.append(line)
        return lines

    def getData(self):
        return self.train, self.test, self.eval
    
    def find_max_seq_len(self, data):
        max_seq_len = 0 
        for d in self.data:
            if len(d) > max_seq_len:
                max_seq_len = len(d)
        # print("max_seq_len: ", max_seq_len)
        return max_seq_len

    def batchify(self, data, batch_size):
        # print("vocab", self.TEXT.vocab.stoi)
        
        # data = [self.TEXT.numericalize(d) for d in data]
        data = self.TEXT.pad(data)
        data = self.TEXT.numericalize(data)
        # print(self.TEXT.reverse(data))
        data = data.T
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // batch_size 
        # print("nbatch", nbatch.shape)
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the bsz batches.
        # data = data.view(bsz, -1).t().contiguous()
        data = data.reshape((data.shape[0]//batch_size, batch_size, data.shape[1]))
        # print("full dataset:", data.shape)
        # return data.to(self.device)
        return data

        def get_next_n_datapoints(self, data, current_itteration, batch_size):
            return data[current_itteration:current_itteration+batch_size]



    def get_batch(self, source, batch, i):
        # print("\n\n")
        # print("incomming source",source.shape)
        source = source[batch]
        # print("batched source", source.shape)
        # print("\n\n")

        # if(i > self.max_seq_len):
        #     data = self.get_the_next_n_datapoints()
        
        # print("\n\n")
        # print("source:", source.shape)
        source = source.T
        # print("source:",source.shape,"i:",i)
        seq_len = min(self.bptt, len(source) - 1 - i)
        # print("seq_len:", seq_len)
        data = source[i:i+seq_len]
        # print("data:",data.shape)
        target = source[i+1:i+1+seq_len].reshape(-1)
        # print("data:",data.shape,"target:",target.shape)
        # return data, target
        # SHOULD RETURN DATA OF SHAPE: [BPTT, batch_size]
        print("get_batch data:",data.shape, "target:",target.shape)
        return data.to(self.device), target.to(self.device)


        # print("shape",data.shape)
        # data = data.size(0) // batch_size 
        # Evenly divide the data across the bsz batches.
        # data = data.view(batch_size, -1).t().contiguous()
        # 
        # return data.to(self.device)


# d = Data('mini_train_set.txt') 
# d = Data('full_train_set.txt') 


# train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)

# print(type(train_txt.examples[0].text))
# print(train_txt.examples[0].text)
# print("my data:", dataset.data[0])

# data = TEXT.numericalize(dataset.data[0])
# print(data)
# print(data.shape)
# print("my data:", len(dataset.data[0]))
# print(TEXT.vocab_cls)


# def batchify(data, bsz):
#     #data = TEXT.numericalize([data.examples[0].text])
#     data = dataset.data
#     print("data", type(data), data)
#     # Divide the dataset into bsz parts.
#     nbatch = []
#     for i in range(0, len(data), bsz):
#         nbatch.append(data[i:i + bsz])
#     print('nbatch', nbatch)
#     exit()
#     # nbatch = data.size(0) // bsz
#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.narrow(0, 0, nbatch * bsz)
#     # Evenly divide the data across the bsz batches.
#     data = data.view(bsz, -1).t().contiguous()
#     return data.to(device)

