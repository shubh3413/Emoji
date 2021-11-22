import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast
import streamlit as st
from torchtext.legacy import data
import random
import spacy
import ast

# specify GPU
device = torch.device("cpu")

st.sidebar.title("CS 613: Natural Language Processing")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 450px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 450px;
        margin-left: -450px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.sidebar.subheader("About")
st.sidebar.markdown('<p style="text-align: justify">Emojis, a condensed form of language that can expresses emotions, have become an inescapable part of our online conversations. The emoji suggestion problem aims at predicting the emojis that are associated with a particular text. Models based on distant supervision that employ Bi-LSTMs and Transformer Networks are currently state-of-the-art for predicting emojis from a given text. On these lines, we implement a model that predicts and inserts emoji. Our model gives comparable performance to the state of the art baselines in the prediction task and also succeeds at placing the emoji at the intended position in the text. </p>', unsafe_allow_html = True)
st.sidebar.subheader("Proposed Architecture")
st.sidebar.image("Model.jpeg")
st.sidebar.subheader("Group Members")
st.sidebar.text('- Paras Gupta')
st.sidebar.text('- Reuben Devanesan')
st.sidebar.text('- Tarun Sharma')
st.sidebar.text('- Shubh lavti')


st.title('Emoji Suggester')

'''Defining Model Architecture... '''
class BERT_Arch(nn.Module):
    def __init__(self, bert):
      
      super(BERT_Arch, self).__init__()

      self.bert = bert 
      
      # dropout layer
      self.dropout = nn.Dropout(0.1)
      
      # relu activation function
      self.relu =  nn.ReLU()

      # dense layer 1
      self.fc1 = nn.Linear(768,128)

      # Embedding Layer
      self.emb = nn.Embedding(128, 64)

      self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):

      # pass the inputs to the model  
      _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
      
      ######  Layer After Bert --- Second last layer
      ###### **********Fine Tuning Layer**********
      x = self.fc1(cls_hs)

      x = self.relu(x)

      x = self.dropout(x)

      xbar = x.unsqueeze(-1)
      
      y = self.emb(torch.LongTensor(torch.arange(128)).to(device))
      
      cos = nn.CosineSimilarity(dim=-2, eps=1e-6)
      output = cos(xbar, y)

      # apply softmax activation
      output = self.softmax(output)

      return output

max_seq_len = 25 

'''Loading Model...'''
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_model():
    # import BERT-base pretrained model
    bert = AutoModel.from_pretrained('bert-base-multilingual-cased')

    # Load the BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False

    # pass the pre-trained BERT to our define architecture
    model = BERT_Arch(bert)
    path = "proposed_saved_weights_mc64_learning4_20epoch.pt"
    model.load_state_dict(torch.load(path, map_location = "cpu"))

    # push the model to GPU
    model = model.to(device)
    return model, tokenizer

def inserter(text):
    next = ['!', '?', '$']
    prev = ['but', 'and', 'or', 'however', 'cooking', 'whereas', 'besides', 'although', 'yet', 'because','since', ',', '.', ';']
    F = 0
    C = 0
    F += 1
    A = []
    for i in text.split(" "):
        A.append(i)

    weights = [0]*len(A)
    count = 0

    for i in range(1, len(A)-1):
        if A[i] in prev:
            weights[i-1]=1
            count+=1
        if A[i] in next:
            weights[i]=1
            count+=1
    if count == 0:
        weights[len(A)-1] = 1
    else:
        weights[len(A)-1] = int(count)
   
    final = []
    for i in range(len(A)):
        for j in range(weights[i]):
            final.append(i)

    random_num = random.choice(final)
    if random_num != len(A)-1:
        C += 1
    
    for i in range(len(A)):
        if i == random_num:
            A.insert(i+1, predemojis[0])

    inserted_text = " ".join(A)
    return inserted_text

class BiLSTMPOSTagger(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, pad_idx):
        
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(self.dropout(outputs))
        return predictions

model, tokenizer = load_model()
emoji_dict = {'❤': 0, '😂': 1, '👍': 2, '🙏': 3, '🙌': 4, '😘': 5, '😍': 6, '😊': 7, '🔥': 8, '👏': 9, '👌': 10, '💪': 11, '👊': 12, '😉': 13, '🎉': 14, '😎': 15, '😁': 16, '💯': 17, '😜': 18, '👀': 19, '💕': 20, '☺': 21, '✨': 22, '✌': 23, '⚽': 24, '😳': 25, '🙈': 26, '😭': 27, '💙': 28, '💋': 29, '🤔': 30, '🎶': 31, '😀': 32, '🤣': 33, '💃': 34, '💜': 35, '🤗': 36, '😄': 37, '😃': 38, '✊': 39, '😩': 40, '☀': 41, '💥': 42, '🏆': 43, '♥': 44, '💛': 45, '😏': 46, '💖': 47, '🤘': 48, '😱': 49, '💚': 50, '👇': 51, '📷': 52, '🙄': 53, '😆': 54, '⚡': 55, '🏀': 56, '😝': 57, '📸': 58, '😬': 59, '💗': 60, '🤷': 61, '🎂': 62, '👉': 63}
emoji_dict_2 = {0: '❤', 1: '😂', 2: '👍', 3: '🙏', 4: '🙌', 5: '😘', 6: '😍', 7: '😊', 8: '🔥', 9: '👏', 10: '👌', 11: '💪', 12: '👊', 13: '😉', 14: '🎉', 15: '😎', 16: '😁', 17: '💯', 18: '😜', 19: '👀', 20: '💕', 21: '☺', 22: '✨', 23: '✌', 24: '⚽', 25: '😳', 26: '🙈', 27: '😭', 28: '💙', 29: '💋', 30: '🤔', 31: '🎶', 32: '😀', 33: '🤣', 34: '💃', 35: '💜', 36: '🤗', 37: '😄', 38: '😃', 39: '✊', 40: '😩', 41: '☀', 42: '💥', 43: '🏆', 44: '♥', 45: '💛', 46: '😏', 47: '💖', 48: '🤘', 49: '😱', 50: '💚', 51: '👇', 52: '📷', 53: '🙄', 54: '😆', 55: '⚡', 56: '🏀', 57: '😝', 58: '📸', 59: '😬', 60: '💗', 61: '🤷', 62: '🎂', 63: '👉'}

text = st.text_input('Enter Text')
flag = st.button('Predict & Insert Emojis')
predemojis = []

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def do_prediction(text):
    example = []
    example.append(text)

    tokens_example = tokenizer.batch_encode_plus(
        example,
        max_length = max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_token_type_ids=False
    )

    example_seq = torch.tensor(tokens_example['input_ids'])
    example_mask = torch.tensor(tokens_example['attention_mask'])

    with torch.no_grad():
        '''Predicting Emojis...'''
        example_preds = model(example_seq.to(device), example_mask.to(device))
        example_preds = example_preds.detach().cpu().numpy()

    result = []
    for i in example_preds:
        lis = np.argsort(i)[-10:]
        result = lis[::-1]

    temp = ""
    for j in result:
        temp += emoji_dict_2[j] + " "
        predemojis.append(emoji_dict_2[j])

    return temp, predemojis

if(flag):
    temp, predemojis = do_prediction(text)
    st.subheader("Input Sentence")
    st.text(text)
    st.subheader("Top 10 Emoji Suggestions")
    st.text(temp)
    # st.text('\n')
    
    TWEET = data.Field(lower=True, tokenize = lambda s: list(ast.literal_eval(s)))
    LABEL = data.Field(unk_token=None, tokenize = lambda s: list(ast.literal_eval(s)))

    fields = {'tweet': ('t', TWEET), 'label': ('l', LABEL)}

    train_data, valid_data, test_data = data.TabularDataset.splits(
        path = '',
        train = 'mc74_train_pos.json',
        validation = 'mc74_val_pos.json',
        test = 'mc74_test_pos.json',
        format = 'json',
        fields = fields,
    )

    MIN_FREQ = 2
    TWEET.build_vocab(train_data, min_freq = MIN_FREQ, vectors = "glove.6B.100d", unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data)    

    INPUT_DIM = len(TWEET.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = len(LABEL.vocab)
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    PAD_IDX = TWEET.vocab.stoi[TWEET.pad_token]

    model_bilstm = BiLSTMPOSTagger(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

    pretrained_embeddings = TWEET.vocab.vectors

    model_bilstm.embedding.weight.data.copy_(pretrained_embeddings)

    model_bilstm.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    model_bilstm = model_bilstm.to(device)

    model_bilstm.load_state_dict(torch.load('tut1-model.pt', map_location = "cpu"))

    def tag_sentence(model, device, sentence, text_field, tag_field):
        model.eval()
        
        if isinstance(sentence, str):
            nlp = spacy.load('en_core_web_sm')
            tokens = [token.text for token in nlp(sentence)]
        else:
            tokens = [token for token in sentence]

        if text_field.lower:
            tokens = [t.lower() for t in tokens]
            
        numericalized_tokens = [text_field.vocab.stoi[t] for t in tokens]
        unk_idx = text_field.vocab.stoi[text_field.unk_token]
        unks = [t for t, n in zip(tokens, numericalized_tokens) if n == unk_idx]
        token_tensor = torch.LongTensor(numericalized_tokens)
        token_tensor = token_tensor.unsqueeze(-1).to(device) 
        predictions = model(token_tensor)
        top_predictions = predictions.argmax(-1)
        predicted_tags = [tag_field.vocab.itos[t.item()] for t in top_predictions]
        return tokens, predicted_tags, unks

    tokens, tags, unks = tag_sentence(model_bilstm, device, text, TWEET, LABEL)

    st.subheader("Text with Emoji inserted")
    if(tags[len(tags) - 1] == '1' or '1' not in tags):
        inserted_text = inserter(text)
        st.text(inserted_text)
    else:
        tokens_list = text.strip().split()
        output = ""
        for i in range(len(tags)):
            output += tokens_list[i] + " "
            if(tags[i] == '1'):
                output +=  predemojis[0] + " "
        st.text(output)
