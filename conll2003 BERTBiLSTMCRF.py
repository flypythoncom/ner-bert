#!/usr/bin/env python
# coding: utf-8

# In[3]:




import sys
import warnings

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
warnings.filterwarnings("ignore")
sys.path.append("../")


# In[4]:


from modules.data.conll2003.prc import conll2003_preprocess


# In[7]:


data_dir = "./modules/data/conll2003/"


# In[8]:


conll2003_preprocess(data_dir)


# ## IO markup

# ### Train

# In[10]:


from modules.data import bert_data


# In[12]:


import os
data = bert_data.LearnData.create(
    train_df_path=os.path.join(data_dir,"eng.train.train.csv"),
    valid_df_path=os.path.join(data_dir,"eng.testa.dev.csv"),
    idx2labels_path=os.path.join(data_dir,"idx2labels.txt"),
    clear_cache=True,device=device
)


# In[14]:


from modules.models.bert_models import BERTBiLSTMCRF


# In[15]:


model = BERTBiLSTMCRF.create(
    len(data.train_ds.idx2label),
    # model_name='bert-base-multilingual-cased',
    model_name='./bert_data',
    lstm_dropout=0.3, crf_dropout=0.3,device=device)


# In[17]:


from modules.train.train import NerLearner


# In[18]:


num_epochs = 100


# In[19]:


learner = NerLearner(
    model, data, "./modules/models/conll2003-BERTBiLSTMCRF-IO.cpt", t_total=num_epochs * len(data.train_dl))


# In[20]:


model.get_n_trainable_params()


# In[22]:


learner.fit(epochs=num_epochs)


# ### Predict

# In[12]:


from modules.data.bert_data import get_data_loader_for_predict


# In[13]:


dl = get_data_loader_for_predict(data, df_path=data.valid_ds.config["df_path"])


# In[14]:


preds = learner.predict(dl)


# In[15]:


from sklearn_crfsuite.metrics import flat_classification_report


# In[16]:


from modules.analyze_utils.utils import bert_labels2tokens, voting_choicer
from modules.analyze_utils.plot_metrics import get_bert_span_report


# In[17]:


pred_tokens, pred_labels = bert_labels2tokens(dl, preds)
true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])


# In[18]:


assert pred_tokens == true_tokens
tokens_report = flat_classification_report(true_labels, pred_labels, labels=data.train_ds.idx2label[4:], digits=4)


# In[20]:


print(tokens_report)


# ### Test

# In[12]:


from modules.data.bert_data import get_data_loader_for_predict


# In[24]:


dl = get_data_loader_for_predict(data, df_path=os.path.join(data_dir,"eng.testa.dev.csv"))


# In[25]:


preds = learner.predict(dl)


# In[26]:


from sklearn_crfsuite.metrics import flat_classification_report


# In[27]:


from modules.analyze_utils.utils import bert_labels2tokens, voting_choicer
from modules.analyze_utils.plot_metrics import get_bert_span_report


# In[28]:


pred_tokens, pred_labels = bert_labels2tokens(dl, preds)
true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])


# In[29]:


assert pred_tokens == true_tokens
tokens_report = flat_classification_report(true_labels, pred_labels, labels=data.train_ds.idx2label[4:], digits=4)


# In[30]:


print(tokens_report)


# In[11]:


# os.path.join(data_dir,"eng.train.train.csv"),


# In[ ]:




