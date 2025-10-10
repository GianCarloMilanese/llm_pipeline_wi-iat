#!/usr/bin/env python3
"Create dataset"

import pandas as pd
import os

directory_o = '../data/train/'
for file in os.listdir(directory_o):
   filename = os.fsdecode(file)
   
   if ('semeval' in filename):
      dataset_train = pd.read_csv('../data/train/'+filename)
      dataset_test = pd.read_csv('../data/test/'+filename.replace('train',''))

     

      list_label = list(dataset_train['2'])

      list_label_train = []
      for el in list_label:
         if el not in list_label_train:
            list_label_train.append(el)

      list_label = list(dataset_test['2'])

      list_label_test = []
      for el in list_label:
         if el not in list_label_test:
            list_label_test.append(el)

      n_label_train = len(list_label_train)
      n_label_test = len(list_label_test)
     
      perc_label_train = {}
      for lab in list_label_train:
         perc_label_train[lab] = round(len(dataset_train[dataset_train['2']==str(lab)])/len(dataset_train),3)
        

      perc_label_test = {}
      for lab in list_label_test:
         perc_label_test[lab] = round(len(dataset_test[dataset_test['2']==str(lab)])/len(dataset_test),3)

      if len(dataset_train)>3000:
         lista_dataset = []
         elementi_per_label = [int(3000*perc_label_train[x]) for x in list(perc_label_train.keys())]
         k = 0
         for lab in list(perc_label_train.keys()):
            lista_dataset.append(dataset_train[dataset_train['2']==str(lab)][:elementi_per_label[k]])
            k += 1

         dataset_train_cut = pd.concat(lista_dataset)

      else:
         dataset_train_cut = dataset_train

      if len(dataset_test)>3000:
         lista_dataset = []
         elementi_per_labelt = [int(3000*perc_label_test[x]) for x in list(perc_label_test.keys())]
         k = 0
       
         for lab in list(perc_label_test.keys()):
            lista_dataset.append(dataset_test[dataset_test['2']==str(lab)][:elementi_per_labelt[k]])
            k+=1


         dataset_test_cut = pd.concat(lista_dataset)
      
      else:
         dataset_test_cut = dataset_test
      
      train_text = dataset_train_cut['1']
      test_text = dataset_test_cut['1']
      
      train_label = dataset_train_cut['2']
      test_label = dataset_test_cut['2']


      list_label = list(dataset_train_cut['2'])

      list_label_train = []
      for el in list_label:
         if el not in list_label_train:
            list_label_train.append(el)

      list_label = list(dataset_test_cut['2'])

      list_label_test = []
      for el in list_label:
         if el not in list_label_test:
            list_label_test.append(el)

      n_label_train = len(list_label_train)
      n_label_test = len(list_label_test)


      perc_label_train_3k = {}
      for lab in list_label:
         perc_label_train_3k[lab] = round(len(dataset_train_cut[dataset_train_cut['2']==str(lab)])/len(dataset_train_cut),3)
        

      perc_label_test_3k = {}
      for lab in list_label_test:
         perc_label_test_3k[lab] = round(len(dataset_test_cut[dataset_test_cut['2']==str(lab)])/len(dataset_test_cut),3)

      balance_elementi = []
      try:

         dict_b = {'Rapporto_train_orig_3k':len(dataset_train)/len(train_text),'Rapporto_test_orig_3k':len(dataset_test)/len(test_text), 'distribuzione_train':perc_label_train, 'dist_train_3k': perc_label_train_3k, 'distribuzione_test':perc_label_test, 'dist_test_3k':perc_label_test_3k}
         balance_elementi.append({'Rapporto_train_orig_3k':len(dataset_train)/len(train_text),'Rapporto_test_orig_3k':len(dataset_test)/len(test_text),'distribuzione_train':perc_label_train, 'distribuzione_test':perc_label_test})

      except:

         import ipdb
         ipdb.set_trace()
      
      
      train = pd.DataFrame({'Text':train_text, 'Label':train_label})
      test = pd.DataFrame({'Text':test_text, 'Label':test_label})

      print(dict_b)

      dataset_train_cut.to_csv('./datasets_preprocessed/'+filename)

      dataset_test_cut.to_csv('./datasets_preprocessed/'+filename.replace('trainAll','test'))
