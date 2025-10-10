# results
import pandas as pd
import os

results = {}
i = 0
for file in os.listdir('./results/'):
   
   filename = os.fsdecode('./results/'+file)
   if ('Average_results' not in filename) and (('traditional' in filename) or ('LLM' in filename)):
   # cardiffnlp_spanish_train_Qwen2.5-7B-Instruct_0-shots_en-prompt_classic_preprocessing_Lemmatization.csv
      df = pd.read_csv(filename)
      filename = str(file)
      if 'semeval' in file:
         d = 'semeval'
         if 'LLM-preprocessing' in file:
            type_pre = file.split('_')[1]
         else:
            type_pre = file.split('_')[0]

         dataset = file.split('_')[2]

         if 'LLM-preprocessing' in file:
            model = file.split('_')[4]
         else:
            model = 'Traditional'
         prompt = 'en'

      elif 'cardiff' in file:
         d = 'cardiff'
         if 'LLM-preprocessing' in file:
            type_pre = file.split('_')[1]
         else:
            type_pre = file.split('_')[0]
         
         
         dataset = file.split('_')[3]
   

         if 'LLM-preprocessing' in file:
            model = file.split('_')[5]
         else:
            model = 'Traditional'

         if 'LLM' in file:
            prompt = file.split('_')[7].replace('-prompt','') 
         else:
            prompt = 'None'
      
      results = {'domain': d, 'dataset':dataset, 'preproc':type_pre, 'f1': sum(list(df['F1_micro']))/len(list(df['F1_micro']))}
      

      df_new = pd.DataFrame.from_dict(results, orient='index')
      df_new.to_csv('./results/Average_results/res_'+d+'_'+dataset+'_'+type_pre+model+'_'+prompt+'_'+'.csv')

# for lang in ['french','italian','spanish','german','portuguese']:
#    df_lang = df[df['language']==lang]
#    for pre in ['Stemming','Lemmatization','Stopwords and Stemming','Stopwords','Stopwords and Lemmatization']:
#       df_lang_pre = df_lang[df_lang['preproc']==pre]
      


# print(results)
# import json
# with open("results.json", "w") as outfile: 
#     json.dump(results, outfile)
#print(set(list(df['preproc'])))
