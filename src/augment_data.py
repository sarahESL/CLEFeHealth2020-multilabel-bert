import pandas as pd
import random
from tqdm import tqdm
from nltk.corpus import stopwords
import re
import pickle


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub('[\W]+', ' ', text.lower())
    text = text.split()
    return text


data_path = 'path/to/concated/traindev/csvdata'

df = pd.read_csv(data_path,
                 escapechar='\\')

# Get maximum number of samples over icd codes
icds_sample_cnt = {}
ICD10CODES = [i for i in df.columns if i != 'filename' and i != 'text']

with open('../data/synonyms_dictionary.pickle', 'rb') as f:
    synonyms_dictionary = pickle.load(f)

for code in ICD10CODES:
    icds_sample_cnt[code] = df[code].sum()


# Augment
augmented_df = pd.DataFrame(columns=df.columns)
max_sample_cnt = max(icds_sample_cnt.value())
STOPWORDS = stopwords.words('spanish') + ICD10CODES

for ind, val in tqdm(df['text'].items()):
    associated_icds = [col for col in ICD10CODES
                       if df.loc[ind, col] == 1]
    min_sample_cnt = 1
    # calculating the minimum
    for icd in associated_icds:
        samples_cnt = icds_sample_cnt[icd]
        if samples_cnt < min_sample_cnt:
            min_sample_cnt = samples_cnt
    for i in range(int(max_sample_cnt / (3*min_sample_cnt)) + 1):  # we add 1, to refuse 0 number of augmentation
        augmented_text = ''
        text = preprocessor(val)
        for token in text:
            if token in STOPWORDS:
                augmented_text += token + ' '
                continue
            similar_tokens = synonyms_dictionary[token]
            if len(similar_tokens) == 0:
                augmented_text += token + ' '
            else:
                new_token = random.choice(similar_tokens)
                augmented_text += new_token + ' '
        current_row = df.iloc[ind]
        current_row.loc['text'] = augmented_text
        current_row.loc['filename'] = 'augmented'
        augmented_df = augmented_df.append(current_row, ignore_index=True)


print(augmented_df.shape)
print(augmented_df.head(5))
augmented_df.to_csv('../data/augmented_df_fasstext.csv', index=False)
