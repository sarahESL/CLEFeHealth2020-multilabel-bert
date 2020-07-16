from transformers import BertTokenizer
import torch
import pickle
import pandas as pd
import os
from tqdm import tqdm
import sys
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from data_processor import MultiLabelTextProcessor, convert_examples_to_features
from bert import BertForMultiLabelSequenceClassification
import logging


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


def predict(model, path, label_list, tokenizer, test_filename='test.csv'):
    predict_processor = MultiLabelTextProcessor(path)
    test_examples = predict_processor.get_test_examples(path, test_filename, size=-1)

    # Hold input data for returning it
    input_data = [{'filename': input_example.guid} for input_example in test_examples]
    max_seq_length = 512
    test_features = convert_examples_to_features(
        test_examples, label_list, max_seq_length, tokenizer)

    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", 2)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=2)

    all_logits = None

    model.eval()
    nb_eval_steps, nb_eval_examples = 0, 0
    for step, batch in enumerate(test_dataloader):
        input_ids, input_mask, segment_ids = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            logits = logits.sigmoid()

        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    return pd.merge(pd.DataFrame(input_data), pd.DataFrame(all_logits, columns=label_list), left_index=True, right_index=True)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

args = {
    "chapter_data_dir": "path/to/augmented/data/with/chapter/codes/csv",  # we need this for getting the chapter label list
    "secondlevel_data_dir": "path/to/augmented/data/with/secondlevel/codes/csv",  # we need this for getting the second level label list
    "details_data_dir": "path/to/augmented/data/with/details/codes/csv",  # we need this for getting the details label list
    "test_data_dir": "path/to/test/data/csv",
    "chapter_model_dir": "path/to/chapter/model",
    "secondlevel_model_dir": "path/to/secondlevel/model",
    "details_model_dir": "path/to/details/model",
    "task_name": "clef_multilabel",
    "no_cuda": False,
    "bert_model": 'bert-base-multilingual-cased',
    "max_seq_length": 512,
    "do_lower_case": False,
    "eval_batch_size": 2,
}

# Setup GPU parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = 0
torch.cuda.set_device(gpu)

# Init data processor
tokenizer = BertTokenizer.from_pretrained(args['bert_model'], args['do_lower_case'])

processors = {
    "clef_multilabel_full": MultiLabelTextProcessor
}
task_name = 'clef_multilabel_full'.lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))
processor = processors[task_name](args['chapter_data_dir'])
label_list = processor.get_labels()
num_labels = len(label_list)


# Classify chapter code
model_state_dict = torch.load(args['chapter_model_dir'])
model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels=num_labels, state_dict=model_state_dict)
model.to(device)

predicted_chapters = predict(model, args['test_data_dir'], label_list, tokenizer)

# Classify second level
processor = processors[task_name](args['secondlevel_data_dir'])
label_list = processor.get_labels()
num_labels = len(label_list)
model_state_dict = torch.load(args['secondlevel_model_dir'])
model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels=num_labels, state_dict=model_state_dict)
model.to(device)
predicted_secondlevel = predict(model, args['test_data_dir'], label_list, tokenizer)
with open('../data/apriori_secondlevel_chapter.pickle', 'rb') as f:
    apriori_dict = pickle.loiad(f)

## Getting possible combinations of chapter and second level
chapter_codes = [i for i in predicted_chapters.columns if i != 'filename' and i != 'text']
secondlevel_codes = [i for i in predicted_secondlevel.columns if i != 'filename' and i != 'text']
combined_codes = []
for code in chapter_codes:
    for code2 in secondlevel_codes:
        com_code = code + code2
        combined_codes.append(com_code)

## Pruning combined codes by apriori
test_filenames = pd.read_csv(args['test_data_dir'])['filename']
predicted_categories = pd.DataFrame({'filename': test_filenames})
for i, filename in tqdm(predicted_secondlevel['filename'].items()):
        for code in combined_codes:
            chapter = code[0]
            secondlevel = code[1:3]
            predicted_categories[code] = apriori_dict[secondlevel][chapter] * predicted_chapters[chapter] * predicted_secondlevel[secondlevel]
category_codes = [col for col in predicted_categories.columns if col != 'filename' and col != 'text']

# Classify details
processor = processors[task_name](args['details_data_dir'])
label_list = processor.get_labels()
num_labels = len(label_list)
model_state_dict = torch.load(args['details_model_dir'])
model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels=num_labels, state_dict=model_state_dict)
model.to(device)
predicted_details = predict(model, args['test_data_dir'], label_list, tokenizer)

with open('../data/apriori_details_category.pickle', 'rb') as f:
    apriori_dict = pickle.load(f)

icd_codes = []
for detail in apriori_dict.keys():
    for category in category_codes:
        icd_code = category + '.' + detail
        icd_codes.append(icd_code)

# Prune by apriori and get final predicted ICD codes
predicted_results = pd.DataFrame({'filename': test_filenames})
for i, filename in tqdm(predicted_details['filename'].items()):
    for code in icd_codes:
        category = code[:3]
        detail = code[4:]
        predicted_results[code] = predicted_details[detail] * apriori_dict[detail][category]

submission_df = pd.DataFrame()
threshold = 0.001
for i, row in tqdm(predicted_results.iterrows()):
    filename = row[1]
    probs = row[2:].sort_values()[::-1]
    probs = [i for i in probs if i > threshold]
    for val in probs:
        for label in row[row == val].index:
            submission_df = submission_df.append({'id': filename, 'label': label}, ignore_index=True)

submission_df.to_csv('path/to/store/submission/data')
