import pandas as pd
import os
import sys
import logging
from tqdm import tqdm


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the sequence.
            labels: (Optional) [string]. The label of the example. Specify
            this for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters."""

    def get_train_examples(self, data_dir, size=-1):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, data_file_name, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class MultiLabelTextProcessor(DataProcessor):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None

    def get_train_examples(self, data_dir, size=-1):
        filename = 'train.csv'
        logger.info("LOOKING AT {}".format(data_dir))
        if size == -1:
            data_df = pd.read_csv(data_dir, escapechar='\\')
            return self._create_examples(data_df, "train")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename))
            return self._create_examples(data_df.sample(size), "train")

    def get_dev_examples(self, data_dir, size=-1):
        filename = 'dev.csv'
        if size == -1:
            data_df = pd.read_csv(data_dir, escapechar='\\')
            return self._create_examples(data_df, "dev")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename))
            return self._create_examples(data_df.sample(size), "dev")

    def get_test_examples(self, data_dir, filename, size=-1):
        data_df = pd.read_csv(os.path.join(data_dir, filename))
        if size == -1:
            return self._create_examples(data_df, "test")
        else:
            return self._create_examples(data_df.sample(size), "test")

    def get_labels(self):
        if self.labels is None:
            self.labels = [val for (i, val) in enumerate(pd.read_csv(self.data_dir).columns) if val != 'filename' and val != 'text']
        return self.labels

    def _create_examples(self, df, set_type, labels_available=True):
        examples = []
        labels_ind = [i for (i, val) in enumerate(df.columns) if val != 'filename' and val != 'text']
        filename_ind = [ind for (ind, val) in enumerate(df.columns) if val == 'filename']
        text_ind = [ind for (ind, val) in enumerate(df.columns) if val == 'text']
        for (i, row) in enumerate(df.values):
            guid = row[filename_ind][0]
            text = row[text_ind][0]
            if labels_available:
                labels = row[labels_ind]
            else:
                raise ValueError('Make sure labels are available!')
            examples.append(
                InputExample(guid=guid, text=text, labels=labels))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in tqdm(enumerate(examples)):
        tokens = tokenizer.tokenize(example.text)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        labels_ids = []
        for label in example.labels:
            labels_ids.append(float(label))

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.labels, labels_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_ids=labels_ids))
    return features
