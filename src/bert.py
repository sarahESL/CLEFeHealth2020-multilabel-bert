from transformers import BertModel, BertConfig
from transformers.modeling_bert import BertPreTrainedModel
import torch


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for multi-label classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `bert_model`: Name of the pretrained model.
        `num_labels`: the number of classes for the classifier. Default = 25, the number of unique chapter codes.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary.
            `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels-1].
    Outputs:
        if `labels` is not `None`:
            Outputs the BinaryCrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    """

    def __init__(self, config, bert_model, num_labels=25):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels

        model_config = BertConfig.from_pretrained(bert_model, output_hidden_states=True)
        self.bert = BertModel.from_pretrained(bert_model, config=model_config)

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        pooled_output = self.bert(input_ids, token_type_ids, attention_mask)[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits
