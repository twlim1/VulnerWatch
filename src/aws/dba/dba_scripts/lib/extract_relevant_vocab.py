"""
# Title: BERT Sequence Classification Explanation
Description: Finding relavent words that influenced BERT classification  
Adopted by: Saba Janamian  
Create date: 05/29/2021  

This notebook is adopted from https://github.com/frankaging/BERT_LRP  
Based on research paper https://arxiv.org/pdf/2101.00196.pdf
"""

#@title Import libraries
import os
import numpy as np
from numpy import newaxis as na
import torch
# from random import shuffle
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from tqdm import tqdm, trange
# this imports most of the helpers needed to eval the model

from transformers import BertModel, BertConfig
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW
import torch.nn as nn
import math

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger('Logger')


#@title Data containers
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_len):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.seq_len = seq_len

##############################################################################
#
# The function to back-out layerwise attended relevance scores.
#
##############################################################################
def rescale_lrp(post_A, inp_relevances):
    inp_relevances = torch.abs(inp_relevances)
    if len(post_A.shape) == 2:
        ref_scale = torch.sum(post_A, dim=-1, keepdim=True) + 1e-7
        inp_scale = torch.sum(inp_relevances, dim=-1, keepdim=True) + 1e-7
    elif len(post_A.shape) == 3:
        ref_scale = post_A.sum(dim=-1, keepdim=True).sum(dim=-1, keepdim=True) + 1e-7
        inp_scale = inp_relevances.sum(dim=-1, keepdim=True).sum(dim=-1, keepdim=True) + 1e-7
    scaler = ref_scale / inp_scale
    inp_relevances = inp_relevances * scaler
    return inp_relevances

def backprop_lrp_fc(weight, bias, activations, R, 
                    eps=1e-7, alpha=0.5, debug=False):
    beta = 1.0 - alpha
    
    weight_p = torch.clamp(weight, min=0.0)
    bias_p = torch.clamp(bias, min=0.0)    
    z_p = torch.matmul(activations, weight_p.T) + bias_p + eps
    s_p = R / z_p
    c_p = torch.matmul(s_p, weight_p)
    
    weight_n = torch.clamp(weight, max=0.0)
    bias_n = torch.clamp(bias, max=0.0)
    z_n = torch.matmul(activations, weight_n.T) + bias_n - eps 
    s_n = R / z_n
    c_n = torch.matmul(s_n, weight_n)

    R_c = activations * (alpha * c_p + beta * c_n)
    
    R_c = rescale_lrp(R, R_c)

    return R_c

def backprop_lrp_nl(weight, activations, R, 
                    eps=1e-7, alpha=0.5, debug=False):
    """
    This is for non-linear linear lrp.
    We use jacobian and first term of Taylor expansions.
    weight: [b, l, h_out, h_in]
    activations: [b, l, h_in]
    R: [b, l, h_out]
    """
    beta = 1.0 - alpha
    R = R.unsqueeze(dim=2) # [b, l, 1, h_out]
    activations = activations.unsqueeze(dim=2) # [b, l, 1, h_in]

    weight_p = torch.clamp(weight, min=0.0) 
    z_p = torch.matmul(activations, weight_p.transpose(2,3)) + eps
    s_p = R / z_p # [b, l, 1, h_out]
    c_p = torch.matmul(s_p, weight_p) # [b, l, 1, h_in]
    
    weight_n = torch.clamp(weight, max=0.0)
    z_n = torch.matmul(activations, weight_n.transpose(2,3)) + eps 
    s_n = R / z_n
    c_n = torch.matmul(s_n, weight_n)

    R_c = activations * (alpha * c_p + beta * c_n)
    
    R_c = R_c.squeeze(dim=2)
    R = R.squeeze(dim=2)
    R_c = rescale_lrp(R, R_c)

    return R_c

def rescale_jacobian(output_relevance, *input_relevances, batch_axes=(0,)):
    assert isinstance(batch_axes, (tuple, list))
    get_summation_axes = lambda tensor: tuple(i for i in range(len(tensor.shape)) if i not in batch_axes)
    ref_scale = abs(output_relevance).sum(dim=get_summation_axes(output_relevance), keepdim=True)
    inp_scales = [abs(inp).sum(dim=get_summation_axes(inp), keepdim=True) for inp in input_relevances]
    total_inp_scale = sum(inp_scales) + 1e-7
    input_relevances = [inp * (ref_scale / total_inp_scale) for inp in input_relevances]
    return input_relevances[0] if len(input_relevances) == 1 else input_relevances

def backprop_lrp_jacobian(jacobians, output, R, inps, eps=1e-7, alpha=0.5, batch_axes=(0,)):
    """
    computes input relevance given output_relevance using z+ rule
    works for linear layers, convolutions, poolings, etc.
    notation from DOI:10.1371/journal.pone.0130140, Eq 60
    """
    
    beta = 1.0 - alpha
    inps = [inp for inp in inps]

    reference_inputs = tuple(map(torch.zeros_like, inps))
    assert len(reference_inputs) == len(inps)

    flat_output_relevance = R.reshape([-1])
    output_size = flat_output_relevance.shape[0]

    assert len(jacobians) == len(inps)

    jac_flat_components = [jac.reshape([output_size, -1]) for jac in jacobians]
    # ^-- list of [output_size, input_size] for each input
    flat_jacobian = torch.cat(jac_flat_components, dim=-1)  # [output_size, combined_input_size]

    # 2. multiply jacobian by input to get unnormalized relevances, add bias
    flat_input = torch.cat([inp.reshape([-1]) for inp in inps], dim=-1)  # [combined_input_size]
    flat_reference_input = torch.cat([ref.reshape([-1]) for ref in reference_inputs], dim=-1)
    import operator
    from functools import reduce 
    num_samples = reduce(operator.mul, [output.shape[batch_axis] for batch_axis in batch_axes], 1)
    input_size_per_sample = flat_reference_input.shape[0] // num_samples
    flat_impact = (flat_jacobian * flat_input[None, :])
    # ^-- [output_size, combined_input_size], aka z_{j<-i}

    # 3. normalize positive and negative relevance separately and add them with coefficients
    flat_positive_impact = torch.clamp(flat_impact, min=0.0)
    flat_positive_normalizer = flat_positive_impact.sum(dim=0, keepdim=True) + eps
    flat_positive_relevance = flat_positive_impact / flat_positive_normalizer

    flat_negative_impact = torch.clamp(flat_impact, max=0.0)
    flat_negative_normalizer = flat_negative_impact.sum(dim=0, keepdim=True) - eps
    flat_negative_relevance = flat_negative_impact / flat_negative_normalizer
    flat_total_relevance_transition = alpha * flat_positive_relevance + beta * flat_negative_relevance

    flat_input_relevance = torch.einsum('o,oi->i', flat_output_relevance, flat_total_relevance_transition)
    # ^-- [combined_input_size]

    # 5. unpack flat_inp_relevance back into individual tensors
    input_relevances = []
    offset = 0
    for inp in inps:
        inp_size = inp.reshape([-1]).shape[0]
        inp_relevance = flat_input_relevance[offset: offset + inp_size].reshape(inp.shape)
        inp_relevance = inp_relevance.contiguous()
        input_relevances.append(inp_relevance)
        offset = offset + inp_size
    
    return rescale_jacobian(R, *input_relevances, batch_axes=batch_axes)

#@title BERT backward LRP containers
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# Main interface
class VocabRelevance:
    def __init__(self, model_folder, num_labels, model=None, tokenizer=None, device=None):
        #@title Hooks required for full lrp for BERT model
        self.func_inputs = collections.defaultdict(list)
        self.func_activations = collections.defaultdict(list)
        
        if device is None:
            #@title Set up device (GPU/CPU)
            # If there's a GPU available...
            if torch.cuda.is_available():    
                # Tell PyTorch to use the GPU.    
                self.device = torch.device('cuda')
                print('There are %d GPU(s) available.' % torch.cuda.device_count())
                print('We will use the GPU:', torch.cuda.get_device_name(0))
            # If not...
            else:
                print('No GPU available, using the CPU instead.')
                self.device = torch.device('cpu')
        else:
            self.device = device

        # If model and tokenizer are none they will get initilized 
        # in in the analysis_task()
        self.model = model
        self.tokenizer = tokenizer

        #@title Configuration and Constants
        self.model_folder = model_folder
        self.bert_config_file = os.path.join(model_folder, 'config.json')
        self.vocab_file = os.path.join(model_folder, 'vocab.txt')
        
        self.num_labels = num_labels
        self.learning_rate = 2e-5
        self.init_checkpoint = os.path.join(model_folder, 'pytorch_model.bin')
        self.warmup_proportion = 0.1
        self.num_train_steps = 20
        self.EVAL_BATCH_SIZE = 1

    @staticmethod
    def convert_examples_to_features(examples, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""
        features = []
        for (ex_index, example) in enumerate(tqdm(examples)):
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = []
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            seq_len = len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            label_id = int(example.label)
            features.append(
                    InputFeatures(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=label_id,
                            seq_len=seq_len))
            # end of for loop

        return features

    def get_inputivation(self, name):
        def hook(model, input, output):
            self.func_inputs[name] = [_in for _in in input]
        return hook

    def get_activation(self, name):
        def hook(model, input, output):
            self.func_activations[name] = output
        return hook

    def get_activation_multi(self, name):
        def hook(model, input, output):
            self.func_activations[name] = [_out for _out in output]
        return hook

    def init_hooks_lrp(self, model):
        """
        Initialize all the hooks required for full lrp for BERT model.
        """
        # in order to backout all the lrp through layers
        # you need to register hooks here.

        model.classifier.register_forward_hook(
            self.get_inputivation('model.classifier'))
        model.classifier.register_forward_hook(
            self.get_activation('model.classifier'))
        model.bert.pooler.dense.register_forward_hook(
            self.get_inputivation('model.bert.pooler.dense'))
        model.bert.pooler.dense.register_forward_hook(
            self.get_activation('model.bert.pooler.dense'))
        model.bert.pooler.register_forward_hook(
            self.get_inputivation('model.bert.pooler'))
        model.bert.pooler.register_forward_hook(
            self.get_activation('model.bert.pooler'))

        model.bert.embeddings.word_embeddings.register_forward_hook(
            self.get_activation('model.bert.embeddings.word_embeddings'))
        model.bert.embeddings.register_forward_hook(
            self.get_activation('model.bert.embeddings'))

        layer_module_index = 0
        for module_layer in model.bert.encoder.layer:
            
            ## Encoder Output Layer
            layer_name_output_layernorm = 'model.bert.encoder.' + str(layer_module_index) + \
                                    '.output.LayerNorm'
            module_layer.output.LayerNorm.register_forward_hook(
                self.get_inputivation(layer_name_output_layernorm))

            layer_name_dense = 'model.bert.encoder.' + str(layer_module_index) + \
                                    '.output.dense'
            module_layer.output.dense.register_forward_hook(
                self.get_inputivation(layer_name_dense))
            module_layer.output.dense.register_forward_hook(
                self.get_activation(layer_name_dense))

            layer_name_output = 'model.bert.encoder.' + str(layer_module_index) + \
                                    '.output'
            module_layer.output.register_forward_hook(
                self.get_inputivation(layer_name_output))
            module_layer.output.register_forward_hook(
                self.get_activation(layer_name_output))
            
            ## Encoder Intermediate Layer
            layer_name_inter = 'model.bert.encoder.' + str(layer_module_index) + \
                                    '.intermediate.dense'
            module_layer.intermediate.dense.register_forward_hook(
                self.get_inputivation(layer_name_inter))
            module_layer.intermediate.dense.register_forward_hook(
                self.get_activation(layer_name_inter))

            layer_name_attn_layernorm = 'model.bert.encoder.' + str(layer_module_index) + \
                                    '.attention.output.LayerNorm'
            module_layer.attention.output.LayerNorm.register_forward_hook(
                self.get_inputivation(layer_name_attn_layernorm))
            
            layer_name_attn = 'model.bert.encoder.' + str(layer_module_index) + \
                                    '.attention.output.dense'
            module_layer.attention.output.dense.register_forward_hook(
                self.get_inputivation(layer_name_attn))
            module_layer.attention.output.dense.register_forward_hook(
                self.get_activation(layer_name_attn))

            layer_name_attn_output = 'model.bert.encoder.' + str(layer_module_index) + \
                                    '.attention.output'
            module_layer.attention.output.register_forward_hook(
                self.get_inputivation(layer_name_attn_output))
            module_layer.attention.output.register_forward_hook(
                self.get_activation(layer_name_attn_output))
            
            layer_name_self = 'model.bert.encoder.' + str(layer_module_index) + \
                                    '.attention.self'
            module_layer.attention.self.register_forward_hook(
                self.get_inputivation(layer_name_self))
            module_layer.attention.self.register_forward_hook(
                self.get_activation_multi(layer_name_self))

            layer_name_value = 'model.bert.encoder.' + str(layer_module_index) + \
                                    '.attention.self.value'
            module_layer.attention.self.value.register_forward_hook(
                self.get_inputivation(layer_name_value))
            module_layer.attention.self.value.register_forward_hook(
                self.get_activation(layer_name_value))

            layer_name_query = 'model.bert.encoder.' + str(layer_module_index) + \
                                    '.attention.self.query'
            module_layer.attention.self.query.register_forward_hook(
                self.get_inputivation(layer_name_query))
            module_layer.attention.self.query.register_forward_hook(
                self.get_activation(layer_name_query))

            layer_name_key = 'model.bert.encoder.' + str(layer_module_index) + \
                                    '.attention.self.key'
            module_layer.attention.self.key.register_forward_hook(
                self.get_inputivation(layer_name_key))
            module_layer.attention.self.key.register_forward_hook(
                self.get_activation(layer_name_key))
            
            layer_module_index += 1

    #@title Sentence preparation
    @staticmethod
    def prep_sentence(sentence, label, set_type='test', debug=True):
        examples = []
        guid = "%s-%s" % (set_type, 0)
        text_a = str(sentence)
        text_b = None
        label = int(str(label))
        if debug:
            print("guid=",guid)
            print("text_a=",text_a)
            print("text_b=",text_b)
            print("label=",label)
        
        examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        
        return examples    

    #@title Other backward functions
    def backward_gradient(self, sensitivity_grads):
        classifier_out = self.func_activations['model.classifier']
        embedding_output = self.func_activations['model.bert.embeddings']
        sensitivity_grads = torch.autograd.grad(classifier_out, embedding_output, 
                                                grad_outputs=sensitivity_grads, 
                                                retain_graph=True)[0]
        return sensitivity_grads

    def backward_gradient_input(self, sensitivity_grads):
        classifier_out = self.func_activations['model.classifier']
        embedding_output = self.func_activations['model.bert.embeddings']
        sensitivity_grads = torch.autograd.grad(classifier_out, embedding_output, 
                                                grad_outputs=sensitivity_grads, 
                                                retain_graph=True)[0]
        return sensitivity_grads * embedding_output

    @staticmethod
    def backward_lat(input_ids, attention_probs):
        
        # backing out using the quasi-attention
        attention_scores = torch.zeros_like(input_ids, dtype=torch.float)
        # we need to distribution the attention on CLS to each head
        # here, we use grad to do this
        attention_scores[:,0] = 1.0
        attention_scores = torch.stack(12 * [attention_scores], dim=1).unsqueeze(dim=2)

        for i in reversed(range(12)):
            attention_scores = torch.matmul(attention_scores, attention_probs[i])
        
        attention_scores = attention_scores.sum(dim=1).squeeze(dim=1).unsqueeze(dim=-1).data
        return attention_scores

    #@title Evaluation top function
    def evaluate_with_hooks(self, test_dataloader, model):
        model.eval() # this line will deactivate dropouts
        test_loss, test_accuracy = 0, 0
        nb_test_steps, nb_test_examples = 0, 0
        pred_logits = []
        actual = []

        gs_scores = []
        gi_scores = []
        lrp_scores = []
        lat_scores = []

        inputs_ids = []
        seqs_lens = []

        for _, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):
            # see https://huggingface.co/transformers/model_doc/bert.html?highlight=attention_mask#transformers.BertModel.forward
            """
            input_ids: Indices of input sequence tokens in the vocabulary.
            attention_mask: Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]: 1 for tokens that are not masked, 0 for tokens that are masked.
            token_type_ids: Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]: 0 corresponds to a sentence A token, 1 corresponds to a sentence B token.
            
            """
            input_ids, input_mask, segment_ids, label_ids, seq_lens = batch
            max_seq_lens = max(seq_lens)[0]
            input_ids = input_ids[:,:max_seq_lens]
            input_mask = input_mask[:,:max_seq_lens]
            segment_ids = segment_ids[:,:max_seq_lens]

            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)
            seq_lens = seq_lens.to(self.device)

            """
            input_ids=None, 
            attention_mask=None, 
            token_type_ids=None, 
            position_ids=None, 
            head_mask=None, 
            inputs_embeds=None, 
            labels=None, 
            output_attentions=None, 
            output_hidden_states=None, 
            return_dict=None
            """

            
            output = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)
            tmp_test_loss = output['loss']
            logits_raw = output['logits']
            # hidden_states = output['hidden_states']
            ctx_attn = output['attentions']
            logits_t = F.softmax(logits_raw, dim=-1)
            logits = logits_t.detach().cpu().numpy()
            pred_logits.append(logits)

            label_ids = label_ids.to('cpu').numpy()
            actual.append(label_ids)
            
            outputs = np.argmax(logits, axis=1)
            tmp_test_accuracy=np.sum(outputs == label_ids)

            sensitivity_class = self.num_labels - 1

            # GS
            gs_score = torch.zeros(logits_t.shape)
            gs_score[:, sensitivity_class] = 1.0
            gs_score = gs_score.to(self.device)
            gs_score = logits_raw*gs_score
            gs_score = self.backward_gradient(gs_score)
            gs_score = torch.norm(gs_score, dim=-1)*torch.norm(gs_score, dim=-1)
            gs_scores.append(gs_score)

            # GI
            gi_score = torch.zeros(logits_t.shape)
            gi_score[:, sensitivity_class] = 1.0
            gi_score = gi_score.to(self.device)
            gi_score = logits_raw*gi_score
            gi_score = self.backward_gradient_input(gi_score)
            gi_score = torch.norm(gi_score, dim=-1)*torch.norm(gi_score, dim=-1)
            gi_scores.append(gi_score)

            # lat
            attention_scores = self.backward_lat(input_ids, ctx_attn)
            lat_scores.append(attention_scores.sum(dim=-1))

            # other meta-data
            input_ids = input_ids.cpu().data
            seq_lens = seq_lens.cpu().data
            inputs_ids.append(input_ids)
            seqs_lens.append(seq_lens)
        
            test_loss += tmp_test_loss.mean().item()
            test_accuracy += tmp_test_accuracy

            nb_test_examples += input_ids.size(0)
            nb_test_steps += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_examples
        
        result = collections.OrderedDict()
        result = {'test_loss': test_loss,
                    str(self.num_labels)+ '-class test_accuracy': test_accuracy}
        logger.info("***** Eval results *****")
        for key in result.keys():
            logger.info("  %s = %s\n", key, str(result[key]))
        # get predictions needed for evaluation
        pred_logits = np.concatenate(pred_logits, axis=0)
        actual = np.concatenate(actual, axis=0)
        pred_label = np.argmax(pred_logits, axis=-1)

        attribution_scores_state_dict = dict()
        attribution_scores_state_dict["inputs_ids"] = inputs_ids
        attribution_scores_state_dict["seqs_lens"] = seqs_lens
        attribution_scores_state_dict["gs_scores"] = gs_scores
        attribution_scores_state_dict["gi_scores"] = gi_scores
        # attribution_scores_state_dict["lrp_scores"] = gi_scores # lrp_scores
        attribution_scores_state_dict["lat_scores"] = lat_scores

        logger.info("***** Finish Attribution Backouts *****")
        return attribution_scores_state_dict

    #@title Analysis top function
    def analysis_task(self, test_examples):
        # bert_config = BertConfig.from_json_file(self.bert_config_file)
        # logger.info("*** Model Config ***")
        # logger.info(bert_config.to_json_string())    

        ################
        # tokenizer
        ################
        if self.tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(self.model_folder, do_lower_case=True)
        else:
            tokenizer = self.tokenizer
        
        ################
        # model
        ################
        if self.model is None:
            model = BertForSequenceClassification.from_pretrained(self.model_folder)
        else:
            model = self.model
        
        self.init_hooks_lrp(model)
        # dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        # classifier = nn.Linear(bert_config.hidden_size, self.num_labels)

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
                    {'params': [p for n, p in model.named_parameters() 
                        if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                    {'params': [p for n, p in model.named_parameters() 
                        if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}]

        ################
        # optimizer
        ################
        # check if we should use this model.parameters()
        optimizer = AdamW(optimizer_parameters,
                        lr=self.learning_rate)

        model = model.to(self.device) # send the model to device



        test_features = self.convert_examples_to_features(
                test_examples,
                128, # TODO allow user to set the max seq length
                tokenizer)
        
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_seq_len = torch.tensor([[f.seq_len] for f in test_features], dtype=torch.long)
        
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_seq_len)
        test_dataloader = DataLoader(test_data, batch_size=self.EVAL_BATCH_SIZE, shuffle=False)

        score_dict = self.evaluate_with_hooks(test_dataloader, model)
        return score_dict

    #@title Result interpretation functions
    @staticmethod
    def inverse_mapping(vocab_dict):
        inverse_vocab_dict = {}
        for k, v in vocab_dict.items():
            inverse_vocab_dict[v] = k
        return inverse_vocab_dict

    @staticmethod
    def translate(token_ids, vocab):
        tokens = []
        for _id in token_ids.tolist():
            tokens.append(vocab[_id])
        return tokens

    @staticmethod
    def load_vocab(vocab_file, pretrain=True):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        index = 0
        with open(vocab_file, "r") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break
                token = token.strip()
                vocab[token] = index
                index += 1
        return vocab


    def load_attribution_scores(self, vocab_data_dir, inputs_ids, seqs_lens, raw_attribution_scores, min_freq=0, 
                                consider_speicial_tokens=False, normalized=True, min_length=0):
        vocab = self.inverse_mapping(self.load_vocab(vocab_data_dir, pretrain=False))
        word_lrp = collections.OrderedDict()
        word_lrp_list = []
        word_lrp_list_original = []
        sentence_lrp = []
        for batch_idx in range(len(inputs_ids)):
            for seq_idx in range(inputs_ids[batch_idx].shape[0]):
                seq_len = seqs_lens[batch_idx][seq_idx].tolist()[0]
                tokens = self.translate(inputs_ids[batch_idx][seq_idx], vocab)[:seq_len]
                attribution_scores = raw_attribution_scores[batch_idx][seq_idx][:seq_len]
                if normalized:
                    # sentence_attribution_scores = F.softmax(torch.abs(attribution_scores), dim=-1).tolist()
                    sentence_max = torch.max(torch.abs(attribution_scores), dim=-1)[0]
                    sentence_attribution_scores = \
                        (torch.abs(attribution_scores)/sentence_max).tolist()
                else:
                    sentence_attribution_scores = attribution_scores.tolist()
                if len(tokens) >= min_length:
                    assert(len(tokens) == len(sentence_attribution_scores))
                    s_lrp = list(zip(tokens, sentence_attribution_scores))
                    sentence_lrp.append(s_lrp)
                    for i in range(len(s_lrp)):
                        token = s_lrp[i][0]
                        score = s_lrp[i][1]
                        position = i

                        word_lrp_list.append((token, score, position))
                        word_lrp_list_original.append((token, score, position))
                        if token in word_lrp.keys():
                            word_lrp[token].append({'score': score, 'position': position})
                        else:
                            word_lrp[token] = [{'score': score, 'position': position}]

        filter_word_lrp = {}
        for k, v in word_lrp.items():
            all_scores = [s['score'] for s in v]
            all_pos = [s['position'] for s in v]
            if len(all_scores) > min_freq:
                filter_word_lrp[k] = (sum(all_scores)*1.0/len(all_scores), max(all_pos))
        filter_word_lrp = [(k, v[0], v[1]) for k, v in filter_word_lrp.items()] 
        filter_word_lrp.sort(key = lambda x: x[1], reverse=True)  
        word_lrp_list.sort(key = lambda x: x[1], reverse=True)

        # print('--------------')
        # print("word_lrp_list_original:", word_lrp_list_original)
        # print("word_lrp:", word_lrp)
        # print("filter_word_lrp:", filter_word_lrp)
        # print("word_lrp_list: ", word_lrp_list)
        # print("sentence_lrp: ", sentence_lrp)
        # print('--------------')
        return filter_word_lrp, word_lrp_list, sentence_lrp, word_lrp_list_original

    def load_attribution_meta(self, dataset_dict):
        attribution_meta = {}
        for item in ["gs_scores", "gi_scores", "lat_scores"]: # "lrp_scores",
            filtered_word_rank, raw_word_rank, sentence_revelance_score, ordered_words = \
                self.load_attribution_scores(self.vocab_file,
                                        inputs_ids=dataset_dict["inputs_ids"], 
                                        seqs_lens=dataset_dict["seqs_lens"],
                                        raw_attribution_scores=dataset_dict[item])
            attribution_meta[item] = {"filtered_word_rank": filtered_word_rank, 
                                    "raw_word_rank": raw_word_rank, 
                                    "sentence_revelance_score": sentence_revelance_score,
                                    "ordered_words": ordered_words}
        return attribution_meta
    
    @staticmethod
    def fix_subtoken(token, position, list_of_tokens):
        index = position
        all_tokens = [t[0] for t in list_of_tokens]

        # print("sub start:", index, token)
        # first find the last subtoken
        while index < len(all_tokens):
            try:
                next_token = all_tokens[index + 1]
            except IndexError:
                break
            
            if '##' in next_token:
                index += 1
            else:
                break

        # print("sub moved to:", index, all_tokens[index])
        constructed_word = all_tokens[index].lstrip('##')
        
        while index > 0:
            try:
                prev_token = all_tokens[index - 1]
            except IndexError:
                break

            if "##" in prev_token:
                constructed_word = f'{prev_token.lstrip("##")}{constructed_word}'
                # print("sub prev: ", index, prev_token)
                index -= 1
            elif '##' not in prev_token:
                constructed_word = f'{prev_token}{constructed_word}'
                break
            else:
                raise ValueError("----> ", prev_token)
                pass
        
        # print("sub final: ",index, prev_token)
        return constructed_word

    @staticmethod
    def check_for_subtoken(token, position, list_of_tokens):
        index = position
        constructed_word = token
        all_tokens = [t[0] for t in list_of_tokens]
        # print(all_tokens)
        # print("up start:", position, constructed_word)
        while index < len(all_tokens):
            try:
                next_token = all_tokens[index + 1]
                # print("next_token:", index, next_token)
            except IndexError:
                break
            
            if "##" not in next_token:
                break
            else:
                constructed_word = f'{constructed_word}{next_token.lstrip("##")}'
                index += 1
        
        # print("up final:",index, constructed_word)
        return constructed_word

    def report_for_gui(self, attribution_meta, score_type="gs_scores", k=20, filtered=True):
        item_words = collections.OrderedDict()
        index = 0
        if filtered:
            lenght = len(attribution_meta[score_type]["filtered_word_rank"])
        else:
            lenght = len(attribution_meta[score_type]["raw_word_rank"])

        for i in range(0, lenght):
            word_rank = None
            if filtered:
                word_rank = attribution_meta[score_type]["filtered_word_rank"]
            else:
                word_rank = attribution_meta[score_type]["raw_word_rank"]

            # print("............>", i, len(attribution_meta[score_type]["raw_word_rank"]), len(word_rank))
            if not any([word_rank[i][0] == '[CLS]', word_rank[i][0] == '[SEP]']):
                if "##" in word_rank[i][0]:
                    token = self.fix_subtoken(word_rank[i][0],
                                        word_rank[i][2],
                                        attribution_meta[score_type]['ordered_words'])
                else:
                    token = self.check_for_subtoken(word_rank[i][0],
                                            word_rank[i][2],
                                                attribution_meta[score_type]['ordered_words'])
                
                if token in item_words.keys():
                    item_words[token] = max(item_words[token], word_rank[i][1])
                else:
                    item_words[token] = word_rank[i][1]

                index += 1
            
            if index >= k:
                break
        words = [(w, round(s, 2)) for w,s in item_words.items()]    
        return words


if __name__ == '__main__':
    # CVE-2021-21659 https://nvd.nist.gov/vuln/detail/CVE-2021-21659
    example_sentence = 'Jenkins URLTrigger Plugin 0.48 and earlier does not configure its XML parser to prevent XML external entity (XXE) attacks.'
    example_label = 0
    vocab_o = VocabRelevance(model_folder='../../../../../models/AV', num_labels=4)
    test_sentence = vocab_o.prep_sentence(example_sentence, example_label)
    analysis_dict = vocab_o.analysis_task(test_sentence)
    attribution_meta = vocab_o.load_attribution_meta(analysis_dict)
    print(vocab_o.report_for_gui(attribution_meta))
