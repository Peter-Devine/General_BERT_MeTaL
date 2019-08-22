####### Temporary hack to use the pip version of MeTaL ##############
# See https://github.com/HazyResearch/metal/issues/182 for more
import os
os.environ['METALHOME'] = "test"
####################################################################

import argparse
import re

import torch
from torch import nn

import metal
from metal.mmtl.task import ClassificationTask
from metal.mmtl import MetalModel
from metal.mmtl.trainer import MultitaskTrainer
from metal.mmtl.payload import Payload

from pytorch_transformers.modeling_bert import BertModel
from pytorch_transformers.modeling_bert import BertConfig

# Import custom utils
import utils.Task_creator
import utils.LM_MTL_Data_Handler

parser=argparse.ArgumentParser()
parser.add_argument('--max_seq_length', default="150", help='Max size of the input')
parser.add_argument('--batch_size', default="32", help='Batch size of training')
parser.add_argument('--data_paths', help='List of all the root file paths of all the datasets (comma separated, e.g. /home/documents/dataset1.tsv,/home/documents/dataset2.tsv)')
parser.add_argument('--input_and_label_columns', help='Arrow (>) separated list of comma separated lists of column names which correspond to the text_A, text_B and label columns are for each dataset (same order as before). If no text_B is available for a task, just leave that column blank, but include all commas. E.g. question_col,context_col,answer_col>text,,class')
parser.add_argument('--num_finetune_epochs', default="1", help='How many epochs to run fine-tuning for')

args = parser.parse_args()
MAX_SEQ_LENGTH = int(args.max_seq_length)
BATCH_SIZE = int(args.batch_size)
DATA_PATHS = args.data_paths.split(",")

DATA_COLUMNS = [{"text_A": column_list.split(",")[0], "text_B": column_list.split(",")[1], "label": column_list.split(",")[2]} for column_list in args.input_and_label_columns.split(">")]
NUM_EPOCHS = int(args.num_finetune_epochs)

assert (len(DATA_PATHS) == len(DATA_COLUMNS)), "Number of datasets different to supplied sets of columns for each dataset"

# Create a list of dictionaries that each have data about one specific dataset (name, data path and target columns)
dataset_dict_list = []
for i in range(len(DATA_PATHS)):
    split_path = os.path.split(DATA_PATHS[i])
    
    # Make dataset name from name of folder in which data files are
    dataset_name = split_path[1]
    if not dataset_name:
        dataset_name = os.path.split(split_path[0])[1]
    
    dataset_dict_list.append({"dataset_name": dataset_name, "data_path": DATA_PATHS[i], "data_columns": DATA_COLUMNS[i]})

class MetalFriendlyBert(nn.Module):
    
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(MetalFriendlyBert, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        self.bert_layer.config.max_position_embeddings = MAX_SEQ_LENGTH
        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x_tensor = torch.tensor(x, dtype=torch.long, device=self.device)
        
        input_ids = x_tensor[:,0,:]
        token_type_ids = x_tensor[:,1,:]
        attention_masks = x_tensor[:,2,:]
        
        # NB: No need to input position IDs, as they are generated automatically as long as the input is inputted consistent with
        # the original data. I.e. the first token in the input is the first word of the inputted text.
        
        bert_output = self.bert_layer.forward(input_ids = input_ids, token_type_ids = token_type_ids, attention_mask=attention_masks)
        
        # HuggingFace's implementation of BERT outputs a tuple with both the final layer output at every token size,
        # as well as just the first token position final output (where the [CLS] token is).
        #
        # The documentation can be found here: https://github.com/huggingface/pytorch-transformers/blob/b0b9b8091b73f929306704bd8cd62b712621cebc/pytorch_transformers/modeling_bert.py#L628
        
        # first_token_final_hidden_state = bert_output[1]
        # averaged_all_token_final_hidden_state_averaged = bert_output[0].mean(dim=1)
        
        return bert_output[1]


# Download BERT as necessary and save as shared module
bert_layer = MetalFriendlyBert()

# Create list of names in which dataset names will be put as they are created
dataset_name_list = []

for dataset_dict in dataset_dict_list:
    
    # Get dataset data from dataset_dict, and split it into inputs and outputs
    split_datasets = LM_MTL_Data_Handler.make_list_dict_from_tsv(dataset_dict)
    
    train_dataset = split_datasets["train"]
    dev_dataset = split_datasets["dev"]
    test_dataset = split_datasets["test"]
    
    train_inputs = [train_datum["inputs"] for train_datum in train_dataset]
    dev_inputs = [dev_datum["inputs"] for dev_datum in dev_dataset]
    test_inputs = [test_datum["inputs"] for test_datum in test_dataset]
    
    train_outputs = [train_datum["outputs"] for train_datum in train_dataset]
    dev_outputs = [dev_datum["outputs"] for dev_datum in dev_dataset]
    test_outputs = [test_datum["outputs"] for test_datum in test_dataset]
    
    # Make task name the same as the dataset name with a monotonically increasing numeral prepended
    task_name = str(len(dataset_name_list)) + dataset_dict["dataset_name"]
    dataset_name_list.append(task_name)
    
    # Find the total number of class within this dataset
    number_of_classes = LM_MTL_Data_Handler.get_total_number_of_classes(train_outputs, dev_outputs, test_outputs)
    
    # Make a list containing train, dev, test data of each task. Also included is task name, batch size and number of classes
    # that the task is predicting over.
    tasks = [
        Task_creator.Task(
            task_name = task_name,
            no_of_lables = number_of_classes,
            batch_size = BATCH_SIZE,
            train_inputs = train_inputs,
            dev_inputs = dev_inputs,
            test_inputs = test_inputs,
            train_outputs = train_outputs,
            dev_outputs = dev_outputs,
            test_outputs = test_outputs
        )
    ]
    
    
    classification_task_list = []
    # Make model constitution of each task. Every task has the same shared BERT input module, and its own head module, with
    # output nodes equal to the number of classes in the task.
    for task in tasks:
        classification_task = ClassificationTask(
            name= task.task_name, 
            input_module= bert_layer, 
            head_module= torch.nn.Linear(768, task.no_of_lables)
        )
        classification_task_list.append(classification_task)



model = MetalModel(classification_task_list)


# Prepare task payloads
    
# Creates a list of payloads ready to be ingested by the MeTaL trainer.
def create_payloads(database_tasks):
    payloads = []
    splits = ["train", "valid", "test"]
    
    for i, task in enumerate(database_tasks):
    
        payload_train_name = f"Payload{i}_train"
        payload_dev_name = f"Payload{i}_dev"
        payload_test_name = f"Payload{i}_test"
    
        task_name = task.task_name
    
        batch_size = task.batch_size
    
        train_inputs = task.train_inputs
        dev_inputs = task.dev_inputs
        test_inputs = task.test_inputs

        train_X_tensor = LM_MTL_Data_Handler.format_observation_for_bert_classifier(train_inputs, seq_len=MAX_SEQ_LENGTH)
        dev_X_tensor = LM_MTL_Data_Handler.format_observation_for_bert_classifier(dev_inputs, seq_len=MAX_SEQ_LENGTH)
        test_X_tensor = LM_MTL_Data_Handler.format_observation_for_bert_classifier(test_inputs, seq_len=MAX_SEQ_LENGTH)

        train_X = {"data": train_X_tensor}
        dev_X = {"data": dev_X_tensor}
        test_X = {"data": test_X_tensor}

        train_Y, dev_Y, test_Y = LM_MTL_Data_Handler.format_labels_for_bert_classifier(task.train_outputs, task.dev_outputs, task.test_outputs)
    
        train_Y = torch.tensor(train_Y, dtype=torch.long)
        dev_Y = torch.tensor(dev_Y, dtype=torch.long)
        test_Y = torch.tensor(test_Y, dtype=torch.long)
        
        payload_train = Payload.from_tensors(task_name, train_X, train_Y, task_name, "train", batch_size=batch_size)
        payload_dev = Payload.from_tensors(task_name, dev_X, dev_Y, task_name, "valid", batch_size=batch_size)
        payload_test = Payload.from_tensors(task_name, test_X, test_Y, task_name, "test", batch_size=batch_size)
    
        payloads.append(payload_train)
        payloads.append(payload_dev)
        payloads.append(payload_test)
    return payloads

payloads = create_payloads(tasks)

# Set up the trainer for the payloads and tasks
trainer = MultitaskTrainer(seed=101)

# Run the model to train
training_metrics = trainer.train_model(model, payloads, n_epochs=NUM_EPOCHS)
