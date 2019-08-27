import argparse
import re
import inspect
import os

####### Temporary hack to use the pip version of MeTaL ##############
# See https://github.com/HazyResearch/metal/issues/182 for more
os.environ['METALHOME'] = "test"
####################################################################

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
import utils.Task_creator as Task_creator
import utils.LM_MTL_Data_Handler as LM_MTL_Data_Handler

parser=argparse.ArgumentParser()
parser.add_argument('--max_seq_length', default="150", help='Max size of the input')
parser.add_argument('--batch_size', default="32", help='Batch size of training')
parser.add_argument('--num_mtl_train_epochs', default="20", help='How many epochs to run fine-tuning for')

args = parser.parse_args()
MAX_SEQ_LENGTH = int(args.max_seq_length)
BATCH_SIZE = int(args.batch_size)
NUM_EPOCHS = int(args.num_mtl_train_epochs)

# Create special module that can be passed into the MTL model which splits up the data so that it can be used in BERT
# I.e., this module splits up the inputted data into tokens, token type ids and attention masks
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

# Get current working directory in order to find the "/data" subdirectory
cwd = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Create list of names in which dataset names will be put as they are created
dataset_name_list = [f for f in os.listdir(os.path.join(cwd, 'data')) if not os.path.isfile(f)]

tasks = []

for dataset_name in dataset_name_list:

    # Get dataset data from dataset_dict, and split it into inputs and outputs
    split_datasets = LM_MTL_Data_Handler.make_list_dict_from_tsv(dataset_name, cwd)

    train_dataset = split_datasets["train"]
    dev_dataset = split_datasets["dev"]
    test_dataset = split_datasets["test"]

    train_inputs = [train_datum["inputs"] for train_datum in train_dataset]
    dev_inputs = [dev_datum["inputs"] for dev_datum in dev_dataset]
    test_inputs = [test_datum["inputs"] for test_datum in test_dataset]

    train_outputs = [train_datum["outputs"] for train_datum in train_dataset]
    dev_outputs = [dev_datum["outputs"] for dev_datum in dev_dataset]
    test_outputs = [test_datum["outputs"] for test_datum in test_dataset]

    # Make task name the same as the dataset name
    task_name = dataset_name

    # Find the total number of class within this dataset
    number_of_classes = LM_MTL_Data_Handler.get_total_number_of_classes(train_outputs, dev_outputs, test_outputs)

    # Make a list containing train, dev, test data of each task. Also included is task name, batch size and number of classes
    # that the task is predicting over.
    tasks.append(Task_creator.Task(
            task_name = task_name,
            no_of_lables = number_of_classes,
            batch_size = BATCH_SIZE,
            train_inputs = train_inputs,
            dev_inputs = dev_inputs,
            test_inputs = test_inputs,
            train_outputs = train_outputs,
            dev_outputs = dev_outputs,
            test_outputs = test_outputs
        ))

    # Make model constitution of each task. Every task has the same shared BERT input module, and its own head module, with
    # output nodes equal to the number of classes in the task.
classification_task_list = []

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

# Create average accuracy metric for MeTaL to select models on.
# The model with the highest average validation score is loaded from a checkpoint at the end of training.
def accuracy(metrics_hist):
    metrics_agg = {}
    dev_accuracies = []
    for key in sorted(metrics_hist.keys()):
        if "accuracy" in key and "dev" in key:
            dev_accuracies.append(metrics_hist[key])

    average_accuracy = sum(dev_accuracies)/len(dev_accuracies)

    metrics_agg["model/dev/all/accuracy"] = average_accuracy

    return metrics_agg

# Run the model to train
training_metrics = trainer.train_model(
                                    model, 
                                    payloads, 
                                    n_epochs=NUM_EPOCHS,
                                    optimizer_config={
                                        "optimizer": "sgd",
                                        "optimizer_common": {"lr": 0.005},
                                        "sgd_config": {"momentum": 0.01},
                                    },
                                    log_every=1,
                                    checkpoint_config={
                                        "checkpoint_metric": "model/dev/all/accuracy",
                                        "checkpoint_metric_mode": "max",
                                        "checkpoint_dir": os.path.join(cwd, "checkpoints")
                                    },
                                    progress_bar=False,                              
                                    metrics_config={
                                        "aggregate_metric_fns": [accuracy]
                                    },
                                    verbose=True
)

print(training_metrics)
