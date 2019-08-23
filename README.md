# How to use this repository
1. Make sure that the tasks that you are aiming to do are either binomial or multiclass classification.
2. Format your classification datasets so that the train, dev and test sets are all in one folder, under the names `train.tsv`, `dev.tsv` and `test.tsv`.
3. These tsv files must have headers for each column, and the columns which contain the text you want to analyse must be called `text_A` and `text_B`.
If you have a two string classification problem (E.g. Natural Language Inference), then the first bit of text (E.g. Premise) should be under the `text_A` header and the second bit of text (E.g. Hypothesis) should be under the `text_B` header.
If you only a one string classification problem (E.g. sentiment analysis), then omit the `text_B` column header.
4. The label of the class of each observation should be listed in a column named `label`.
5. These files should then be put into a folder named after the task, and this folder should be placed in the `./data` subdirectory of this project.
An example of this can be seen in `./data/example_task`.
6. Run the following command to train an MTL model on these tasks:
```
python Run_MTL_Model.py --max_seq_length=50 --batch_size=32 --num_finetune_epochs=20
```
Where the `max_seq_length` is the max sequence length of the BERT model. This is the upper limit of the amount of tokens that you can pass into BERT. Default is 150, but smaller is faster and more memory efficient.  
`batch_size` is the batch size of the datasets when they are trained. Larger is faster, but is more memory intensive, meaning that a balance should be struck depending on your computational resources. Default is 32.  
`num_finetune_epochs` is the number of epochs in which the model is trained. Default is 1.

Training results will then be printed.

## Project TODO:
* Support fine-tuning and fine-tuning settings (E.g. disambiguate fine tuning epochs from pre-training epochs).
* Support non-classification tasks:
    * Regression tasks
    * Seq to seq tasks
    * Seq tagging tasks
* Support complex model configuration
    * E.g. Instead of simply having a shared BERT input and individual heads, make it so that the BERT output could be combined with other inputs before reaching heads, or any other arbitrary level of complexity.
