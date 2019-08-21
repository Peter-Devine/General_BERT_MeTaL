class Task:
    task_name = ""
    no_of_lables = 0
    batch_size = 0
    
    train_inputs = []
    dev_inputs = []
    test_inputs = []
    
    train_outputs = []
    dev_outputs = []
    test_outputs = []
    
    def __init__(self, task_name, batch_size, no_of_lables, train_inputs=None, dev_inputs=None, test_inputs=None, train_outputs=None, dev_outputs=None, test_outputs=None):
        self.task_name = task_name
        self.batch_size = batch_size
        self.no_of_lables = no_of_lables
        
        self.train_inputs = train_inputs
        self.dev_inputs = dev_inputs
        self.test_inputs = test_inputs
        
        self.train_outputs = train_outputs
        self.dev_outputs = dev_outputs
        self.test_outputs = test_outputs


# In[ ]:




