from utils.load_huggingface_dataset import load_huggingface_dataset_train_and_test
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import random
import torch

format_s_dict = {
    'sst2': 'Review: {text}\nSentiment:{label}',
    'agnews': 'Article: {text}\nAnswer:{label}',
    'trec': 'Question: {text}\nAnswer Type:{label}',
    'emo': 'Dialogue: {text}\nEmotion:{label}',
    'amazon': 'Review: {text}\nSentiment:{label}'
}

label_dict = {'sst2': {0: ' Negative', 1: ' Positive'},
    'agnews': {0: ' World', 1: ' Sports', 2: ' Business', 3: ' Technology'},
    'trec': {0: ' Abbreviation', 1: ' Entity', 2: ' Description', 3: ' Person',
                          4: ' Location',
                          5: ' Number'},
    'emo': {0: ' Others', 1: ' Happy', 2: ' Sad', 3: ' Angry'},
    'amazon': {0: ' Negative', 1: ' Positive'}
}

class Demo:
    def __init__(self, task_name, sample, test=False):
        self.task_name = task_name
        self.text = sample['text']
        self.label = sample['label']
        self.test = test
        self.prompt = self._make_input_text(self.text, self.label)
   
    
    def _make_input_text(self, text, label):
        format_template = format_s_dict[self.task_name]
        if self.test:
            prompt = format_template.format(text=text, label='')
        else:
            prompt = format_template.format(text=text, label=label_dict[self.task_name][label])
        return prompt


class Prompts:
    def __init__(self, demos):
        prompt = [d.prompt for d in demos]
        self.prompts = "\n".join(prompt)
        self.label = demos[-1].label
        if self.prompts[-1] != ':':
            last_label_index = self.prompts.rfind(':')
            self.prompts = self.prompts[:last_label_index+1]


class ICLDataset(Dataset):
    def __init__(self, data_list, tokenizer, task_name, model_max_length):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.task_name = task_name
        self.model_max_length = model_max_length
        self.model_type = 'llama' if model_max_length == 4096 else 'gpt'
        self.max_input_len = self._max_leng()

    def __len__(self):
        return len(self.data_list)
    
    def _max_leng(self):
        max_input_len = 0
        for data in self.data_list:
            input_ids = self.tokenizer.encode(data.prompts)
            max_input_len = max(len(input_ids), max_input_len)
        return min(max_input_len, self.model_max_length)
    
    def _tokenize_input(self, text):
        model_input = self.tokenizer(text = text, padding=False, return_tensors="pt")
        return model_input.input_ids, model_input.attention_mask

    def _get_edge(self, token_ids):
        edge_tuples_first = []
        edge_tuples_last= []
        pad_id = self.tokenizer.pad_token_id

        token_ids = torch.reshape(token_ids,(-1,)).tolist() # turn token ids into a list
        token_ids = token_ids[:token_ids.index(pad_id)] if pad_id in token_ids else token_ids # truncate pad tokens

        label_words = list(label_dict[self.task_name].values())
        last_token_id = self.tokenizer.convert_tokens_to_ids(':')
        last_token_position = next(i for i in reversed(range(len(token_ids))) if token_ids[i] == last_token_id)

        label_word_ids = []
        for word in label_words:
            if self.model_type == 'gpt':
                label_word_ids.append(self.tokenizer.encode(word, add_special_tokens=False)[0])
            elif self.model_type == 'llama':
                label_word_ids.append(self.tokenizer.encode(word, add_special_tokens=False)[1])

        label_positions = []
        for idx, token_id in enumerate(token_ids):
            if token_id in label_word_ids and token_ids[idx-1] == last_token_id:
                label_positions.append(idx)
  
        for position in label_positions:
            edge_tuples_last.append([position, last_token_position])
            for indx, id in enumerate(token_ids[:position]):
                edge_tuples_first.append([indx, position])

        edge_index_first = torch.tensor(edge_tuples_first, dtype=torch.long).t().contiguous()
        edge_index_last = torch.tensor(edge_tuples_last, dtype=torch.long).t().contiguous()
        return edge_index_first, edge_index_last

    def __getitem__(self, index):
        text = self.data_list[index].prompts
        label = self.data_list[index].label
        
        input_ids, attention_mask = self._tokenize_input(text)
        
        edges_first, edges_last = self._get_edge(input_ids)

        return Data(num_nodes=len(input_ids), text=text, label=label, input_ids = input_ids, attention_mask = attention_mask, edge_index = edges_first, edge_index_last=edges_last)

class ICLDataModule(LightningDataModule):
    def __init__(self, task_name, num_test, num_demo_per_class, batch_size, tokenizer, model_max_length=1024):
        super().__init__()
        self.task_name = task_name
        self.num_test = num_test
        self.num_demo_per_class = num_demo_per_class
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.setup()
    
    def setup(self, stage=None):
        dataset = load_huggingface_dataset_train_and_test(self.task_name)
        train = dataset['train']
        test = dataset['test']
        self.num_test = min(self.num_test, len(test))
        self.test_samples = random.sample([Demo(self.task_name, s, test=True) for s in test], self.num_test)
        self.train_samples = [Demo(self.task_name, s) for s in train]
        random.shuffle(self.train_samples)
        self.val_samples = self.train_samples[:self.num_test]
        self.train_samples = self.train_samples[self.num_test:]

        selected_demo = []
        selected_auxiliary = []
        num_class = len(label_dict[self.task_name])

        for n in range(num_class):
            pool = [x for x in self.train_samples if x.label==n]
            selected_demo.append(pool.pop(random.randint(0, len(pool))))
            selected_auxiliary.extend(random.sample(pool, min(self.num_demo_per_class,len(pool))))
        self.selected_demo = selected_demo
        self.selected_auxiliary = selected_auxiliary

    def train_dataloader(self):
        train_samples = []

        for aux in self.selected_auxiliary:
            train_samples.append(Prompts(self.selected_demo+[aux]))

        train_dataset = ICLDataset(train_samples, self.tokenizer, self.task_name, self.model_max_length)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_prompts = []
        for test_sample in self.val_samples:
            prompt = Prompts(self.selected_demo + [test_sample])
            val_prompts.append(prompt)
        
        val_dataset = ICLDataset(val_prompts, self.tokenizer, self.task_name, self.model_max_length)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        test_prompts = []
        for test_sample in self.test_samples:
            prompt = Prompts(self.selected_demo + [test_sample])
            test_prompts.append(prompt)
        
        test_dataset = ICLDataset(test_prompts, self.tokenizer, self.task_name, self.model_max_length)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


