from utils.preproc import ICLDataModule, label_dict
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam, AdamW
from models.gpt2_gnn import GPT2LMHeadModelwithGNN
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, get_linear_schedule_with_warmup
from peft import get_peft_model, PrefixTuningConfig, TaskType, LoraConfig
from adapters import GPT2AdapterModel
import evaluate
import argparse
import random
import string


OPTIMIZERS = {'Adam': Adam, 'AdamW': AdamW}

class ICLforClassification(LightningModule):
    def __init__(
        self, model_name_or_path, task_name, exp_type, tokenizer, run_id, optimizer, learning_rate, warmup_steps, training_steps, do_schedule, virtual_token):
        super().__init__()

        self.run_id = run_id
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.training_steps = training_steps
        self.model_name_or_path = model_name_or_path
        self.optimizer = optimizer
        self.task_name = task_name
        self.do_schedule = do_schedule

        self.tokenizer = tokenizer
        self.label_map = {tokenizer.encode(v, add_special_tokens=False)[0]: k for k, v in
                          label_dict[task_name].items()}
        self.interest_index = list(self.label_map.keys())
        
        self.save_hyperparameters()

        self.exp_type = exp_type
        config = GPT2Config.from_pretrained(model_name_or_path)

        if self.exp_type == 'gnn':
            self.model = GPT2LMHeadModelwithGNN.from_pretrained(self.model_name_or_path, config=config)
            for name, param in self.model.named_parameters():
                if "gnn" not in name:
                    param.requires_grad = False
        elif self.exp_type == 'fpft':
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name_or_path, config=config)
        elif self.exp_type == 'prefix':
            model = GPT2LMHeadModel.from_pretrained(self.model_name_or_path, config=config)
            peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=virtual_token)
            self.model = get_peft_model(model, peft_config)
        elif self.exp_type == 'adapter':
            self.model = GPT2AdapterModel.from_pretrained(self.model_name_or_path, config=config)
            self.model.add_adapter(self.task_name)
            self.model.add_causal_lm_head('LMhead')
            self.model.train_adapter(self.task_name)
        elif self.exp_type == 'lora':
            model = GPT2LMHeadModel.from_pretrained(self.model_name_or_path, config=config)
            peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
            self.model = get_peft_model(model, peft_config)

        self.accuracy = evaluate.load("accuracy",experiment_id=run_id)

    def forward(self, input_ids, attention_mask, edge_index_first, edge_index_last):
        if self.exp_type == 'gnn':
            output = self.model(input_ids=input_ids,attention_mask=attention_mask, edge_index_first=edge_index_first,edge_index_last=edge_index_last)
        else:
            output = self.model(input_ids=input_ids,attention_mask=attention_mask)
        logits = output.logits[:, -1, self.interest_index]
        return logits

    def training_step(self, batch, batch_idx):
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        label = batch.label
        edge_index_first = batch.edge_index
        edge_index_last = batch.edge_index_last
        logits = self(input_ids, attention_mask, edge_index_first, edge_index_last)

        loss = F.cross_entropy(logits, label)

        self.log(self.task_name+"_train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        label = batch.label.tolist() 
        edge_index_first = batch.edge_index
        edge_index_last = batch.edge_index_last
        logits = self(input_ids, attention_mask, edge_index_first, edge_index_last)
        probs = F.softmax(logits,dim=-1)
   
        predictions = [probs.argmax().item()]

        acc = self.accuracy.compute(predictions=predictions, references=label)['accuracy']

        self.log(self.task_name+"_val_accuracy", acc, prog_bar=True, logger=True)
        return {self.task_name+"_val_accuracy", acc }

    def test_step(self, batch, batch_idx):
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        label = batch.label.tolist() 
        edge_index_first = batch.edge_index
        edge_index_last = batch.edge_index_last
        logits = self(input_ids, attention_mask, edge_index_first, edge_index_last)
        probs = F.softmax(logits,dim=-1)
   
        predictions = [probs.argmax().item()]

        acc = self.accuracy.compute(predictions=predictions, references=label)['accuracy']

        self.log(self.task_name+"_accuracy", acc, prog_bar=True, logger=True)
        return {self.task_name+"_accuracy", acc }

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        if self.do_schedule:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.training_steps)
            return [optimizer], [scheduler]
        else:
            return [optimizer]
    
    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument("--mode", type=str, default='do_train')
        parser.add_argument("--exp_type", type=str, default='gnn')
        parser.add_argument("--task_name", type=str, default='sst2')
        parser.add_argument("--num_test", type=int, default=1000)
        parser.add_argument("--num_demo_per_class", type=int, default=5)
        parser.add_argument("--epochs", type=int, default=10, help="training epochs")
        parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate")
        parser.add_argument("--batch_size", type=int, default=1, help="batch size")
        parser.add_argument("--optimizer", type=str, default='Adam', help="optimizer")
        parser.add_argument("--model_name", type=str, default='gpt2-xl', help="name of the model")
        parser.add_argument("--warm_up_steps", type=int, default=0, help="number of warm up steps")
        parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus")
        parser.add_argument("--random_seed", type=int, default=0, help="random_seed")
        parser.add_argument("--project_name", type=str, default='test', help="name of the project")
        parser.add_argument("--checkpoint", type=str, default='')
        parser.add_argument("--do_schedule", type=bool, default=False)
        parser.add_argument("--early_stop", type=int, default=15)
        parser.add_argument("--virtual_token", type=int, default=100)
        parser.add_argument("--checkpoint_dir", type=str, default='')
        return parser

def main(args):
    seed_everything(args.random_seed)

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        
    random_string = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(10))

    optimizer = OPTIMIZERS[args.optimizer]

    icldata = ICLDataModule(args.task_name, args.num_test, args.num_demo_per_class, args.batch_size, tokenizer)
    tokenizer = icldata.tokenizer

    training_steps = len(icldata.train_dataloader())*args.epochs

    model = ICLforClassification(args.model_name, args.task_name, args.exp_type, tokenizer, random_string, optimizer, args.learning_rate, args.warm_up_steps, training_steps, args.do_schedule, args.virtual_token)

    wandb_logger = WandbLogger(project= args.project_name)
    early_stop_callback = EarlyStopping(monitor=args.task_name+"_val_accuracy", patience=args.early_stop, mode="max")
    checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir,filename=args.project_name + random_string, monitor=args.task_name+"_val_accuracy",mode="max",save_top_k=1)

    if args.num_gpu == 1:
        trainer = Trainer(callbacks=[checkpoint_callback, early_stop_callback],max_epochs=args.epochs, accelerator='gpu', devices=[0],logger=wandb_logger)
    else:
        trainer = Trainer(callbacks=[checkpoint_callback, early_stop_callback],max_epochs=args.epochs, strategy='ddp', accelerator="gpu", devices=-1,logger=wandb_logger)
            
    if args.checkpoint:
        if args.mode == 'do_train':
            trainer.fit(model, train_dataloaders=icldata.train_dataloader(), val_dataloaders=icldata.val_dataloader(), ckpt_path=args.checkpoint)
        elif args.mode == 'do_test':
            trainer.test(model, icldata.test_dataloader(), ckpt_path=args.checkpoint)
    else:
        if args.mode == 'do_train':
            trainer.fit(model, train_dataloaders=icldata.train_dataloader(), val_dataloaders=icldata.val_dataloader())
            trainer.test(dataloaders=icldata.test_dataloader(),ckpt_path='best')
        elif args.mode == 'do_test':
            trainer.test(model, icldata.test_dataloader())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = ICLforClassification.add_model_specific_args(parser)

    args = parser.parse_args()
   
    main(args)