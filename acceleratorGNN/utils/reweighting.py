import warnings
from functools import wraps, partial
import torch
from torch import nn
from transformers import PreTrainedModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.optim import Adam
import torch.nn.functional as F
import evaluate

label_dict = {'sst2': {0: ' Negative', 1: ' Positive'},
    'agnews': {0: ' World', 1: ' Sports', 2: ' Business', 3: ' Technology'},
    'trec': {0: ' Abbreviation', 1: ' Entity', 2: ' Description', 3: ' Person',
                          4: ' Location',
                          5: ' Number'},
    'emo': {0: ' Others', 1: ' Happy', 2: ' Sad', 3: ' Angry'},
}

class Predictor:
    def __init__(self, label_id_dict, pad_token_id, task_name, tokenizer, layer,
                 naive_class_embs=None,
                 naive_final_emb=None) -> None:
        self.naive_class_embs = naive_class_embs
        self.naive_final_emb = naive_final_emb
        self.label_id_dict = label_id_dict
        self.pad_token_id = pad_token_id
        self.task_name = task_name
        self.tokenizer = tokenizer
        self.layer = layer

        if task_name == 'sst2':
            self.prefix_idxs = [tokenizer.encode('Sentiment', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        elif task_name == 'agnews':
            self.prefix_idxs = [tokenizer.encode('Answer', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        elif task_name == 'trec':
            self.prefix_idxs = [tokenizer.encode(' Type', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        elif task_name == 'emo':
            self.prefix_idxs = [tokenizer.encode('Emotion', add_special_tokens=False)[-1],
                                tokenizer.encode(':', add_special_tokens=False)[0]]
        else:
            raise NotImplementedError(f"task_name: {task_name}")

    def get_pos(self, inputs):
        label_id_dict = self.label_id_dict
        pad_token_id = self.pad_token_id
        #final_pos = (inputs['input_ids'] != pad_token_id).int().sum(-1) - 1
        final_pos = len(inputs['input_ids'])-1
        device = inputs['input_ids'].device
        bsz, sql = inputs['input_ids'].shape
        class_poss = []
        for idx in label_id_dict.values():
            class_idx = idx
            for offset, prefix_idx in enumerate(reversed(self.prefix_idxs)):
                class_idx += prefix_idx * 100000 ** (offset + 1)
            input_ids = inputs['input_ids'].detach().clone()
            input_ids[:, 1:] += inputs['input_ids'][:, :-1] * 100000
            input_ids[:, 2:] += inputs['input_ids'][:, :-2] * 100000 * 100000
            class_pos = torch.arange(sql, device=device).unsqueeze(0).repeat(bsz, 1)[
                input_ids == class_idx].squeeze()
            class_poss.append(class_pos)
        return class_poss, final_pos

    def _cal_all_key_and_values_of_class(self, inputs, past_key_values, one_class_one_list=False,
                                         include_final=False):
        class_poss, final_pos = self.get_pos(inputs)

        if include_final:
            class_poss.append(final_pos)

        def get_vecs(ker_or_value, class_poss):
            batch_idx = torch.arange(inputs['input_ids'].shape[0])
            class_vecs = []
            for poss in class_poss:
                class_vec = ker_or_value[batch_idx, :, poss, :]
                class_vecs.append(class_vec.unsqueeze(-2))
            if not one_class_one_list:
                class_vecs = torch.cat(class_vecs, dim=-2)
            return class_vecs

        key_and_values = []
        for layer in range(0, self.layer):
            key_and_values.append(tuple([get_vecs(_, class_poss) for _ in past_key_values[layer]]))
        return key_and_values  # tuple of tuple of tensor (bsz, n_head, num_class, d_head)

    def cal_all_key_and_values_of_class(self, inputs, results, one_class_one_list=False,
                                        include_final=False):
        past_key_values = results.past_key_values
        key_and_values = self._cal_all_key_and_values_of_class(inputs, past_key_values,
                                                               one_class_one_list=one_class_one_list,
                                                               include_final=include_final)
        return key_and_values  # tuple of tuple of tensor (bsz, n_head, num_class, d_head)

    def get_attention(self, inputs, results, layer):
        class_poss, final_pos = self.get_pos(inputs)
        batch_idx = torch.arange(inputs['input_ids'].shape[0])
        scores = []
        for class_pos in class_poss:
            attention = results.attentions[layer][batch_idx, :, final_pos, class_pos]
            score = attention
            if class_pos.numel() == 1:
                score = score.sum(-1)
            else:
                score = score.sum()
            if inputs['input_ids'].shape[0] != 1:
                warnings.warn(f'Only support batch_size=1 now!')
            scores.append(score.unsqueeze(0))
        scores = torch.cat(scores, dim=0)
        return scores

    def cal_all_sim_attn(self, inputs, results):
        sims = []
        for layer in range(0, self.layer):
            sim = self.get_attention(inputs=inputs, results=results, layer=layer)
            sims.append(sim.unsqueeze(1))
        sims = torch.cat(sims, dim=1)
        sims = sims.reshape(inputs['input_ids'].shape[0], -1)
        return sims


class AttentionAdapterBase(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.use_flag = True

    def forward(self, attn_weights):
        if self.use_flag:
            return self._forward(attn_weights)
        else:
            return attn_weights

    def _forward(self, attn_weights):
        raise NotImplementedError

    def register_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


def gpt2_attn(self, query, key, value, attention_mask=None, head_mask=None, attention_adapter=None):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))

    if self.scale_attn_weights:
        attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

    if self.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(self.layer_idx + 1)

    if not self.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
        attn_weights = torch.where(causal_mask, attn_weights,
                                   self.masked_bias.to(attn_weights.dtype))

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.Softmax(dim=-1)(attn_weights)

    if attention_adapter is not None:
        attn_weights = attention_adapter(attn_weights)

    attn_weights = attn_weights.type(value.dtype)
    attn_weights = self.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, value)

    return attn_output, attn_weights

class AttentionerManagerBase:
    def __init__(self, model: PreTrainedModel, predictor: Predictor, n_demo, device,n_head):
        self.n_demo = n_demo
        self.n_head = n_head
        self.device = device
        self.model = model
        self.attention_adapters = self.register_attentioner_to_model()
        self.model.forward = manager_decoractor(self)(self.model.forward)
        self.predictor = predictor

    @property
    def input_ids(self):
        return self._input_ids

    @input_ids.setter
    def input_ids(self, input_ids):
        self._input_ids = input_ids
        class_poss, final_poss = self.predictor.get_pos({'input_ids': input_ids})
        for attention_adapter in self.attention_adapters:
            attention_adapter.register_input_ids(input_ids)
            attention_adapter.class_poss = class_poss
            attention_adapter.final_poss = final_poss

    def register_input_ids(self, input_ids):
        self.input_ids = input_ids

    def register_attentioner_to_model(self):
        raise NotImplementedError

    def zero_grad(self, set_to_none=True):
        if set_to_none:
            for attention_adapter in self.attention_adapters:
                attention_adapter.params = None
        else:
            for attention_adapter in self.attention_adapters:
                attention_adapter.zero_grad(set_to_none=True)

    def grad_process(self, grad, use_abs=True):
        assert len(grad.shape) == 4
        grad = grad.sum(1)
        if use_abs:
            grad = abs(grad)
        return grad

    def grad(self, *args, **kwargs):
        grads = []
        for attention_adapter in self.attention_adapters:
            grads.append(self.grad_process(attention_adapter.params.grad, *args, **kwargs))
        return grads

    def params(self):
        params = []
        for attention_adapter in self.attention_adapters:
            params.append(attention_adapter.weight)
        return params


def manager_decoractor(manager: AttentionerManagerBase):
    def model_forward_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            input_ids = kwargs.get('input_ids', None)
            if input_ids is None:
                input_ids = args[0]
            manager.register_input_ids(input_ids)
            return fn(*args, **kwargs)

        return wrapper

    return model_forward_decorator


class GPT2AttentionerManager(AttentionerManagerBase):
    def __init__(self, model: PreTrainedModel, n_demo, predictor: Predictor, device, n_head=1):
        super().__init__(model, predictor, n_demo, device,n_head=n_head)

    def register_attentioner_to_model(self):
        attention_adapters = []
        for i, layer in enumerate(self.model.transformer.h):
            attention_adapter = AttentionAdapter(n_demo=self.n_demo, device=self.device,
                                                 n_head=self.n_head)
            layer.attn._attn = partial(gpt2_attn, layer.attn,
                                       attention_adapter=attention_adapter)
            attention_adapters.append(attention_adapter)
        return attention_adapters


class AttentionAdapter(AttentionAdapterBase):
    def __init__(self, n_demo, n_head, device) -> None:
        super().__init__()
        self.n_demo = n_demo
        self.n_head = n_head
        self.weight = torch.nn.Parameter(
            torch.zeros((n_head, n_demo), requires_grad=True, device=device))
        self.class_poss = None
        self.final_poss = None

    def _forward(self, attn_weights):
        class_poss = self.class_poss
        final_poss = self.final_poss
        weight = self.weight.exp()
        bsz, n_head, seq_len, _ = attn_weights.shape
        assert bsz == 1
        mask_mat = torch.ones((1, n_head, seq_len, seq_len), device=attn_weights.device)
        mask_mat[:, :, final_poss, class_poss] = weight.reshape(1, self.n_head, self.n_demo)
        return attn_weights * mask_mat

    @property
    def grad(self):
        return self.weight.grad

    def zero_grad(self, set_to_none: bool = False) -> None:
        if self.weight.grad is not None:
            if set_to_none:
                self.weight.grad = None
            else:
                self.weight.grad = torch.zeros_like(self.weight.grad)

def get_label_id_dict(label_dict, tokenizer):
    label_id_dict = {k: tokenizer.encode(v, add_special_tokens=False)[0] for k, v in
                          label_dict.items()}
    # for v in label_dict.values():
    #     token_num = len(tokenizer.encode(v, add_special_tokens=False))
    #     if token_num != 1:
    #         warnings.warn(f"{v} in {args.task_name} has token_num: {token_num} which is not 1")
    return label_id_dict

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/hk-project-gnn4nlp/nu4126/acceleratorGNN')
    from utils.preproc import ICLDataModule
    from pytorch_lightning import seed_everything

    seed_everything(0)

    model_name = 'gpt2-xl'
    task_name = 'sst2'
    lr = 0.01
    num_demo = 200
    num_epoch = 10

    config = GPT2Config.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    icl_data = ICLDataModule(task_name, 1000, num_demo, 1, tokenizer)
    train_loader = icl_data.train_dataloader()
    val_loader = icl_data.val_dataloader()

    label_map = {tokenizer.encode(v, add_special_tokens=False)[0]: k for k, v in
                          label_dict[task_name].items()}
    interest_index = list(label_map.keys())

    model = GPT2LMHeadModel.from_pretrained(model_name, config=config).to('cuda')
    label_id_dict = get_label_id_dict(label_dict[task_name], tokenizer)
    predictor = Predictor(label_id_dict=label_id_dict, pad_token_id=tokenizer.pad_token_id,
                              task_name=task_name, tokenizer=tokenizer, layer=config.n_layer)
    
    attentionermanger = GPT2AttentionerManager(model, 1*len(label_dict[task_name]),
                                                       predictor=predictor,
                                                      device='cuda', n_head = config.n_head)
    attentionermanger.params()
    params = attentionermanger.params()
    optimizer = Adam(params, lr=lr)
    accuracy = evaluate.load("accuracy")
                             
    loss_list = []
    model.train()
    for epoch in range(num_epoch):
        loss_item = 0.
        for idx, batch in enumerate(train_loader):
            input_ids = batch.input_ids.to('cuda')
            attention_mask = batch.attention_mask.to('cuda')
            label = batch.label.to('cuda')

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits[:, -1, interest_index]

            loss = F.cross_entropy(logits, label)
            
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            loss_item += loss.item()
            params[0].retain_grad()
            print(params[0])
        loss_list.append(loss_item / idx)
        average_loss = float(loss_item / idx)
        print(f'{average_loss}/{num_epoch}')
    
        model.eval()
        with torch.no_grad():
            true_labels, predict_labels = [], []
            for idx, batch in enumerate(val_loader):
                input_ids = batch.input_ids.to('cuda')
                attention_mask = batch.attention_mask.to('cuda')
                label = batch.label.to('cuda')
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits[:, -1, interest_index]

                probs = F.softmax(logits,dim=-1)
        
                predictions = probs.argmax().item()

                true_labels.extend(label)
                predict_labels.append(probs)

            accuracy.compute(predictions=predictions, references=label)['accuracy']
            print(accuracy)

