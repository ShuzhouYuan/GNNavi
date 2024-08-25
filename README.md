# GNNavi: Navigating the Information Flow in Large Language Models by Graph Neural Network
This repository contains the code for the paper [*GNNavi: Navigating the Information Flow in Large Language Models by Graph Neural Network*](https://arxiv.org/pdf/2402.11709).

![image](./image/model_overview.png)

## Environments and Dependencies
- python==3.9.16
- pytorch==2.0.1
- pytorch-lightning==2.0.2
- torch-geometric==2.3.0
- transformers==4.37.2 
- peft==0.5.0
- adapters==0.1.1

Install all the required dependencies:
```
pip install -r requirements.txt
```

## Training
Training with the default setting:
```
./train.sh [model] [method]
```
Please replace [model] with **gpt2** or **llama**.

Please replace [method] with following methods:

- gnn : GNNavi
- lora : LoRA 
- prefix : Prefix tuning
- adapter : Adapter
- fpft: Full Parameter Fine-tuning


## GNNAVI
In the code, the default setting for GNNavi is GNNavi-GCN. You can use other GNN architechtures by changing the GNN layer in `./src/models/gpt2_gnn.py` or `./src/models/llama_gnn.py`. 


## How to cite

Please cite our work as follows:

```
@inproceedings{yuan-etal-2024-gnnavi,
    title = "{GNN}avi: Navigating the Information Flow in Large Language Models by Graph Neural Network",
    author = {Yuan, Shuzhou  and
      Nie, Ercong  and
      F{\"a}rber, Michael  and
      Schmid, Helmut  and
      Schuetze, Hinrich},
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.237",
    pages = "3987--4001",
    abstract = "Large Language Models (LLMs) exhibit strong In-Context Learning (ICL) capabilities when prompts with demonstrations are used. However, fine-tuning still remains crucial to further enhance their adaptability. Prompt-based fine-tuning proves to be an effective fine-tuning method in low-data scenarios, but high demands on computing resources limit its practicality. We address this issue by introducing a prompt-based parameter-efficient fine-tuning (PEFT) approach. GNNavi leverages insights into ICL{'}s information flow dynamics, which indicates that label words act in prompts as anchors for information propagation. GNNavi employs a Graph Neural Network (GNN) layer to precisely guide the aggregation and distribution of information flow during the processing of prompts by hardwiring the desired information flow into the GNN. Our experiments on text classification tasks with GPT-2 and Llama2 show GNNavi surpasses standard prompt-based fine-tuning methods in few-shot settings by updating just 0.2{\%} to 0.5{\%} of parameters. We compare GNNavi with prevalent PEFT approaches, such as prefix tuning, LoRA and Adapter in terms of performance and efficiency. Our analysis reveals that GNNavi enhances information flow and ensures a clear aggregation process.",
}
```

