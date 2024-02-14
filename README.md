# GNNAVI: Navigating the Information Flow in Large Language Models by Graph Neural Network
This repository contains the code for the paper *GNNAVI: Navigating the Information Flow in Large Language Models by Graph Neural Network*.

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

Please replace [exp] with following methods:

- gnn : GNNAVI
-prefix : Prefix tuning
-adapter : Adapter
-lora : LoRA 
-fpft: Full Parameter Fine-tuning

```
## Model
In the code, the default method of GNNAVI is GNNAVI-GCN. You can use other GNN architechtures by changing the GNN layer in `./models/gpt2_gnn.py` or `./models/llama_gnn.py`. 
