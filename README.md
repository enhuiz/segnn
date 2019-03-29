# SegNN

This is the code for comp5421 program assignment 1.

## Install Dependencies

```bash
$ pip3 install -r requirements.txt
```

## Run Example

Run the following commands under the project directory.

```bash
$ python3 scripts/nn_make.py
$ ./run_zero_nn.sh
```

## Create Your Own Model

Steps to add a model (let's call it MyNN):

1. Create you mynn.py under `segnn/models/mynn.py` and finish design it.
2. Change `scripts/nn_make.py`, create you way to initialize the model.
3. Create `run_my_nn.sh`.

Then just run the previous steps, you result will be automatically output to the exp/ and stdout.

## Directories

```bash
.
├── data
│   └── comp5421_TASK2          <- data folder
├── exp                         <- experiment result, output of the scripts
│   └── zero_nn
│       ├── test                <- predicted labels for test
│       ├── train               
│       │   ├── ckpt            <- pytorch .pth checkpoints, will be used to predict test/val
│       │   └── config.json     <- backup training configuration in json format
│       └── val                 <- predicted labels for val
├── requirements.txt
├── run_zero_nn.sh              <- one sample model, an example
├── scripts
│   ├── nn_forward.py           <- create labels
│   ├── nn_make.py              <- make proto .pth model to zoo/, so that nn_train.py can take it and train
│   ├── nn_train.py             <- train .pth
│   ├── parse_options.sh        <- shell utils to parse options, like argparse in python
│   └── run.sh                  <- run train, feedforward and evaluation
├── segnn
│   ├── data.py                 <- data preprocessing
│   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   │   └── zero_nn.py          <- an example model, you can design your model like this
│   └── utils.py                <- where you add your utils
└── zoo                         <- .pth created by nn_make.py
    └── zero_nn.pth             
```
