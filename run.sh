#!/bin/bash

wget -P ./data https://raw.githubusercontent.com/neychev/made_nlp_course/master/datasets/Machine_translation_EN_RU/data.txt

poetry install

poetry run python run_sentencepiece.py
poetry run python train_transformer_nn.py
