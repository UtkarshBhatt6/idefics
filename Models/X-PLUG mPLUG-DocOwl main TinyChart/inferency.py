import torch
from PIL import Image
import pandas as pd
from tinychart.model.builder import load_pretrained_model
from tinychart.mm_utils import get_model_name_from_path
from tinychart.eval.run_tiny_chart import inference_model
from tinychart.eval.eval_metric import parse_model_output, evaluate_cmds
def show_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img.show()


# Build the model
model_path = "mPLUG/TinyChart-3B-768"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    device="cuda",

    )
def calc_output(image_path,text):
    
    response = inference_model([image_path], text, model, tokenizer, image_processor, context_len, conv_mode="phi", max_new_tokens=1024)
    return evaluate_cmds(parse_model_output(response))

def compute_accuracy(data_path):
    df=pd.read_json(data_path)
    #rows=len(df)
    total =500
    correct=0
    for idx, row in df.iterrows():
        imgname = row['imgname']
        question = row['query']
        label = row['label']
        image_url = f'../../ChartQADataset/test/png/{imgname}'

        model_answer = calc_output(image_url, question)
        print(f'{idx}, model_answer: {model_answer}, actual_answer: {label}',end=' ')
        if model_answer == label:
            correct += 1
        print(f'correct cnt: {correct}')
    print(f'accuracy: {correct}')

path='../../ChartQADataset/test/test_augmented.json'
compute_accuracy(path)
