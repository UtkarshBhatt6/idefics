from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re
import requests
from io import BytesIO

# load vision encoder-decoder
model_name = "ahmed-masry/unichart-base-960"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = DonutProcessor.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# load Reasoning LLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl", device_map="auto")



def vision_model_output(image_url,input_prompt):
    
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=4,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    
    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = sequence.split("<s_answer>")[1].strip()
    return sequence

def csv2triples(csv, separator='|', delimiter='&'):  
    lines = csv.strip().split(delimiter)
    header = lines[0].split(separator)
    triples = []
    for line in lines[1:]:
        if not line:
            continue
        values = line.split(separator)
        entity = values[0]
        for i in range(1, len(values)):
            if i >= len(header):
                break
            #triples.append((entity, header[i], values[i]))
            #---------------------------------------------------------
            temp = [entity, header[i]]  
            triples.append('(' + temp[0].strip() + ',' + temp[1].strip() + ',' + values[i].strip() + ')')
            #---------------------------------------------------------
    return triples

# data = "Characteristic | Installed base in million units & 2017 | 105 & 2013 | 128 & 2010 | 109 & 2008 | 63 & 2006 | 64 & 1999 | 54 & 1995 | 64"
# print(csv2triples(data))

def build_prompt(data, summary, question):
    inputs = '<data>'+','.join(data) + '\n\n' + '<summary> ' + summary+ '\n\n<question> ' + question
    ins = '''
    Given the following triplet data (marked by <data>) with the summary (marked by <summary>) and the question related to the data (marked by <question>), give the answer with no output of hints, explanations or notes.
    '''
    return ins.strip() + '\n\n' + inputs

def vision_model_output(image_url,input_prompt):
    input_ids = tokenizer(final_input, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids)
    final_answer = tokenizer.decode(outputs[0])
    final_answer = final_answer.replace('</s>', '').replace('<pad>', '').strip()
    print(final_answer)

# generate data table
input_prompt = "<extract_data_table> <s_answer>"
image_url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png"
data_table = model_output(image_url,input_prompt)
print(data_table)

# generate summary

input_prompt = "<summarize_chart> <s_answer>"
# image_url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png"
summary = model_output(image_url,input_prompt)
print(summary)


# convert to triples

data=csv2triples(data_table)
print(data)

# get final_input
final_input=build_prompt(data,summary,'''facebook percentage of users in 60+ age group ?''')
print(final_input)

# get final answer

input_ids = tokenizer(final_input, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids)
final_answer = tokenizer.decode(outputs[0])
final_answer = final_answer.replace('</s>', '').replace('<pad>', '').strip()
print(final_answer)
