from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re
import requests
from io import BytesIO
import pandas as pd
# load vision encoder-decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "ahmed-masry/unichart-base-960"
vision_model = VisionEncoderDecoderModel.from_pretrained(model_name,cache_dir='/NS/ssdecl/work').to(device)
processor = DonutProcessor.from_pretrained(model_name,cache_dir='/NS/ssdecl/work')

# load Reasoning LLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl",cache_dir='/NS/ssdecl/work')
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl",cache_dir='/NS/ssdecl/work', device_map="auto")

def unichart_output(image_url,input_prompt):
    
    #response = requests.get(image_url)
    image = Image.open(image_url).convert("RGB")
    decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values
    
    outputs = vision_model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=vision_model.decoder.config.max_position_embeddings,
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

#def build_prompt(data, summary, question):
#    inputs = '<data>'+','.join(data) + '\n\n' + '<summary> ' + summary+ '\n\n<question> ' + question
  #  ins = '''
 #   Given the following triplet data (marked by <data>) with the summary (marked by <summary>) and the question related to the data (marked by <question>), give the answer with no output of hints, explanations or notes.If the answer is numerical numbers like 3,720 million units,then you should output only 3720,removing units and , seperator. '''
#    return ins.strip() + '\n\n' + inputs
def build_prompt(data, summary, question):
    inputs = '<data>' + ','.join(data) + '\n<summary> ' + summary + '\n\n<question> ' + question
    ins = '''
Given the following triplet data (marked by <data>) with the summary (marked by <summary>) and the question related to the data (marked by <question>), provide only the answer with no explanations, hints, or additional text.

If the answer is numerical, output only the number, removing any units (e.g., million, metric tons, etc.) and comma separators. For example, if the answer is "3,720 million units", output "3720".
If the answer is not numerical (e.g., a name, location, or any other text), return the answer as is, without modification.
    '''
    return ins.strip() + '\n\n' + inputs

def t5_output(final_input):
    input_ids = tokenizer(final_input, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids)
    final_answer = tokenizer.decode(outputs[0])
    final_answer = final_answer.replace('</s>', '').replace('<pad>', '').strip()
    return final_answer

def calc_output(image_url,question):
    input_prompt = "<extract_data_table> <s_answer>"
    data_table = unichart_output(image_url,"<extract_data_table> <s_answer>")
    summary =unichart_output(image_url,"<summarize_chart> <s_answer>")
    data=csv2triples(data_table)
    final_input=build_prompt(data,summary,question)
    answer=t5_output(final_input)
    return answer
def extract_number(s):
    match = re.match(r"[\d\.]+", s)  # Matches digits and decimal points at the start
    return match.group(0) if match else s  # Return number or original string if no match
def clean_string(s):
    return s.split()[0] 
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
        model_answer=  model_answer.replace('%','').replace('$','').replace(',','' )
        #model_answer=clean_string(model_answer)
        if model_answer.endswith('.0'):
            model_answer = model_answer[:-2]  # Remove '.0'
        elif model_answer.endswith('.'):
            model_answer = model_answer[:-1]  # Remove trailing '.'

        print(f'{idx}, model_answer: {model_answer}, actual_answer: {label}',end=' ')
        if model_answer == label:
            correct += 1
        print(f'correct cnt: {correct}')    
    print(f'accuracy: {correct}')       
path='../../ChartQADataset/test/test_augmented.json'
compute_accuracy(path)
    
