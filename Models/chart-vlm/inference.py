from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch, os, re
import requests
from io import BytesIO

# load vision encoder-decoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "ahmed-masry/unichart-base-960"
vision_model = VisionEncoderDecoderModel.from_pretrained(model_name,cache_dir='/NS/ssdecl/work').to(device)
processor = DonutProcessor.from_pretrained(model_name,cache_dir='/NS/ssdecl/work')

# load Reasoning LLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl",cache_dir='/NS/ssdecl/work')
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl",cache_dir='/NS/ssdecl/work', device_map="auto")

def unichart_output(batch_image_urls,input_prompt):
    
    batch_pixel_values = []
    for image_url in batch_image_urls:
        image = Image.open(image_url).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        batch_pixel_values.append(pixel_values)

    batch_pixel_values = torch.cat(batch_pixel_values).to(device)
    decoder_input_ids = processor.tokenizer(input_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    decoder_input_ids = decoder_input_ids.to(device)

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

    batch_sequences = processor.batch_decode(outputs.sequences)
    batch_sequences = [seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "").split("<s_answer>")[1].strip() for seq in batch_sequences]
    return batch_sequences

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

def build_prompt(data, summary, question):
    inputs = '<data>'+','.join(data) + '\n\n' + '<summary> ' + summary+ '\n\n<question> ' + question
    ins = '''
    Given the following triplet data (marked by <data>) with the summary (marked by <summary>) and the question related to the data (marked by <question>), give the answer with no output of hints, explanations or notes.
    '''
    return ins.strip() + '\n\n' + inputs

def t5_output(batch_final_inputs):
    input_ids = tokenizer(batch_final_inputs, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
    outputs = model.generate(input_ids)
    batch_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    batch_answers = [answer.strip() for answer in batch_answers]
    return batch_answers

def calc_output_batch(batch_image_urls, batch_questions):
    batch_data_tables = []
    batch_summaries = []

    batch_data_tables = unichart_output(batch_image_urls,"<extract_data_table> <s_answer>")
    batch_summaries = unichart_output(batch_image_urls, "<summarize_chart> <s_answer>")
    
    batch_final_inputs = []
    for data_table, summary, question in zip(batch_data_tables, batch_summaries, batch_questions):
        data = csv2triples(data_table)
        final_input = build_prompt(data, summary, question)
        batch_final_inputs.append(final_input)
    
    batch_answers = t5_output(batch_final_inputs)
    return batch_answers

def compute_accuracy(data_path, batch_size=5):
    df = pd.read_json(data_path)
    correct = 0
    total_rows = len(df)
    
    for i in range(0, total_rows, batch_size):
        batch_df = df[i:i + batch_size]
        
        # Collect batch inputs
        batch_image_urls = []
        batch_questions = []
        batch_labels = []
        
        for _, row in batch_df.iterrows():
            imgname = row['imgname']
            question = row['query']
            label = row['label']
            image_url = f'{data_path}/{imgname}'
            batch_image_urls.append(image_url)
            batch_questions.append(question)
            batch_labels.append(label)
        
        batch_model_answers = calc_output(batch_image_urls, batch_questions)
        
        for model_answer, label in zip(batch_model_answers, batch_labels):
            print(f'model_answer: {model_answer}, actual_answer: {label}')
            if model_answer == label:
                correct += 1
    
    print(f'Correct answers: {correct}')

path='../../ChartQADataset/test/test_augmented.json'
compute_accuracy(path)

    