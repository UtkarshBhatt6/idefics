import matplotlib.pyplot as plt 
import pandas as pd
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import IdeficsForVisionText2Text, AutoProcessor,AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig

# eval_json=pd.read_json("../../ChartQADataset/val/val_augmented.json")
# print(eval_json)
def check_inference(model, processor, prompts, max_new_tokens=50):
    tokenizer = processor.tokenizer
    bad_words = ["<image>", "<fake_token_around_image>"]
    if len(bad_words) > 0:
        bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids

    eos_token = "</s>"
    eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)

    inputs = processor(prompts, return_tensors="pt").to(device)
    generated_ids = model.generate(**inputs, eos_token_id=[eos_token_id], bad_words_ids=bad_words_ids, max_new_tokens=max_new_tokens, early_stopping=True)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)

def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite

def ds_transforms(example_batch,path):
    image_size = processor.image_processor.image_size
    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    image_transform = transforms.Compose([
        convert_to_rgb,
        transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std),
    ])

    prompts = []
    for i in range(len(example_batch['query'])):
        # We split the captions to avoid having very long examples, which would require more GPU ram during training
        caption = example_batch['query'][i].split(".")[0]
        img_name=example_batch['imgname'][i]
        # image_url="train/png/"+img_name
        image_url=path+img_name
        image = Image.open(image_url)
        curr_prompt= [
                #
                image,
                f"Question: {caption} Answer: Answer is {example_batch['label'][i]}.",
            ]
        print(f"currprompt is {i}: {curr_prompt}")
        prompts.append(

           curr_prompt
        )


    inputs = processor(prompts, transform=image_transform, return_tensors="pt").to(device)
    # print("printing input_ids: ",inputs["input_ids"])
    inputs["labels"] = inputs["input_ids"]
    print("inputs: ",inputs)

    return inputs
# def ds_transforms(example_batch,path):
#     image_size = processor.image_processor.image_size
#     image_mean = processor.image_processor.image_mean
#     image_std = processor.image_processor.image_std

#     image_transform = transforms.Compose([
#         convert_to_rgb,
#         transforms.RandomResizedCrop((image_size, image_size), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=image_mean, std=image_std),
#     ])

#     prompts = []
#     for i in range(len(example_batch['sentence'])):
#         # We split the captions to avoid having very long examples, which would require more GPU ram during training
#         caption = example_batch['sentence'][i].split(".")[0]
#         img_name=example_batch['image'][i]
#         # image_url="train/png/"+img_name
#         image_url=path+img_name
#         image = Image.open(image_url)
#         curr_prompt= [
#                 #
#                 image,
#                 f"Question: {caption} Answer: Answer is {example_batch['text_label'][i]}.",
#             ]
#         print(f"currprompt is {i}: {curr_prompt}")
#         prompts.append(

#            curr_prompt
#         )


#     inputs = processor(prompts, transform=image_transform, return_tensors="pt").to(device)
#     # print("printing input_ids: ",inputs["input_ids"])
#     inputs["labels"] = inputs["input_ids"]
#     print("inputs: ",inputs)

#     return inputs

class ChartQADataset(Dataset):
    """ChartQA Dataset."""

    def __init__(self, json_file, root_dir='./', transform=None):
        """
        Arguments:
            json_file (string): Path to the csv file with annotations.

        """
        self.tacos_df = pd.read_json(json_file)
        self.root_dir = root_dir
        self.transform = transform
        self.image_name=self.tacos_df['imgname']
        self.sentences = self.tacos_df['query']
        #self.labels = self.tacos_df['Sentiment']
        self.text_labels = self.tacos_df['label']
        #self.abstracts = self.tacos_df['Aspect']

    def __len__(self):
        return len(self.tacos_df)

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # image=torch.tensor(self.image_name[idx])
        # sentence=torch.tensor(self.sentences[idx]).squeeze()
        # text_label=torch.tensor(self.text_labels[idx]).squeeze()
        image=self.image_name.iloc[idx]
        sentence = self.sentences.iloc[idx]
        text_label = self.text_labels.iloc[idx]
        #label = self.labels[idx]
        #print('text label',text_label)
        #abstract = self.abstracts[idx]

        sample = {'image':image, 'sentence': sentence,  'text_label': text_label}
        return sample

    def __repr__(self):
        return f"ChartQADataset(num_samples={len(self)})"
    

# tacos_dataset_train = ChartQADataset(json_file='../../ChartQADataset/train/train_augmented.json')
# tacos_dataset_val = ChartQADataset(json_file='../../ChartQADataset/val/val_augmented.json')
tacos_dataset_train = ChartQADataset(json_file='../../ChartQADataset/train/train_augmented_few.json')
tacos_dataset_val = ChartQADataset(json_file='../../ChartQADataset/val/val_augmented_few.json')

dataset_train = Dataset.from_dict(
        {"image":list(tacos_dataset_train.image_name),"sentence": list(tacos_dataset_train.sentences), "text_label": list(tacos_dataset_train.text_labels)})
dataset_val = Dataset.from_dict(
        {"image":list(tacos_dataset_val.image_name),"sentence": list(tacos_dataset_val.sentences), "text_label": list(tacos_dataset_val.text_labels)})



# print('training completed')
device = "cuda" if torch.cuda.is_available() else "cpu"

# checkpoint = "HuggingFaceM4/tiny-random-idefics"
checkpoint = "HuggingFaceM4/idefics-9b"

# # Here we skip some special modules that can't be quantized properly
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_skip_modules=["lm_head", "embed_tokens"],
)

processor = AutoProcessor.from_pretrained(checkpoint, use_auth_token=False)
# processor = AutoTokenizer.from_pretrained(checkpoint,use_fast=False, use_auth_token=False)
# Simply take-off the quantization_config arg if you want to load the original model
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, quantization_config=bnb_config, device_map="auto",cache_dir = '/NS/ssdecl/work/')
print(model)

model_name = checkpoint.split("/")[1]
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)
dataset_train=pd.read_json('../../ChartQADataset/train/train_augmented_few.json')
dataset_val=pd.read_json('../../ChartQADataset/val/val_augmented_few.json')
train_ds =ds_transforms(dataset_train,'../../ChartQADataset/train/png_few/')
eval_ds =ds_transforms(dataset_val,'../../ChartQADataset/val/png_few/')
# train_ds =ds_transforms(dataset_train,'../../ChartQADataset/train/png/')
# eval_ds =ds_transforms(dataset_val,'../../ChartQADataset/val/png/')
model = get_peft_model(model, config)
model.print_trainable_parameters()
training_args = TrainingArguments(
    output_dir=f"{model_name}-output",
    learning_rate=2e-4,
    fp16=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    dataloader_pin_memory=False,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=40,
    eval_steps=20,
    logging_steps=20,
    max_steps=40,
    remove_unused_columns=False,
    push_to_hub=False,
    label_names=["labels"],
    load_best_model_at_end=True,
    report_to=None,
    optim="paged_adamw_8bit",
)
print("no error in training_args ")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
)                       
print("no error in trainer")
trainer.train()
print("no error in trainer.train()")
url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
prompts = [
    # "Instruction: provide an answer to the question. Use the image to answer.\n", 
    url,
    "Question: What's on the picture? Answer:",
]
check_inference(model, processor, prompts, max_new_tokens=5)
image = Image.open('../../ChartQADataset/train/png/two_col_81284.png')
print(image)
plt.imshow(image)
prompts = [
    # "Instruction: provide an answer to the question. Use the image to answer.\n",
  image,
    "Question: What is this image about ? Answer:",
]
check_inference(model, processor, prompts, max_new_tokens=5)    
