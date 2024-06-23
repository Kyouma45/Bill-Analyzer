import streamlit as st
import re
import torch
from PIL import Image
from peft import PeftModel, PeftConfig
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor
from transformers import PaliGemmaForConditionalGeneration

REPO_ID="google/paligemma-3b-pt-224"
FINETUNED_MODEL_ID = "Kyouma45/fine_tune_paligemma"
PROMPT="Extract JSON"
processor = AutoProcessor.from_pretrained(REPO_ID)

st.title('Bill Analyzer')

if st.button('Load LLM'):
    config = PeftConfig.from_pretrained(FINETUNED_MODEL_ID)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
    )
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(REPO_ID,quantization_config=bnb_config)
    model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_ID)
    #model = PaliGemmaForConditionalGeneration.from_pretrained(FINETUNED_MODEL_ID)
    print('Loading Complete')

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.session_state.uploaded_image = image
else:
    print('Upload Image')


def token2json(tokens, is_inner_value=False, added_vocab=None):
        """
        Convert a (generated) token sequence into an ordered JSON format.
        """
        if added_vocab is None:
            added_vocab = processor.tokenizer.get_added_vocab()

        output = {}

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            key_escaped = re.escape(key)

            end_token = re.search(rf"</s_{key_escaped}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(
                    f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE | re.DOTALL
                )
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = token2json(content, is_inner_value=True, added_vocab=added_vocab)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if leaf in added_vocab and leaf[0] == "<" and leaf[-2:] == "/>":
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + token2json(tokens[6:], is_inner_value=True, added_vocab=added_vocab)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}

if st.button('Process Image'):
    st.session_state.inputs = processor(text=PROMPT, images=st.session_state.uploaded_image, return_tensors="pt")
    st.session_state.generated_ids = model.generate(**st.session_state.inputs, max_new_tokens=512)
    st.session_state.image_token_index = model.config.image_token_index
    st.session_state.num_image_tokens = (st.session_state.
                                         generated_ids == st.session_state.image_token_index).sum().item()
    st.session_state.num_text_tokens = len(processor.tokenizer.encode(PROMPT))
    st.session_state.num_prompt_tokens = st.session_state.num_image_tokens + st.session_state.num_text_tokens + 2
    st.session_state.generated_text = processor.batch_decode(st.session_state.generated_ids[:, st.session_state.num_prompt_tokens:],
                                            skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    st.session_state.generated_json = token2json(st.session_state.generated_text)
    st.write(st.session_state.generated_json)