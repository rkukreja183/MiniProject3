
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
model_path="/user/r.kukreja/u27878/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"
INPUT_CSV = "directional_ablation_results.csv"         
OUTPUT_CSV = "directional_ablation_judge_results.csv" 

BATCH_SIZE = 8    
MAX_LENGTH = 2048  
MAX_NEW_TOKENS = 256

SAVE_EVERY_BATCHES = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

system_prompt = """
You are an impartial evaluator. You will be given:

1. A user task/prompt
2. A model-generated response to that prompt

Your job is to evaluate the response on two independent dimensions:

A. Happiness / sadness score
Rate the emotional tone of the response from 1 to 5:
1 = very sad / strongly negative emotional tone
2 = somewhat sad / mildly negative emotional tone
3 = neutral / no clear happy or sad tone
4 = somewhat happy / mildly positive emotional tone
5 = very happy / strongly positive emotional tone

Important:
- This score should measure only the emotional tone of the response.
- Do NOT reduce the happiness score because the response is incoherent.
- A nonsensical response can still be happy, sad, or neutral.
- Focus on affective markers such as joy, sadness, disappointment, optimism, grief, excitement, etc.
- If the response has mixed emotions, choose the dominant emotional tone.
- If there is no clear emotional tone, choose 3.

B. Coherence / task-answering score
Rate the response from 1 to 5:
1 = completely incoherent, nonsensical, or unrelated to the prompt
2 = mostly incoherent or mostly fails to answer the prompt
3 = partially coherent and partially answers the prompt
4 = mostly coherent and mostly answers the prompt
5 = fully coherent, clear, and directly answers the prompt

Important:
- This score should measure only whether the response makes sense and answers the given prompt.
- Do NOT change the coherence score because the response is happy, sad, or neutral.
- A very sad response can be perfectly coherent.
- A very happy response can be completely incoherent.

Return your answer only as valid JSON with the following fields:

{
  "happiness_score": <integer from 1 to 5>,
  "coherence_score": <integer from 1 to 5>,
  "happiness_explanation": "<brief explanation>",
  "coherence_explanation": "<brief explanation>"
}
"""

def make_prompts_for_judge(prompt, response):
    return f"""User Task:
{prompt}

Model Response:
{response}

Evaluate the model response using the rubric. Return only valid JSON.
"""


tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, 
    local_files_only=True
)
model = model.to(device)
model.eval()


def get_judge_response(
    model,
    tokenizer,
    prompts,
    # max_length=2048,
    max_new_tokens=256,
):
    if isinstance(prompts, str):
        prompts = [prompts]

    texts = []

    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        texts.append(text)

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        # max_length=max_length
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,             
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = []

    input_len = inputs["input_ids"].shape[1]

    for i in range(len(prompts)):
        generated_ids = output_ids[i, input_len:]

        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        ).strip()

        responses.append(response)
        print(response)
    return responses


def parse_judge_json(raw_text):

    result = {
        "happiness_score": None,
        "coherence_score": None,
        "happiness_explanation": None,
        "coherence_explanation": None,
    }

    if raw_text is None:
        return result

    text = raw_text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return result

        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return result

    hs = data.get("happiness_score")
    cs = data.get("coherence_score")

    try:
        hs = int(hs)
        if hs < 1 or hs > 5:
            hs = None
    except Exception:
        hs = None

    try:
        cs = int(cs)
        if cs < 1 or cs > 5:
            cs = None
    except Exception:
        cs = None

    result["happiness_score"] = hs
    result["coherence_score"] = cs
    result["happiness_explanation"] = data.get("happiness_explanation")
    result["coherence_explanation"] = data.get("coherence_explanation")

    return result

df = pd.read_csv(INPUT_CSV)

df["layer"] = df["layer"].fillna(-1).astype(int)

for col in [
    "judge_raw",
    "happiness_score",
    "coherence_score",
    "happiness_explanation",
    "coherence_explanation",
]:
    if col not in df.columns:
        df[col] = None

rows_to_judge = df[df["happiness_score"].isna() | df["coherence_score"].isna()].index.tolist()

print(f"Total rows: {len(df)}")
print(f"Rows to judge: {len(rows_to_judge)}")

num_batches = (len(rows_to_judge) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_num in tqdm(range(num_batches), desc="Judging batches"):
    start = batch_num * BATCH_SIZE
    end = start + BATCH_SIZE
    batch_indices = rows_to_judge[start:end]

    batch_prompts = []

    for idx in batch_indices:
        row = df.loc[idx]

        judge_prompt = make_prompts_for_judge(
            prompt=row["prompt"],
            response=row["response"]
        )

        batch_prompts.append(judge_prompt)
        #print(batch_prompts)

    try:
        raw_outputs = get_judge_response(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            # max_length=MAX_LENGTH,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    except RuntimeError as e:
        print(f"\nRuntime error on batch {batch_num}: {e}")
        raise e

    for idx, raw in zip(batch_indices, raw_outputs):
        parsed = parse_judge_json(raw)

        df.at[idx, "judge_raw"] = raw
        df.at[idx, "happiness_score"] = parsed["happiness_score"]
        df.at[idx, "coherence_score"] = parsed["coherence_score"]
        df.at[idx, "happiness_explanation"] = parsed["happiness_explanation"]
        df.at[idx, "coherence_explanation"] = parsed["coherence_explanation"]

    if (batch_num + 1) % SAVE_EVERY_BATCHES == 0:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved progress after batch {batch_num + 1}/{num_batches}")
df.to_csv(OUTPUT_CSV, index=False)

print(f"Done. Saved judged dataframe to: {OUTPUT_CSV}")