import os
import sys
from typing import List, Dict
from tqdm import tqdm
import json
import numpy as np
import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig


sys.path.append("/kaggle/working/InternVL/internvl_chat/")

from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (
    InternVisionConfig,
    InternVisionModel,
    InternVLChatConfig,
    InternVLChatModel,
)

SYSTEM_PROMPT = """You are an advanced Vision–Language Model (VLM).  
Your task is to accurately extract information from a business card image according to the constraints and output it in structured JSON format.

# Constraints
1. Extract the following 15 fields:
   - Company Name
   - Name
   - Email
   - Phone Number
   - Address

2. If any field is not present on the card, output its value as an empty string (`""`).

3. Perform appropriate character normalization (e.g., half-width/full-width, hiragana/katakana) to minimize OCR errors.

4. The output must be valid JSON.
"""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_model_8bit(model_id: str):
    config = InternVLChatConfig.from_pretrained(model_id)
    # config.system_message = SYSTEM_PROMPT
    # 1. BitsAndBytesConfig で 8bit 指定（load_in_8bit と排他）
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,            # アウトライヤー閾値（必要に応じて調整）
        llm_int8_has_fp16_weight=True      # 重みは fp16 に保持
    )
    # 2. モデル読み込み：torch_dtype は float16 を指定
    model = InternVLChatModel.from_pretrained(
        model_id,
        # quantization_config=quant_config,  # ここに設定を渡す
        torch_dtype=torch.bfloat16,         # bfloat16 ではなく float16
        use_flash_attn=False,
        trust_remote_code=True,
        # load_in_8bit=True,
        low_cpu_mem_usage=True,
        config=config
    ).eval().cuda()
    return model

def find_sublist_matches(lista, listb):
    """
    lista の中から listb と完全一致する連続部分を探し、
    (開始インデックス, マッチしたサブリスト) のタプル一覧を返す。
    """
    n, m = len(lista), len(listb)
    matches = []
    # 探索範囲は i = 0 ... n-m
    for i in range(n - m + 1):
        # スライス比較：listb と一致すればマッチ
        if lista[i : i + m] == listb:
            matches.append(i)
    return matches

def extract_probs_for_fields(
    outputs: Dict[str,str],
    probs: List[float],
    decode_tokens: List[str],
    tokenizer: AutoTokenizer,
) -> Dict[str, float]:
    """
    outputs: {"Company": "...", ...}
    probs:   decode_tokens と同長の確率リスト
    decode_tokens: BPE されたトークン列
    tokenizer: 同じトークナイザーインスタンス
    key_order: ["Company", "Name", "Email", "Phone Number", "Address"]
    """
    def tokenize_key(key: str) -> List[str]:
        # key を内部のトークナイザーで BPE に分割し、decode_tokens と同じ表現に落とす
        toks = tokenizer.tokenize(key)
        # print(f"key: {key}, toks: {toks}")
        # transformers の fast tokenizers は '▁' などの前処理記号を含むので
        # decode_tokens に合わせて normalize
        return [t.replace("Ġ", " ") for t in toks]
        # return toks

    def find_subsequence(seq: List[str], subseq: List[str]) -> int:
        """seq の中から subseq が最初に現れる位置を返す。なければ -1"""
        L, M = len(seq), len(subseq)
        for i in range(L - M + 1):
            if seq[i:i+M] == subseq:
                return i
        return -1
    
    key_order = list(outputs.keys())

    field_probs: Dict[str, float] = {}
    n = len(decode_tokens)

    # 1) 各キーをトークン列に分割し、decode_tokens 上での開始位置を取得
    key_positions: Dict[str,int] = {}
    for key in key_order:
        key_toks = tokenize_key(key)
        # print(f"key: {key}, key_toks: {key_toks}")
        pos = find_subsequence(decode_tokens, key_toks)
        key_positions[key] = pos

    # 2) 各キーごとに、その直後（start + len(key_toks)）から
    #    次のキーの start もしくは '"}' の位置までを値領域とみなす
    #    なお区切りトークン '":"’ がキー末尾に含まれる場合は、それも一緒にスキップします
    #    （多くのモデルはキー→'":"’→値・・・→'","' という出力をします）
    #    ここでは簡単化のため「キー位置＋キー長＋1」で値先頭としていますが、
    #    必要なら find_subsequence で '":"’ を確実に検出しても OK。
    closing_pos = find_subsequence(decode_tokens, ['"}'])

    for idx, key in enumerate(key_order):
        start = key_positions.get(key, -1)
        if start < 0:
            outputs[key+"_probability"] = None
            continue
        key_len = len(tokenize_key(key))
        # 値開始位置
        val_start = start + key_len + 1

        # 次キー or closing_pos
        next_keys = [
            key_positions[k]
            for k in key_order[idx+1:]
            if key_positions.get(k, -1) > start
        ]
        candidates = [p for p in next_keys if p >= 0]
        if closing_pos is not None:
            candidates.append(closing_pos)
        if not candidates:
            outputs[key+"_probability"] = None
            continue
        val_end = min(candidates)

        if val_end <= val_start:
            outputs[key+"_probability"] = None
            continue

        segment = probs[val_start:val_end]
        outputs[key+"_probability"] = sum(segment) / len(segment) if segment else None

    return outputs

def test_ocr_images(args):
    # InternVLモデルとプロセッサの初期化
    model = load_model_8bit(args.model_path)
    # print(model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=False
    )

    space = tokenizer('""')
    print(f"space: {space}")

    with open(args.test_path, "r") as f:
        data = f.readlines()
    test_json = [json.loads(line) for line in data]
    print(f"Test data length: {len(test_json)}")
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    result_json = []
    # ディレクトリ内の画像を処理
    for data in tqdm(test_json):
        # 画像を読み込み
        pixel_values = load_image(data["image"], max_num=6).to(torch.bfloat16).cuda()

        # プロンプトの設定
        question = data["conversations"][0]["value"]

        # 推論の実行
        outputs, probs, output_tokens = model.chat(tokenizer, pixel_values, question, generation_config)
        # print(f"outputs: {outputs}")
        # print(f"probs: {probs}")
        # print(f"output_tokens: {output_tokens}")

        decode_tokens = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        # decode_tokens = decode_tokens.split(decode_tokens.sep.strip())[0].strip()
        # print(f"decode_tokens: {decode_tokens}")

        outputs = outputs.replace("```json", "").replace("```", "").replace("'", "\'")
        result_dict = json.loads(outputs)

        result = extract_probs_for_fields(
            result_dict,
            probs,
            decode_tokens,
            tokenizer
        )
        print(f"result: {result}")

        # output = {}
        # ocr_items = ["Company", "Name", "Email", "Phone Number", "Address"]

        # for k, v in result_dict.items():
        #     if k in ocr_items:
        #         output[k] = v
        #     else:
        #         continue

        #     # トークンを取得
        #     key_tokens = tokenizer(k)['input_ids']
        #     item_tokens = tokenizer(v)['input_ids']
        #     print(f"key_tokens: {key_tokens}")
        #     print(f"item_tokens: {item_tokens}")

        #     # key_tokensのトークンがoutput_tokensのどこにあるかを探す
        #     start_index = find_sublist_matches(output_tokens, key_tokens)[0]
        #     # item_tokensのトークンがoutput_tokensのどこにあるかを探す
        #     item_start_index = find_sublist_matches(output_tokens, item_tokens)
        #     print(f"item_start_index: {item_start_index}")
        #     # start_indexよりも後で、item_start_indexが最初に出現するインデックスを取得
        #     item_start_index = [i for i in item_start_index if i > start_index][0]
        #     # print(f"key: {k}, value: {v}")
        #     # print(f"key_tokens: {key_tokens}")
        #     # print(f"item_tokens: {item_tokens}")
        #     # print(f"start_index: {start_index}")
        #     # print(f"item_start_index: {item_start_index}")
        #     item_probs = probs[item_start_index:item_start_index+len(item_tokens)]
        #     item_probs = np.mean(item_probs)
        #     # print(f"item_probs: {item_probs}")
        #     output[k + "_probabilities"] = item_probs

        # print(f"output: {output}")
                
            
        # company_name = result_dict.get("Company", "")
        # company_name_tokens = tokenizer(company_name, )['input_ids']
        # print(f"company_name: {company_name}")
        # print(f"company_name_tokens: {company_name_tokens}") 

        # start_index = find_sublist_matches(output_tokens, company_name_tokens)
        # print(f"start_index: {start_index}")

        # output["会社名"] = company_name
        # company_name_probs = probs[start_index:start_index+len(company_name_tokens)]
        # company_name_probs = np.mean(company_name_probs)
        # output["会社名_probabilities"] = company_name_probs

        # print(f"output: {output}")

        result_json.append({data["image"]: result})

    # 結果を保存
    with open(args.output_path, "w") as f:
        for result in result_json:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()

    test_ocr_images(args)
    # print()
