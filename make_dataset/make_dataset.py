import sys
import json
import time as timer
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from random import randint
from urllib.request import urlopen
import argparse
import numpy as np
import requests
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from domain.business_card import BusinessCard

import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

IMAGE_SIZE = (700, 500)
PICSUM_URL = f"https://picsum.photos/{IMAGE_SIZE[0]}/{IMAGE_SIZE[1]}"
FONT_PATH = "https://github.com/googlefonts/morisawa-biz-ud-mincho/raw/main/fonts/ttf/BIZUDPMincho-Regular.ttf"

def color_invert(r: int, g: int, b: int) -> str:
    mono = (0.114 * r) + (0.587 * g) + (0.299 * b)
    if mono >= 127:
        return "#000000"

    return "#FFFFFF"


def fetch_image_from_url(url: str, timeout: int) -> Image.Image | None:
    try:
        image = Image.open(BytesIO(requests.get(url, timeout=timeout).content))
    except (TypeError, ValueError, ConnectionError, OSError, BufferError):
        print("Failed to get image: %s", url)
        return None
    if image.mode != "RGB":
        print("Convert image mode to RGB: %s", url)
        image = image.convert("RGB")
    return image


def dummy_business_card(i: int, args) -> BusinessCard | None:
    image = fetch_image_from_url(PICSUM_URL, args.timeout)
    if image is None:
        return None

    mean_color = np.mean(np.array(image), axis=(0, 1)).astype(int)

    text_color = color_invert(*mean_color)
    draw = ImageDraw.Draw(image)

    faker = Faker("ja_JP")

    text_x = randint(50, 100)

    # 左上の適当な位置とサイズを選び、会社名を書く
    company_point_x, company_point_y, company_size = (
        text_x,
        randint(50, 100),
        randint(20, 30),
    )
    company_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        company_size,
    )
    company = faker.company()
    draw.text(
        (company_point_x, company_point_y),
        company,
        font=company_font,
        fill=text_color,
    )
    company_bounding_box = draw.textbbox(
        (company_point_x, company_point_y),
        company,
        font=company_font,
    )

    # 会社名の下に適当な位置とサイズで名前を書く
    name_point_x, name_point_y, name_size = (
        text_x,
        company_bounding_box[3] + randint(10, 20),
        randint(30, 50),
    )
    name_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        name_size,
    )
    name = faker.name()
    draw.text(
        (name_point_x, name_point_y),
        name,
        font=name_font,
        fill=text_color,
    )

    detail_font_size = 20
    # 左下の適当な位置とサイズでメールアドレスを書く
    email_point_x, email_point_y = (
        text_x,
        randint(320, 360),
    )
    email_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        detail_font_size,
    )
    email = faker.email()
    draw.text(
        (email_point_x, email_point_y),
        email,
        font=email_font,
        fill=text_color,
    )
    email_bounding_box = draw.textbbox(
        (email_point_x, email_point_y),
        email,
        font=email_font,
    )

    # メールアドレスの下に電話番号を書く
    phone_point_x, phone_point_y = (
        text_x,
        email_bounding_box[3] + 5,
    )
    phone_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        detail_font_size,
    )
    phone = faker.phone_number()
    draw.text(
        (phone_point_x, phone_point_y),
        phone,
        font=phone_font,
        fill=text_color,
    )
    phone_bounding_box = draw.textbbox(
        (phone_point_x, phone_point_y),
        phone,
        font=phone_font,
    )

    # 電話番号の下に会社URLを書く
    url_point_x, url_point_y = (
        text_x,
        phone_bounding_box[3] + 5,
    )
    url_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        detail_font_size,
    )
    url = faker.url()
    draw.text(
        (url_point_x, url_point_y),
        url,
        font=url_font,
        fill=text_color,
    )
    url_bounding_box = draw.textbbox(
        (url_point_x, url_point_y),
        url,
        font=url_font,
    )

    # 会社URLの下に住所を書く
    address_point_x, address_point_y = (
        text_x,
        url_bounding_box[3] + 5,
    )
    address_font = ImageFont.truetype(
        urlopen(FONT_PATH),
        detail_font_size,
    )
    address = faker.address()
    draw.text(
        (address_point_x, address_point_y),
        address,
        font=address_font,
        fill=text_color,
    )

    # 画像を保存
    image.save(f"{args.image_dir}/{i}.png")

    business_card = BusinessCard(
        image_path=f"{args.image_dir}/{i}.png",
        prompt=args.prompt,
        company=company,
        name=name,
        email=email,
        phone_number=phone,
        address=address,
    )
    return business_card.create_conversations()

import concurrent.futures

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--dataset_length", type=int)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--prompt", type=str)
    args = parser.parse_args()

    cards = []
    start_time = timer.time()
    print("start make dataset")
    i = 0

    with tqdm() as pbar:
        while i < args.dataset_length:
            card = dummy_business_card(i, args)
            if card is None:
                # 失敗したら無視 or 必要なら再試行
                continue
            card["id"] = i
            i += 1
            print(card)
            cards.append(card)
            pbar.update(1)

    print(f"Generated {len(cards)} cards")
    print(f"end make dataset")
    print(f"処理時間: {int(timer.time() - start_time)}秒")

    with open(args.label_path, "w") as f:
        for c in cards:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")