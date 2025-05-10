import os
import random
import numpy as np
import json
from tqdm import tqdm
from faker import Faker
from PIL import Image, ImageDraw, ImageFont
Faker.seed(0)
random.seed(0)
# 日本語ロケールの Faker を用意
fake = Faker("ja_JP")

from faker.providers.address import Provider as AddressProvider

class JapaneseAddressProvider(AddressProvider):
    """
    AddressProvider を継承し street_suffixes を日本語向けに上書き
    """
    street_suffixes = ["丁目", "番地", "号"]  # 日本語の番地サフィックスのみ

    def street_address(self):
        """
        street_name() と組み合わせ、番地情報のみを返すようにカスタム
        """
        # 例: "西新宿2丁目3番地"
        return f"{self.street_name()}{self.street_suffix()}{self.building_number()}"
    
PROMPT="<image>\nPlease output Company Name, Department, Position, Family Name, Given Name, Family Name kana, Given Name kana, Mail Address, Phone Number, Mobile Number, Post Code, Prefecture, City Name, Address Details, Building, URL from address from business card."

# 名刺に含める項目のキー
FIELDS = [
    "Company Name", "Department", "Position",
    "Family Name", "Given Name", "Family Name kana", "Given Name kana",
    "Mail Address", "Phone Number", "Mobile Number",
    "Post Code", "Prefecture", "City Name", "Address Details", "Building",
    "URL"
]

# フォントファイルへのパス（環境に合わせて変更）
FONT_PATH_LIST = [
    "/kaggle/input/BIZUDMincho-Bold.ttf",
    "/kaggle/input/BIZUDMincho-Regular.ttf",
    "/kaggle/input/BIZUDPMincho-Bold.ttf",
    "/kaggle/input/BIZUDPMincho-Regular.ttf",
]

# --- 背景生成関数群 ---

def generate_linear_gradient(size, start_color, end_color, horizontal=False):
    """線形グラデーション"""
    W, H = size
    base = Image.new('RGB', size, start_color)
    top  = Image.new('RGB', size, end_color)
    mask = Image.new('L', size)
    draw = ImageDraw.Draw(mask)
    for i in range(W if horizontal else H):
        v = int(255 * (i / (W-1) if horizontal else i / (H-1)))
        rect = ((i, 0, i+1, H) if horizontal else (0, i, W, i+1))
        draw.rectangle(rect, fill=v)
    return Image.composite(top, base, mask)
# :contentReference[oaicite:0]{index=0}

def generate_radial_gradient(size, inner_color, outer_color):
    """放射状（ラジアル）グラデーション"""
    W, H = size
    x = np.linspace(-1, 1, W)[None, :]
    y = np.linspace(-1, 1, H)[:, None]
    d = np.sqrt(x*x + y*y)
    d = np.clip(d, 0, 1)[..., None]
    start = np.array(inner_color, dtype=float)
    end   = np.array(outer_color, dtype=float)
    img = (start + (end - start) * d).astype(np.uint8)
    return Image.fromarray(img)
# :contentReference[oaicite:1]{index=1}

def generate_noise_texture(size, stddev=50):
    """ガウシアンノイズによるテクスチャ"""
    noise = Image.effect_noise(size, stddev).convert('RGB')
    return noise
# :contentReference[oaicite:2]{index=2}

def generate_stripes_pattern(size, stripe_width=20):
    """縦ストライプ模様"""
    W, H = size
    img = Image.new('RGB', size, '#ffffff')
    draw = ImageDraw.Draw(img)
    c1 = tuple(random.choices(range(50,200), k=3))
    c2 = tuple(random.choices(range(200,256), k=3))
    for x in range(0, W, stripe_width*2):
        draw.rectangle([x,0,x+stripe_width,H], fill=c1)
    for x in range(stripe_width, W, stripe_width*2):
        draw.rectangle([x,0,x+stripe_width,H], fill=c2)
    return img
# :contentReference[oaicite:3]{index=3}

def generate_dot_grid_pattern(size, dot_spacing=30, dot_radius=5):
    """ドット＆グリッドパターン"""
    W, H = size
    img = Image.new('RGB', size, '#f0f0f0')
    draw = ImageDraw.Draw(img)
    col = tuple(random.choices(range(100,200), k=3))
    for y in range(0, H, dot_spacing):
        for x in range(0, W, dot_spacing):
            draw.ellipse([x-dot_radius, y-dot_radius, x+dot_radius, y+dot_radius], fill=col)
    return img
# :contentReference[oaicite:4]{index=4}

# 背景関数リスト
BACKGROUND_FUNCS = [
    lambda size: generate_linear_gradient(
        size,
        tuple(random.randint(200,255) for _ in range(3)),
        tuple(random.randint(0,200) for _ in range(3)),
        horizontal=random.choice([True, False])
    ),
    lambda size: generate_radial_gradient(
        size,
        tuple(random.randint(200,255) for _ in range(3)),
        tuple(random.randint(0,200) for _ in range(3))
    ),
    lambda size: generate_noise_texture(size, stddev=random.randint(30,80)),
    lambda size: generate_stripes_pattern(size, stripe_width=random.randint(10,30)),
    lambda size: generate_dot_grid_pattern(size,
                                         dot_spacing=random.randint(20,50),
                                         dot_radius=random.randint(3,8))
]

# --- 名刺データ生成関数群 ---
# 部署ベースと接尾辞リストから自動生成
DEPARTMENT_BASES = [
    '営業','人事','総務','開発','マーケティング','企画','経理','法務','広報','IT','技術','管理',
    '研究','運営','商品開発','品質管理','カスタマーサポート','プロジェクト','R&D','CSR','システム',
    '製造','購買','物流','資材','安全管理','内部監査','品質保証','事業企画','経営企画','財務','調達',
    '製品設計','UXデザイン','営業推進','事業開発','教育','情報','広告','投資','法務',
]
SUFFIXES = ['部','部署','支店']
# 特殊な部署名称
SPECIAL_DEPTS = ['本社','本店']
# 複合名称（例：〇〇事業部〇〇部）
COMPLEX_BASES = ['営業','開発','商品開発','事業','ビジネス']
DEPARTMENT_SUFFIXES = [
    # ベース + 接尾辞
    *(f + s for f in DEPARTMENT_BASES for s in SUFFIXES),
    # 本社／本店
    *SPECIAL_DEPTS,
    # 複合パターン
    *(f + '事業部' + s for f in COMPLEX_BASES for s in SUFFIXES),
]

def color_invert(r: int, g: int, b: int) -> str:
    mono = (0.114 * r) + (0.587 * g) + (0.299 * b)
    if mono >= 127:
        return "#000000"

    return "#FFFFFF"

def generate_department():
    return random.choice(DEPARTMENT_SUFFIXES)

def generate_position():
    return fake.job()

def generate_building():
    return fake.building_number()

# --- 名刺生成処理 ---

def generate_one_card(idx: int, output_dir: str) -> dict:
    """1枚の名刺を生成し、項目データを辞書で返す"""
    W, H = 850, 540
    # 背景をランダム選択
    bg = random.choice(BACKGROUND_FUNCS)((W, H))
    img = bg.copy()

    mean_color = np.mean(np.array(img), axis=(0, 1)).astype(int)

    text_color = color_invert(*mean_color)

    draw = ImageDraw.Draw(img)

    # フォント設定
    FONT_PATH = random.choice(FONT_PATH_LIST)
    name_font  = ImageFont.truetype(FONT_PATH, size=36)
    small_font = ImageFont.truetype(FONT_PATH, size=24)

    data = {
        "Company Name":     fake.company(),
        "Department":       generate_department(),
        "Position":         generate_position(),
        "Family Name":      fake.last_name(),
        "Given Name":       fake.first_name(),
        "Family Name kana": fake.last_kana_name(),
        "Given Name kana":  fake.first_kana_name(),
        "Mail Address":     fake.ascii_free_email(),
        "Phone Number":     fake.phone_number(),
        "Mobile Number":    fake.phone_number(),
        "Post Code":        fake.postcode(),
        "Prefecture":       fake.prefecture(),
        "City Name":        fake.city(),
        "Address Details":  f"{fake.chome()}{fake.ban()}{fake.building_number()}",
        "Building":         fake.building_name()+fake.building_number(),
        "URL":              fake.url()
    }

    # テキスト描画
    margin_x, margin_y = 50, 50
    line_h = 50
    x, y = margin_x, margin_y

    # 会社名・部署・役職
    draw.text((x, y), data["Company Name"], font=name_font,  fill=text_color)
    y += line_h
    draw.text((x, y),
              f"{data['Department']} ／ {data['Position']}",
              font=small_font, fill=text_color)
    y += line_h

    # 氏名
    full_name = (
        f"{data['Family Name']} {data['Given Name']} "
        f"({data['Family Name kana']} {data['Given Name kana']})"
    )
    draw.text((x, y), full_name, font=name_font, fill=text_color)
    y += int(line_h * 1.2)

    # メールアドレス
    draw.text((x, y), f"Mail: {data['Mail Address']}", font=small_font, fill=text_color)
    y += int(line_h * 0.7)
    # 電話番号
    draw.text((x, y), f"Tel: {data['Phone Number']}", font=small_font, fill=text_color)
    # y += int(line_h * 0.5)
    mobile_x = x + int(x * 6)
    # 携帯番号
    draw.text((mobile_x, y), f"Mobile: {data['Mobile Number']}", font=small_font, fill=text_color)
    y += int(line_h * 0.7)


    # 郵便番号
    draw.text((x, y), f"Post Code: {data['Post Code']}", font=small_font, fill=text_color)
    y += int(line_h * 0.7)
    # 住所
    draw.text((x, y), f"Address: {data['Prefecture']} {data['City Name']} {data['Address Details']}", font=small_font, fill=text_color)
    y += int(line_h * 0.7)
    # 建物名
    draw.text((x, y), f"Building: {data['Building']}", font=small_font, fill=text_color)
    y += int(line_h * 0.7)
    # URL
    draw.text((x, y), f"URL: {data['URL']}", font=small_font, fill=text_color)
    y += int(line_h * 0.7)

    # ファイル出力
    os.makedirs(output_dir, exist_ok=True)
    filename = f"business_card_{idx+1}.png"
    img.save(os.path.join(output_dir, filename))

    value_str = "{"
    for key in FIELDS:
        value_str += f'"{key}":"{data[key]}",'
    value_str = value_str[:-1] + "}"

    return {"id":idx, "image": os.path.join(output_dir, filename), "conversations": [{"from": "human", "value": PROMPT}, {"from": "gpt", "value": value_str}]}

def generate_business_cards(n: int, output_dir: str = "cards") -> list:
    """任意枚数の名刺を一括生成してリストで返す"""
    result = []
    for i in tqdm(range(n)):
        result.append(generate_one_card(i, output_dir))
    return result

if __name__ == "__main__":
    MAKE_NUMBER = 10000
    # 例：名刺を output_dir フォルダに生成
    cards = generate_business_cards(MAKE_NUMBER, output_dir="/kaggle/input/generate_image/train/image2")
    for c in cards:
        print(c)
    # jsonlで保存
    with open("/kaggle/input/generate_image/train/labels2.jsonl", "w") as f:
        for card in cards:
            f.write(json.dumps(card, ensure_ascii=False) + "\n")
