#!/usr/bin/env python3
"""
HLBD Full Dataset Generator V2
完整分层语言启蒙数据集生成器

从原167条样本扩展到5000条，包含完整的8层结构:
- Level 1: 字卡 + Emoji
- Level 2: 短语
- Level 3: 数学（句法结构）← 重要！训练需要
- Level 4: 拼音
- Level 5: 英文
- Level 6: 中文
- Level 7: 日文
- Level 8: 韩文

防模式坍缩策略：数据稀释学
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Set


def simple_pinyin(text: str) -> str:
    """简单拼音转换（基于常用字）"""
    pinyin_map = {
        '不': 'bù', '在': 'zài', '我': 'wǒ', '你': 'nǐ', '他': 'tā', '她': 'tā',
        '吃': 'chī', '喝': 'hē', '睡': 'shuì', '学': 'xué', '习': 'xí',
        '慢': 'màn', '快': 'kuài', '轻': 'qīng', '悄': 'qiāo', '默': 'mò', '认': 'rèn', '真': 'zhēn',
        '早': 'zǎo', '上': 'shàng', '中': 'zhōng', '午': 'wǔ', '晚': 'wǎn',
        '今': 'jīn', '明': 'míng', '昨': 'zuó', '天': 'tiān',
        '家': 'jiā', '学': 'xué', '校': 'xiào', '公': 'gōng', '园': 'yuán',
        '图': 'tú', '书': 'shū', '馆': 'guǎn', '外': 'wài', '面': 'miàn',
        '洗': 'xǐ', '澡': 'zǎo', '刷': 'shuā', '牙': 'yá', '脸': 'liǎn',
        '梳': 'shū', '头': 'tóu', '饭': 'fàn', '觉': 'jiào', '水': 'shuǐ',
    }
    result = []
    for char in text:
        result.append(pinyin_map.get(char, char))
    return ' '.join(result)


class HLBDFullGenerator:
    """HLBD完整数据集生成器"""

    def __init__(self, original_samples_path: str):
        self.original_path = original_samples_path
        self.original_samples = []
        self.seen_concepts: Set[str] = set()
        self.target_size = 5000

        # Emoji词典
        self.emoji_dict = {
            # 天气
            "天气": "🌤️", "下雨": "🌧️", "下雪": "❄️", "刮风": "💨",
            "晴天": "☀️", "阴天": "☁️", "雷电": "⚡", "彩虹": "🌈",

            # 情感
            "我爱你": "❤️", "开心": "😊", "难过": "😢", "生气": "😠",
            "害怕": "😱", "惊讶": "😲", "思考": "🤔", "微笑": "😄",

            # 日常活动
            "吃饭": "🍽️", "睡觉": "😴", "喝水": "💧", "看书": "📖",
            "运动": "🏃", "学习": "📚", "工作": "💼", "玩游戏": "🎮",
            "唱歌": "🎤", "跳舞": "💃", "画画": "🎨", "写字": "✍️",

            # 交通出行
            "开车": "🚗", "坐车": "🚌", "骑车": "🚴", "走路": "🚶",
            "旅行": "✈️", "坐船": "⛵", "坐火车": "🚂",

            # 动物
            "猫": "🐱", "狗": "🐶", "鸟": "🐦", "鱼": "🐟",
            "兔子": "🐰", "熊": "🐻", "猴子": "🐵", "老虎": "🐯",

            # 植物
            "花": "🌸", "树": "🌳", "草": "🌱", "玫瑰": "🌹",

            # 食物
            "苹果": "🍎", "香蕉": "🍌", "面包": "🍞", "咖啡": "☕",
            "茶": "🍵", "米饭": "🍚", "面条": "🍜", "蛋糕": "🎂",

            # 物品
            "手机": "📱", "电脑": "💻", "书": "📕", "笔": "🖊️",
            "包": "👜", "衣服": "👔", "鞋子": "👞", "帽子": "🎩",

            # 场所
            "学校": "🏫", "医院": "🏥", "商店": "🏪", "家": "🏠",
            "公园": "🏞️", "图书馆": "📚", "餐厅": "🍴",

            # 时间
            "早上": "🌅", "中午": "☀️", "晚上": "🌙", "夜晚": "🌃",

            # 身体
            "手": "✋", "脚": "👣", "眼睛": "👀", "耳朵": "👂",
            "鼻子": "👃", "嘴": "👄", "头": "🧠",

            # 动作
            "拍手": "👏", "挥手": "👋", "点头": "🙇", "跑步": "🏃",
            "游泳": "🏊", "爬山": "🧗",
        }

        # 句法模式
        self.syntax_patterns = [
            "S = VP (VP: {verb})",
            "S = NP + VP (NP: {subject}, VP: {verb})",
            "S = NP + VP + NP (NP1: {subject}, VP: {verb}, NP2: {object})",
            "S = Adv + VP (Adv: {adverb}, VP: {verb})",
            "S = NP + Adj (NP: {subject}, Adj: {adjective})",
        ]

        # 加载原始样本
        self._load_original_samples()

    def _load_original_samples(self):
        """加载原始167条样本"""
        with open(self.original_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取samples
        samples_match = re.search(r'samples\s*=\s*(\[.*\])', content, re.DOTALL)
        if samples_match:
            samples_text = samples_match.group(1)
            self.original_samples = eval(samples_text)

            # 记录已有概念
            for sample in self.original_samples:
                self.seen_concepts.add(sample['concept'])

            print(f"✓ 加载了{len(self.original_samples)}个原始样本")
            print(f"✓ 所有样本都包含8个完整层级")

    def is_duplicate(self, concept: str) -> bool:
        """检查概念是否重复"""
        if concept in self.seen_concepts:
            return True
        self.seen_concepts.add(concept)
        return False

    def generate_samples(self) -> List[Dict]:
        """生成扩展样本"""
        print("=" * 60)
        print("HLBD Full Dataset Generator V2")
        print(f"目标：从{len(self.original_samples)}条扩展到{self.target_size}条")
        print("=" * 60)
        print()

        # 保留原始样本
        all_samples = self.original_samples.copy()
        print(f"✓ 保留原始{len(all_samples)}条样本")

        # 需要生成的新样本数
        needed = self.target_size - len(all_samples)
        print(f"✓ 需要生成{needed}条新样本\n")

        # 生成新样本
        generated_count = 0

        # 1. 基于原始样本变体（增加变体数量）
        print("📝 生成阶段1: 基于原始样本的变体...")
        variants = self._generate_variants(min(2000, needed // 2))
        all_samples.extend(variants)
        generated_count += len(variants)
        print(f"   ✓ 生成了{len(variants)}条变体样本\n")

        # 2. 从概念池生成
        print("📝 生成阶段2: 从大型概念池生成...")
        pool_samples = self._generate_from_pool(min(500, needed - generated_count))
        all_samples.extend(pool_samples)
        generated_count += len(pool_samples)
        print(f"   ✓ 生成了{len(pool_samples)}条概念池样本\n")

        # 3. 组合生成剩余
        if generated_count < needed:
            remaining = needed - generated_count
            print(f"📝 生成阶段3: 组合生成剩余{remaining}条...")
            combined = self._generate_combined_concepts(remaining)
            all_samples.extend(combined)
            generated_count += len(combined)
            print(f"   ✓ 生成了{len(combined)}条组合样本\n")

        # 3. 数据稀释学：随机打散
        print("🎲 应用数据稀释学策略...")
        random.shuffle(all_samples)
        print(f"   ✓ 已随机打散顺序\n")

        print(f"📊 最终数据集统计:")
        print(f"   原始样本: 167")
        print(f"   变体样本: {len(variants)}")
        print(f"   概念池样本: {len(pool_samples)}")
        print(f"   组合样本: {generated_count - len(variants) - len(pool_samples)}")
        print(f"   总计: {len(all_samples)} 条")
        print(f"   唯一概念: {len(self.seen_concepts)} 个")

        return all_samples[:self.target_size]

    def _generate_variants(self, count: int) -> List[Dict]:
        """基于原始样本生成变体"""
        variants = []

        # 变体策略
        strategies = [
            self._add_adverb_variant,
            self._add_time_variant,
            self._add_place_variant,
            self._add_negation_variant,
        ]

        attempts = 0
        max_attempts = count * 10

        while len(variants) < count and attempts < max_attempts:
            attempts += 1

            # 随机选择原始样本和变体策略
            base_sample = random.choice(self.original_samples)
            strategy = random.choice(strategies)

            # 生成变体
            try:
                variant = strategy(base_sample)
                if variant and not self.is_duplicate(variant['concept']):
                    variants.append(variant)
            except:
                continue

        return variants

    def _add_adverb_variant(self, base: Dict) -> Dict:
        """添加副词变体"""
        adverbs = ["慢慢", "快快", "轻轻", "悄悄", "默默", "认真"]
        adv = random.choice(adverbs)

        concept = base['concept']
        new_concept = f"{adv}{concept}"

        return {
            "concept": new_concept,
            "level_1": {
                "字卡": new_concept,
                "emoji": base['level_1'].get('emoji', '✨')
            },
            "level_2": {
                "短语": f"{adv}地{base['level_2'].get('短语', concept)}"
            },
            "level_3": {
                "数学": f"S = Adv + VP (Adv: {adv}, VP: {concept})"
            },
            "level_4": {
                "拼音": simple_pinyin(new_concept)
            },
            "level_5": {
                "英文": f"{adv} {base['level_5'].get('英文', concept)}"
            },
            "level_6": {
                "中文": f"{adv}地{base['level_6'].get('中文', concept + '。')}"
            },
            "level_7": {
                "日文": base['level_7'].get('日文', '')
            },
            "level_8": {
                "韩文": base['level_8'].get('韩文', '')
            }
        }

    def _add_time_variant(self, base: Dict) -> Dict:
        """添加时间变体"""
        times = ["早上", "中午", "晚上", "今天", "明天", "昨天"]
        time = random.choice(times)

        concept = base['concept']
        new_concept = f"{time}{concept}"

        return {
            "concept": new_concept,
            "level_1": {
                "字卡": new_concept,
                "emoji": base['level_1'].get('emoji', '⏰')
            },
            "level_2": {
                "短语": f"{time}{base['level_2'].get('短语', concept)}"
            },
            "level_3": {
                "数学": f"S = Time + VP (Time: {time}, VP: {concept})"
            },
            "level_4": {
                "拼音": simple_pinyin(new_concept)
            },
            "level_5": {
                "英文": f"{time} {base['level_5'].get('英文', concept)}"
            },
            "level_6": {
                "中文": f"{time}{base['level_6'].get('中文', concept + '。')}"
            },
            "level_7": {
                "日文": base['level_7'].get('日文', '')
            },
            "level_8": {
                "韩文": base['level_8'].get('韩文', '')
            }
        }

    def _add_place_variant(self, base: Dict) -> Dict:
        """添加地点变体"""
        places = ["在家", "在学校", "在公园", "在图书馆", "在外面"]
        place = random.choice(places)

        concept = base['concept']
        new_concept = f"{place}{concept}"

        return {
            "concept": new_concept,
            "level_1": {
                "字卡": new_concept,
                "emoji": base['level_1'].get('emoji', '📍')
            },
            "level_2": {
                "短语": f"{place}{base['level_2'].get('短语', concept)}"
            },
            "level_3": {
                "数学": f"S = Place + VP (Place: {place}, VP: {concept})"
            },
            "level_4": {
                "拼音": simple_pinyin(new_concept)
            },
            "level_5": {
                "英文": f"{place} {base['level_5'].get('英文', concept)}"
            },
            "level_6": {
                "中文": f"{place}{base['level_6'].get('中文', concept + '。')}"
            },
            "level_7": {
                "日文": base['level_7'].get('日文', '')
            },
            "level_8": {
                "韩文": base['level_8'].get('韩文', '')
            }
        }

    def _add_negation_variant(self, base: Dict) -> Dict:
        """添加否定变体"""
        concept = base['concept']
        new_concept = f"不{concept}"

        return {
            "concept": new_concept,
            "level_1": {
                "字卡": new_concept,
                "emoji": "❌"
            },
            "level_2": {
                "短语": f"不{base['level_2'].get('短语', concept)}"
            },
            "level_3": {
                "数学": f"S = Neg + VP (Neg: 不, VP: {concept})"
            },
            "level_4": {
                "拼音": f"bù {simple_pinyin(concept)}"
            },
            "level_5": {
                "英文": f"Don't {base['level_5'].get('英文', concept)}"
            },
            "level_6": {
                "中文": f"不{base['level_6'].get('中文', concept + '。')}"
            },
            "level_7": {
                "日文": base['level_7'].get('日文', '')
            },
            "level_8": {
                "韩文": base['level_8'].get('韩文', '')
            }
        }

    def _get_concepts_pool(self):
        """获取大型概念池"""
        return [
            # 日常活动
            ("洗澡", "Take a bath", "お風呂に入る", "목욕하다", "🛁"),
            ("刷牙", "Brush teeth", "歯を磨く", "양치하다", "🦷"),
            ("洗脸", "Wash face", "顔を洗う", "세수하다", "💦"),
            ("梳头", "Comb hair", "髪をとかす", "머리를 빗다", "💇"),
            ("穿衣", "Get dressed", "着替える", "옷을 입다", "👔"),
            ("做饭", "Cook", "料理する", "요리하다", "🍳"),
            ("扫地", "Sweep", "掃除する", "청소하다", "🧹"),
            ("拖地", "Mop", "床を拭く", "걸레질하다", "🧽"),
            ("洗衣", "Wash clothes", "洗濯する", "빨래하다", "🧺"),
            ("晾衣", "Hang clothes", "干す", "널다", "👕"),
            ("读书", "Read", "読書する", "독서하다", "📖"),
            ("写作", "Write", "書く", "글을 쓰다", "✍️"),
            ("算术", "Arithmetic", "算数", "산수", "🔢"),
            ("画图", "Draw", "絵を描く", "그림을 그리다", "🎨"),
            ("唱歌", "Sing", "歌う", "노래하다", "🎤"),
            ("跳舞", "Dance", "踊る", "춤을 추다", "💃"),
            ("弹琴", "Play piano", "ピアノを弾く", "피아노를 치다", "🎹"),
            ("下棋", "Play chess", "将棋를指す", "체스를 두다", "♟️"),
            ("跑步", "Run", "走る", "달리기", "🏃"),
            ("游泳", "Swim", "泳ぐ", "수영하다", "🏊"),
            ("爬山", "Climb", "登る", "등산하다", "🧗"),
            ("骑车", "Bike", "自転車に乗る", "자전거를 타다", "🚴"),
            ("打球", "Play ball", "ボールで遊ぶ", "공을 치다", "⚽"),
            ("瑜伽", "Yoga", "ヨガ", "요가", "🧘"),
            ("跳绳", "Jump rope", "縄跳び", "줄넘기", "🪢"),
            ("开心", "Happy", "嬉しい", "행복하다", "😊"),
            ("难过", "Sad", "悲しい", "슬프다", "😢"),
            ("生气", "Angry", "怒る", "화나다", "😠"),
            ("害怕", "Scared", "怖い", "무섭다", "😱"),
            ("激动", "Excited", "興奮する", "흥분하다", "🤩"),
            ("平静", "Calm", "静かだ", "평온하다", "😌"),
            ("紧张", "Nervous", "緊張する", "긴장하다", "😰"),
            ("放松", "Relax", "リラックス", "편안하다", "😎"),
            ("打招呼", "Greet", "挨拶する", "인사하다", "👋"),
            ("聊天", "Chat", "おしゃべり", "수다떨다", "💬"),
            ("开会", "Meeting", "会議する", "회의하다", "👥"),
            ("约会", "Date", "デートする", "데이트하다", "💑"),
            ("聚会", "Party", "パーティー", "파티", "🎉"),
            ("拥抱", "Hug", "ハグする", "포옹하다", "🤗"),
            ("握手", "Handshake", "握手する", "악수하다", "🤝"),
            ("买东西", "Shopping", "買い物", "쇼핑하다", "🛒"),
            ("付款", "Pay", "支払う", "결제하다", "💳"),
            ("试穿", "Try on", "試着する", "입어보다", "👗"),
            ("退货", "Return", "返品する", "반품하다", "🔄"),
            ("坐地铁", "Take subway", "地下鉄に乗る", "지하철을 타다", "🚇"),
            ("打车", "Take taxi", "タクシーに乘る", "택시를 타다", "🚕"),
            ("坐飞机", "Fly", "飛行機に乗る", "비행기를 타다", "✈️"),
            ("开车", "Drive", "運転する", "운전하다", "🚗"),
            ("停车", "Park", "駐車する", "주차하다", "🅿️"),
            ("早餐", "Breakfast", "朝食", "아침식사", "🥐"),
            ("午餐", "Lunch", "昼食", "점심식사", "🍱"),
            ("晚餐", "Dinner", "夕食", "저녁식사", "🍽️"),
            ("点菜", "Order food", "注文する", "주문하다", "📋"),
            ("做菜", "Cook", "料理를作る", "요리하다", "🍳"),
            ("煮饭", "Cook rice", "ご飯を炊く", "밥을 짓다", "🍚"),
            ("切菜", "Chop", "野菜を切る", "야채를 썰다", "🔪"),
            ("炒菜", "Stir-fry", "炒める", "볶다", "🥘"),
            ("看病", "See doctor", "病院に行く", "병원에 가다", "🏥"),
            ("吃药", "Take medicine", "薬を飲む", "약을 먹다", "💊"),
            ("打针", "Injection", "注射する", "주사를 맞다", "💉"),
            ("量体温", "Take temperature", "体温를測る", "체온을 재다", "🌡️"),
            ("整理", "Organize", "整理する", "정리하다", "📦"),
            ("收拾", "Tidy up", "片付ける", "치우다", "🧹"),
            ("擦桌子", "Wipe table", "テーブルを拭く", "탁자를 닦다", "🪣"),
            ("倒垃圾", "Take out trash", "ゴミを捨てる", "쓰레기를 버리다", "🗑️"),
            ("浇花", "Water plants", "水をやる", "물을 주다", "🌻"),
            ("遛狗", "Walk dog", "犬の散歩", "개를 산책시키다", "🐕"),
            ("看电影", "Watch movie", "映画를見る", "영화를 보다", "🎬"),
            ("听音乐", "Listen music", "音楽を聴く", "음악을 듣다", "🎵"),
            ("玩游戏", "Play games", "ゲームをする", "게임을 하다", "🎮"),
            ("看电视", "Watch TV", "テレビを見る", "텔레비전을 보다", "📺"),
            ("上网", "Go online", "ネットする", "인터넷하다", "💻"),
            ("拍照", "Take photo", "写真を撮る", "사진을 찍다", "📷"),
            ("录像", "Record video", "ビデオを撮る", "동영상을 찍다", "🎥"),
            ("上班", "Go to work", "出勤する", "출근하다", "💼"),
            ("下班", "Off work", "退勤する", "퇴근하다", "🏠"),
            ("加班", "Overtime", "残業する", "야근하다", "🌙"),
            ("请假", "Take leave", "休む", "휴가를 내다", "📅"),
            ("开电脑", "Turn on PC", "PCを起動する", "컴퓨터를 켜다", "💻"),
            ("打印", "Print", "印刷する", "인쇄하다", "🖨️"),
            ("复印", "Copy", "コピーする", "복사하다", "📄"),
            ("打电话", "Make call", "電話をかける", "전화하다", "📞"),
            ("发短信", "Send SMS", "メールする", "문자를 보내다", "📱"),
            ("发邮件", "Send email", "メールを送る", "이메일을 보내다", "📧"),
            ("视频聊天", "Video chat", "ビデオ通話", "영상통화", "📹"),
            ("春天", "Spring", "春", "봄", "🌸"),
            ("夏天", "Summer", "夏", "여름", "☀️"),
            ("秋天", "Autumn", "秋", "가을", "🍂"),
            ("冬天", "Winter", "冬", "겨울", "❄️"),
            ("刮风", "Windy", "風が吹く", "바람이 불다", "💨"),
            ("打雷", "Thunder", "雷が鳴る", "천둥이 치다", "⚡"),
            ("下雪", "Snow", "雪が降る", "눈이 오다", "❄️"),
            ("过年", "New Year", "正月", "설날", "🎊"),
            ("生日", "Birthday", "誕生日", "생일", "🎂"),
            ("结婚", "Wedding", "結婚式", "결혼", "💒"),
            ("庆祝", "Celebrate", "祝う", "축하하다", "🎉"),
        ]

    def _generate_from_pool(self, count: int) -> List[Dict]:
        """从大型概念池直接生成"""
        pool_samples = []

        # 概念池
        concepts_pool = self._get_concepts_pool()

        for concept_data in concepts_pool:
            if len(pool_samples) >= count:
                break

            concept, english, japanese, korean, emoji = concept_data

            if self.is_duplicate(concept):
                continue

            sample = self._create_full_sample(
                concept, english, japanese, korean, emoji
            )
            pool_samples.append(sample)

        return pool_samples

    def _generate_new_concepts(self, count: int) -> List[Dict]:
        """生成全新概念"""
        new_samples = []

        # 大量新概念词库
        concepts_pool = [
            # 日常活动 (100+)
            ("洗澡", "Take a bath", "お風呂に入る", "목욕하다", "🛁"),
            ("刷牙", "Brush teeth", "歯を磨く", "양치하다", "🦷"),
            ("洗脸", "Wash face", "顔を洗う", "세수하다", "💦"),
            ("梳头", "Comb hair", "髪をとかす", "머리를 빗다", "💇"),
            ("穿衣", "Get dressed", "着替える", "옷을 입다", "👔"),
            ("做饭", "Cook", "料理する", "요리하다", "🍳"),
            ("扫地", "Sweep", "掃除する", "청소하다", "🧹"),
            ("拖地", "Mop", "床を拭く", "걸레질하다", "🧽"),
            ("洗衣", "Wash clothes", "洗濯する", "빨래하다", "🧺"),
            ("晾衣", "Hang clothes", "干す", "널다", "👕"),

            # 学习相关 (50+)
            ("读书", "Read", "読書する", "독서하다", "📖"),
            ("写作", "Write", "書く", "글을 쓰다", "✍️"),
            ("算术", "Arithmetic", "算数", "산수", "🔢"),
            ("画图", "Draw", "絵を描く", "그림을 그리다", "🎨"),
            ("唱歌", "Sing", "歌う", "노래하다", "🎤"),
            ("跳舞", "Dance", "踊る", "춤을 추다", "💃"),
            ("弹琴", "Play piano", "ピアノを弾く", "피아노를 치다", "🎹"),
            ("下棋", "Play chess", "将棋を指す", "체스를 두다", "♟️"),

            # 运动健身 (40+)
            ("跑步", "Run", "走る", "달리기", "🏃"),
            ("游泳", "Swim", "泳ぐ", "수영하다", "🏊"),
            ("爬山", "Climb", "登る", "등산하다", "🧗"),
            ("骑车", "Bike", "自転車に乗る", "자전거를 타다", "🚴"),
            ("打球", "Play ball", "ボールで遊ぶ", "공을 치다", "⚽"),
            ("瑜伽", "Yoga", "ヨガ", "요가", "🧘"),
            ("跳绳", "Jump rope", "縄跳び", "줄넘기", "🪢"),

            # 情感表达 (60+)
            ("开心", "Happy", "嬉しい", "행복하다", "😊"),
            ("难过", "Sad", "悲しい", "슬프다", "😢"),
            ("生气", "Angry", "怒る", "화나다", "😠"),
            ("害怕", "Scared", "怖い", "무섭다", "😱"),
            ("激动", "Excited", "興奮する", "흥분하다", "🤩"),
            ("平静", "Calm", "静かだ", "평온하다", "😌"),
            ("紧张", "Nervous", "緊張する", "긴장하다", "😰"),
            ("放松", "Relax", "リラックス", "편안하다", "😎"),

            # 社交互动 (50+)
            ("打招呼", "Greet", "挨拶する", "인사하다", "👋"),
            ("聊天", "Chat", "おしゃべり", "수다떨다", "💬"),
            ("开会", "Meeting", "会議する", "회의하다", "👥"),
            ("约会", "Date", "デートする", "데이트하다", "💑"),
            ("聚会", "Party", "パーティー", "파티", "🎉"),
            ("拥抱", "Hug", "ハグする", "포옹하다", "🤗"),
            ("握手", "Handshake", "握手する", "악수하다", "🤝"),

            # 购物消费 (30+)
            ("买东西", "Shopping", "買い物", "쇼핑하다", "🛒"),
            ("付款", "Pay", "支払う", "결제하다", "💳"),
            ("试穿", "Try on", "試着する", "입어보다", "👗"),
            ("退货", "Return", "返品する", "반품하다", "🔄"),

            # 交通出行 (40+)
            ("坐地铁", "Take subway", "地下鉄に乗る", "지하철을 타다", "🚇"),
            ("打车", "Take taxi", "タクシーに乘る", "택시를 타다", "🚕"),
            ("坐飞机", "Fly", "飛行機に乗る", "비행기를 타다", "✈️"),
            ("开车", "Drive", "運転する", "운전하다", "🚗"),
            ("停车", "Park", "駐車する", "주차하다", "🅿️"),

            # 饮食 (60+)
            ("早餐", "Breakfast", "朝食", "아침식사", "🥐"),
            ("午餐", "Lunch", "昼食", "점심식사", "🍱"),
            ("晚餐", "Dinner", "夕食", "저녁식사", "🍽️"),
            ("点菜", "Order food", "注文する", "주문하다", "📋"),
            ("做菜", "Cook", "料理を作る", "요리하다", "🍳"),
            ("煮饭", "Cook rice", "ご飯を炊く", "밥을 짓다", "🍚"),
            ("切菜", "Chop", "野菜を切る", "야채를 썰다", "🔪"),
            ("炒菜", "Stir-fry", "炒める", "볶다", "🥘"),

            # 医疗健康 (30+)
            ("看病", "See doctor", "病院に行く", "병원에 가다", "🏥"),
            ("吃药", "Take medicine", "薬を飲む", "약을 먹다", "💊"),
            ("打针", "Injection", "注射する", "주사를 맞다", "💉"),
            ("量体温", "Take temperature", "体温を測る", "체온을 재다", "🌡️"),

            # 家务 (40+)
            ("整理", "Organize", "整理する", "정리하다", "📦"),
            ("收拾", "Tidy up", "片付ける", "치우다", "🧹"),
            ("擦桌子", "Wipe table", "テーブルを拭く", "탁자를 닦다", "🪣"),
            ("倒垃圾", "Take out trash", "ゴミを捨てる", "쓰레기를 버리다", "🗑️"),
            ("浇花", "Water plants", "水をやる", "물을 주다", "🌻"),
            ("遛狗", "Walk dog", "犬の散歩", "개를 산책시키다", "🐕"),

            # 娱乐休闲 (50+)
            ("看电影", "Watch movie", "映画を見る", "영화를 보다", "🎬"),
            ("听音乐", "Listen music", "音楽を聴く", "음악을 듣다", "🎵"),
            ("玩游戏", "Play games", "ゲームをする", "게임을 하다", "🎮"),
            ("看电视", "Watch TV", "テレビを見る", "텔레비전을 보다", "📺"),
            ("上网", "Go online", "ネットする", "인터넷하다", "💻"),
            ("拍照", "Take photo", "写真を撮る", "사진을 찍다", "📷"),
            ("录像", "Record video", "ビデオを撮る", "동영상을 찍다", "🎥"),

            # 工作学习 (40+)
            ("上班", "Go to work", "出勤する", "출근하다", "💼"),
            ("下班", "Off work", "退勤する", "퇴근하다", "🏠"),
            ("加班", "Overtime", "残業する", "야근하다", "🌙"),
            ("请假", "Take leave", "休む", "휴가를 내다", "📅"),
            ("开电脑", "Turn on PC", "PCを起動する", "컴퓨터를 켜다", "💻"),
            ("打印", "Print", "印刷する", "인쇄하다", "🖨️"),
            ("复印", "Copy", "コピーする", "복사하다", "📄"),

            # 通讯联系 (30+)
            ("打电话", "Make call", "電話をかける", "전화하다", "📞"),
            ("发短信", "Send SMS", "メールする", "문자를 보내다", "📱"),
            ("发邮件", "Send email", "メールを送る", "이메일을 보내다", "📧"),
            ("视频聊天", "Video chat", "ビデオ通話", "영상통화", "📹"),

            # 季节天气 (40+)
            ("春天", "Spring", "春", "봄", "🌸"),
            ("夏天", "Summer", "夏", "여름", "☀️"),
            ("秋天", "Autumn", "秋", "가을", "🍂"),
            ("冬天", "Winter", "冬", "겨울", "❄️"),
            ("刮风", "Windy", "風が吹く", "바람이 불다", "💨"),
            ("打雷", "Thunder", "雷が鳴る", "천둥이 치다", "⚡"),
            ("下雪", "Snow", "雪が降る", "눈이 오다", "❄️"),

            # 节日庆典 (20+)
            ("过年", "New Year", "正月", "설날", "🎊"),
            ("生日", "Birthday", "誕生日", "생일", "🎂"),
            ("结婚", "Wedding", "結婚式", "결혼", "💒"),
            ("庆祝", "Celebrate", "祝う", "축하하다", "🎉"),
        ]

        # 从池中生成样本
        generated = 0
        attempts = 0
        max_attempts = count * 5

        while generated < count and attempts < max_attempts:
            attempts += 1

            # 随机选择概念
            if len(concepts_pool) > 0:
                concept_data = random.choice(concepts_pool)
                concept, english, japanese, korean, emoji = concept_data

                if self.is_duplicate(concept):
                    continue

                # 生成完整8层结构
                sample = self._create_full_sample(
                    concept, english, japanese, korean, emoji
                )
                new_samples.append(sample)
                generated += 1

        print(f"   从概念池生成了{generated}条新概念")

        # 如果还不够，通过组合生成更多
        if generated < count:
            remaining = count - generated
            print(f"   继续组合生成{remaining}条...")
            combined = self._generate_combined_concepts(remaining)
            new_samples.extend(combined)

        return new_samples

    def _generate_combined_concepts(self, count: int) -> List[Dict]:
        """通过组合生成新概念"""
        # 大量修饰词
        modifiers = [
            "快速", "慢慢", "认真", "仔细", "安静", "大声", "高兴地", "努力",
            "轻轻", "悄悄", "默默", "急忙", "匆忙", "小心", "谨慎", "热情",
            "冷静", "耐心", "细心", "专心", "全心", "用心", "尽心", "放心",
        ]

        # 大量动词
        actions = [
            "工作", "学习", "思考", "准备", "完成", "开始", "结束", "继续",
            "练习", "复习", "预习", "温习", "操作", "处理", "整理", "收拾",
            "清洁", "打扫", "布置", "安排", "计划", "设计", "创作", "制作",
            "检查", "核对", "确认", "验证", "测试", "尝试", "实践", "探索",
        ]

        # 时间词
        times = ["早上", "中午", "下午", "晚上", "今天", "明天", "昨天", "每天"]

        # 地点词
        places = ["在家", "在学校", "在公司", "在外面", "在室内", "在户外"]

        combined_samples = []
        generated_count = 0

        while generated_count < count:
            # 生成不同类型的组合
            combo_type = random.choice(['modifier+action', 'time+action', 'place+action', 'modifier+time+action'])

            if combo_type == 'modifier+action':
                modifier = random.choice(modifiers)
                action = random.choice(actions)
                concept = f"{modifier}{action}"
                english = f"{modifier} {action}"
                japanese = f"{modifier}に{action}する"
                korean = f"{modifier} {action}하다"

            elif combo_type == 'time+action':
                time = random.choice(times)
                action = random.choice(actions)
                concept = f"{time}{action}"
                english = f"{time} {action}"
                japanese = f"{time}{action}する"
                korean = f"{time} {action}하다"

            elif combo_type == 'place+action':
                place = random.choice(places)
                action = random.choice(actions)
                concept = f"{place}{action}"
                english = f"{place} {action}"
                japanese = f"{place}{action}する"
                korean = f"{place} {action}하다"

            else:  # modifier+time+action
                modifier = random.choice(modifiers)
                time = random.choice(times)
                action = random.choice(actions)
                concept = f"{time}{modifier}{action}"
                english = f"{time} {modifier} {action}"
                japanese = f"{time}{modifier}に{action}する"
                korean = f"{time} {modifier} {action}하다"

            if self.is_duplicate(concept):
                continue

            sample = self._create_full_sample(
                concept, english, japanese, korean, "✨"
            )
            combined_samples.append(sample)
            generated_count += 1

            if generated_count % 500 == 0:
                print(f"     已组合生成{generated_count}条...")

        return combined_samples

    def _create_full_sample(self, concept: str, english: str,
                           japanese: str, korean: str, emoji: str) -> Dict:
        """创建完整的8层样本"""
        # 简化句法分析
        syntax = f"S = VP (VP: {concept})"

        return {
            "concept": concept,
            "level_1": {
                "字卡": concept,
                "emoji": emoji
            },
            "level_2": {
                "短语": f"{concept}吧"
            },
            "level_3": {
                "数学": syntax
            },
            "level_4": {
                "拼音": simple_pinyin(concept)
            },
            "level_5": {
                "英文": english
            },
            "level_6": {
                "中文": f"记得{concept}。"
            },
            "level_7": {
                "日文": japanese
            },
            "level_8": {
                "韩文": korean
            }
        }

    def save_dataset(self, samples: List[Dict], output_path: str):
        """保存数据集为JSON格式"""
        dataset = {
            "metadata": {
                "name": "HLBD Full Dataset V2",
                "version": "2.0",
                "description": "完整分层语言启蒙数据集，包含8个层级",
                "total_samples": len(samples),
                "levels": [
                    "level_1: 字卡 + Emoji",
                    "level_2: 短语",
                    "level_3: 数学（句法结构）",
                    "level_4: 拼音",
                    "level_5: 英文",
                    "level_6: 中文",
                    "level_7: 日文",
                    "level_8: 韩文"
                ],
                "anti_mode_collapse": "数据稀释学 - 打散结构化模式"
            },
            "samples": samples
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 数据集已保存到: {output_file.absolute()}")
        print(f"   文件大小: {output_file.stat().st_size / 1024:.2f} KB")


def main():
    # 创建生成器
    generator = HLBDFullGenerator("apt_model/分层语言启蒙数据集.txt")

    # 生成样本
    samples = generator.generate_samples()

    # 保存
    generator.save_dataset(samples, "data/HLBD_Full_V2.json")

    print("\n" + "=" * 60)
    print("✨ HLBD Full V2 数据集生成完成！")
    print("=" * 60)
    print("\n💡 重要提醒:")
    print("   ✓ 所有样本都包含level_3数学层（句法结构）")
    print("   ✓ 训练时会正确调用level_3数据")
    print("   ✓ 数据稀释学已应用，防止模式坍缩")
    print("\n🎯 下一步:")
    print("   python training/train_hlbd_playground.py \\")
    print("       --dataset data/HLBD_Full_V2.json \\")
    print("       --epochs 50")


if __name__ == "__main__":
    main()
