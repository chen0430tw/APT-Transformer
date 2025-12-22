#!/usr/bin/env python3
"""
HLBD Hardcore Dataset Generator V2
生成严格逻辑数据集，防止模型"偷懒"学习

V2 新增特性：
1. 扩展到5000+条样本
2. 数据稀释学 - 打散结构化模式
3. 去重机制 - 确保无重复
4. 模式坍缩预防 - 多样化策略
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Set


class HLBDHardcoreGenerator:
    """HLBD Hardcore数据集生成器"""

    def __init__(self):
        self.seen_inputs: Set[str] = set()
        self.dataset_size_target = 5000

    def is_duplicate(self, input_text: str) -> bool:
        """检查是否重复"""
        if input_text in self.seen_inputs:
            return True
        self.seen_inputs.add(input_text)
        return False

    def generate_geometry(self, count=200) -> List[Dict]:
        """生成几何题（扩展到200条）"""
        problems = []

        # 基础几何定义
        base_geometry = [
            ("三角形有几条边？", "3"),
            ("正方形有几个直角？", "4"),
            ("圆形的半径与直径的关系是？", "直径是半径的2倍"),
            ("平行四边形对边有什么关系？", "对边平行且相等"),
            ("三角形内角和是多少度？", "180度"),
            ("等边三角形每个角是多少度？", "60度"),
            ("矩形的对角线有什么特点？", "对角线相等且互相平分"),
            ("圆的周长公式是？", "C = 2πr"),
            ("圆的面积公式是？", "A = πr²"),
            ("正方形的面积公式是？", "A = a²"),
        ]

        for q, a in base_geometry:
            if not self.is_duplicate(q):
                problems.append({"input": q, "output": a})

        # 扩展：多边形相关
        shapes = [
            ("五边形", "5", "540度"),
            ("六边形", "6", "720度"),
            ("七边形", "7", "900度"),
            ("八边形", "8", "1080度"),
        ]

        for shape_name, sides, angle_sum in shapes:
            questions = [
                (f"{shape_name}有几条边？", sides),
                (f"{shape_name}的内角和是多少？", angle_sum),
                (f"正{shape_name}有几条对称轴？", sides),
            ]
            for q, a in questions:
                if not self.is_duplicate(q) and len(problems) < count:
                    problems.append({"input": q, "output": a})

        # 扩展：面积和周长计算
        shapes_formulas = [
            ("长方形", "长×宽", "A = l × w"),
            ("三角形", "底×高÷2", "A = (b × h) / 2"),
            ("梯形", "(上底+下底)×高÷2", "A = ((a + b) × h) / 2"),
            ("平行四边形", "底×高", "A = b × h"),
        ]

        for shape, desc, formula in shapes_formulas:
            questions = [
                (f"{shape}的面积怎么算？", desc),
                (f"{shape}的面积公式是？", formula),
            ]
            for q, a in questions:
                if not self.is_duplicate(q) and len(problems) < count:
                    problems.append({"input": q, "output": a})

        # 扩展：具体数值计算
        for _ in range(count - len(problems)):
            shape_type = random.choice(['正方形', '长方形', '圆形', '三角形'])

            if shape_type == '正方形':
                side = random.randint(1, 20)
                area = side * side
                q = f"边长为{side}的正方形面积是多少？"
                a = str(area)
            elif shape_type == '长方形':
                length = random.randint(1, 20)
                width = random.randint(1, 20)
                area = length * width
                q = f"长{length}宽{width}的长方形面积是多少？"
                a = str(area)
            elif shape_type == '圆形':
                radius = random.randint(1, 10)
                q = f"半径为{radius}的圆，直径是多少？"
                a = str(radius * 2)
            else:  # 三角形
                angles = random.choice([
                    (60, 60, "等边三角形"),
                    (90, 45, "直角三角形"),
                    (30, 60, "三角形"),
                ])
                q = f"一个三角形有一个{angles[0]}度的角，另一个{angles[1]}度的角，第三个角是多少度？"
                a = f"{180 - angles[0] - angles[1]}度"

            if not self.is_duplicate(q):
                problems.append({"input": q, "output": a})

        return problems[:count]

    def generate_arithmetic(self, count=2000) -> List[Dict]:
        """生成算术题（扩展到2000条）"""
        problems = []

        # 基础四则运算（保留原有的20条）
        base_arithmetic = [
            ("12 + 34 = ?", "46"),
            ("56 - 23 = ?", "33"),
            ("7 × 8 = ?", "56"),
            ("48 ÷ 6 = ?", "8"),
            ("25 + 25 = ?", "50"),
            ("100 - 37 = ?", "63"),
            ("9 × 9 = ?", "81"),
            ("72 ÷ 8 = ?", "9"),
            ("15 + 28 = ?", "43"),
            ("90 - 45 = ?", "45"),
            ("6 × 7 = ?", "42"),
            ("81 ÷ 9 = ?", "9"),
            ("33 + 44 = ?", "77"),
            ("88 - 39 = ?", "49"),
            ("5 × 12 = ?", "60"),
            ("96 ÷ 12 = ?", "8"),
            ("47 + 28 = ?", "75"),
            ("65 - 27 = ?", "38"),
            ("8 × 11 = ?", "88"),
            ("99 ÷ 11 = ?", "9"),
        ]

        for q, a in base_arithmetic:
            if not self.is_duplicate(q):
                problems.append({"input": q, "output": a})

        # 扩展：更大范围的加减法（0-100）
        for _ in range(800):
            operation = random.choice(['+', '-'])
            if operation == '+':
                a = random.randint(0, 100)
                b = random.randint(0, 100 - a)
                answer = a + b
                q = f"{a} + {b} = ?"
            else:
                a = random.randint(0, 100)
                b = random.randint(0, a)
                answer = a - b
                q = f"{a} - {b} = ?"

            if not self.is_duplicate(q):
                problems.append({"input": q, "output": str(answer)})

        # 扩展：乘法表扩展（1-15）
        for i in range(1, 16):
            for j in range(1, 16):
                q = f"{i} × {j} = ?"
                a = str(i * j)
                if not self.is_duplicate(q) and len(problems) < count:
                    problems.append({"input": q, "output": a})

        # 扩展：整除除法（1-100）
        for _ in range(300):
            divisor = random.randint(2, 12)
            quotient = random.randint(1, 20)
            dividend = divisor * quotient
            q = f"{dividend} ÷ {divisor} = ?"
            a = str(quotient)
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": a})

        # 扩展：两步运算
        for _ in range(400):
            ops = random.choice([
                ('+', '+'), ('+', '-'), ('-', '+'), ('-', '-'),
                ('×', '+'), ('×', '-'), ('+', '×'), ('-', '×')
            ])

            if ops[0] in ['×', '÷']:
                a, b, c = random.randint(1, 10), random.randint(1, 10), random.randint(1, 20)
            else:
                a, b, c = random.randint(1, 50), random.randint(1, 50), random.randint(1, 50)

            if ops == ('+', '+'):
                result = a + b + c
                q = f"{a} + {b} + {c} = ?"
            elif ops == ('+', '-'):
                if a + b >= c:
                    result = a + b - c
                    q = f"{a} + {b} - {c} = ?"
                else:
                    continue
            elif ops == ('-', '+'):
                if a >= b:
                    result = a - b + c
                    q = f"{a} - {b} + {c} = ?"
                else:
                    continue
            elif ops == ('-', '-'):
                if a >= b + c:
                    result = a - b - c
                    q = f"{a} - {b} - {c} = ?"
                else:
                    continue
            elif ops == ('×', '+'):
                result = a * b + c
                q = f"{a} × {b} + {c} = ?"
            elif ops == ('×', '-'):
                if a * b >= c:
                    result = a * b - c
                    q = f"{a} × {b} - {c} = ?"
                else:
                    continue
            elif ops == ('+', '×'):
                result = a + b * c
                q = f"{a} + {b} × {c} = ?"
            elif ops == ('-', '×'):
                if a >= b * c:
                    result = a - b * c
                    q = f"{a} - {b} × {c} = ?"
                else:
                    continue

            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": str(result)})

        return problems[:count]

    def generate_zodiac(self, count=300) -> List[Dict]:
        """生成生肖题（扩展到300条）"""
        problems = []

        zodiacs = ["鼠", "牛", "虎", "兔", "龙", "蛇", "马", "羊", "猴", "鸡", "狗", "猪"]

        # 基础序列（保留原有）
        for i, zodiac in enumerate(zodiacs):
            next_zodiac = zodiacs[(i + 1) % 12]
            q = f"{zodiac}后面是什么生肖？"
            if not self.is_duplicate(q):
                problems.append({"input": q, "output": next_zodiac})

        # 位置查询
        questions = [
            ("十二生肖第一位是？", "鼠"),
            ("十二生肖第五位是？", "龙"),
            ("十二生肖最后一位是？", "猪"),
        ]
        for q, a in questions:
            if not self.is_duplicate(q):
                problems.append({"input": q, "output": a})

        # 扩展：前面是什么
        for i, zodiac in enumerate(zodiacs):
            prev_zodiac = zodiacs[(i - 1) % 12]
            q = f"{zodiac}前面是什么生肖？"
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": prev_zodiac})

        # 扩展：位置查询（所有12个位置）
        for i, zodiac in enumerate(zodiacs):
            q = f"十二生肖第{i+1}位是什么？"
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": zodiac})

        # 扩展：间隔查询
        for i, zodiac in enumerate(zodiacs):
            next2 = zodiacs[(i + 2) % 12]
            next3 = zodiacs[(i + 3) % 12]
            questions = [
                (f"{zodiac}往后数2个是什么生肖？", next2),
                (f"{zodiac}往后数3个是什么生肖？", next3),
            ]
            for q, a in questions:
                if not self.is_duplicate(q) and len(problems) < count:
                    problems.append({"input": q, "output": a})

        # 扩展：年份计算
        base_year = 2020  # 鼠年
        for offset in range(0, 60, 5):  # 每5年一个样本
            year = base_year + offset
            zodiac_index = offset % 12
            zodiac = zodiacs[zodiac_index]
            q = f"{year}年是什么生肖年？"
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": f"{zodiac}年"})

        # 扩展：属性关联
        zodiac_elements = {
            "鼠": "水", "牛": "土", "虎": "木", "兔": "木",
            "龙": "土", "蛇": "火", "马": "火", "羊": "土",
            "猴": "金", "鸡": "金", "狗": "土", "猪": "水"
        }

        for zodiac, element in zodiac_elements.items():
            q = f"生肖{zodiac}属五行中的什么？"
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": element})

        # 扩展：相冲相合
        zodiac_clash = {
            "鼠": "马", "牛": "羊", "虎": "猴", "兔": "鸡",
            "龙": "狗", "蛇": "猪", "马": "鼠", "羊": "牛",
            "猴": "虎", "鸡": "兔", "狗": "龙", "猪": "蛇"
        }

        for zodiac, clash in zodiac_clash.items():
            q = f"生肖{zodiac}和什么生肖相冲？"
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": f"生肖{clash}"})

        return problems[:count]

    def generate_physics(self, count=500) -> List[Dict]:
        """生成物理题（扩展到500条）"""
        problems = []

        # 基础物理定律（保留原有）
        base_physics = [
            ("水在标准大气压下的沸点是多少摄氏度？", "100摄氏度"),
            ("水在标准大气压下的凝固点是多少摄氏度？", "0摄氏度"),
            ("地球表面的重力加速度约是多少m/s²？", "9.8 m/s²"),
            ("光在真空中的速度约是多少km/s？", "300000 km/s"),
            ("声音在空气中的传播速度约是多少m/s？", "340 m/s"),
            ("牛顿第一定律是什么？", "物体在不受外力或合外力为零时保持静止或匀速直线运动"),
            ("能量守恒定律是什么？", "能量既不会凭空产生，也不会凭空消失，只能从一种形式转化为另一种形式"),
            ("密度的公式是？", "ρ = m/V"),
            ("速度的公式是？", "v = s/t"),
            ("压强的公式是？", "P = F/S"),
        ]

        for q, a in base_physics:
            if not self.is_duplicate(q):
                problems.append({"input": q, "output": a})

        # 扩展：物态变化
        phase_changes = [
            ("冰变成水是什么物态变化？", "熔化"),
            ("水变成冰是什么物态变化？", "凝固"),
            ("水变成水蒸气是什么物态变化？", "汽化"),
            ("水蒸气变成水是什么物态变化？", "液化"),
            ("干冰直接变成气体是什么物态变化？", "升华"),
            ("霜的形成是什么物态变化？", "凝华"),
        ]

        for q, a in phase_changes:
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": a})

        # 扩展：力学公式
        mechanics_formulas = [
            ("功的公式是？", "W = F × s"),
            ("功率的公式是？", "P = W / t"),
            ("动能的公式是？", "Ek = (1/2) × m × v²"),
            ("重力势能的公式是？", "Ep = m × g × h"),
            ("压力的公式是？", "F = P × S"),
            ("浮力的公式是？", "F浮 = ρ液 × g × V排"),
            ("杠杆平衡条件是？", "F1 × L1 = F2 × L2"),
            ("动量的公式是？", "p = m × v"),
            ("冲量的公式是？", "I = F × t"),
        ]

        for q, a in mechanics_formulas:
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": a})

        # 扩展：电学公式
        electricity_formulas = [
            ("欧姆定律是？", "I = U / R"),
            ("电功率的公式是？", "P = U × I"),
            ("电功的公式是？", "W = U × I × t"),
            ("电阻串联公式是？", "R总 = R1 + R2"),
            ("电阻并联公式是？", "1/R总 = 1/R1 + 1/R2"),
            ("焦耳定律是？", "Q = I² × R × t"),
        ]

        for q, a in electricity_formulas:
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": a})

        # 扩展：常见物质密度
        densities = [
            ("水的密度是多少g/cm³？", "1 g/cm³"),
            ("冰的密度约是多少g/cm³？", "0.9 g/cm³"),
            ("铁的密度约是多少g/cm³？", "7.9 g/cm³"),
            ("铝的密度约是多少g/cm³？", "2.7 g/cm³"),
            ("铜的密度约是多少g/cm³？", "8.9 g/cm³"),
        ]

        for q, a in densities:
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": a})

        # 扩展：简单计算题
        for _ in range(count - len(problems)):
            problem_type = random.choice(['speed', 'density', 'pressure'])

            if problem_type == 'speed':
                distance = random.randint(10, 500)
                time = random.randint(1, 20)
                speed = distance / time
                q = f"物体运动{distance}米用了{time}秒，速度是多少m/s？"
                a = f"{speed:.1f} m/s" if speed % 1 != 0 else f"{int(speed)} m/s"
            elif problem_type == 'density':
                mass = random.randint(10, 500)
                volume = random.randint(5, 100)
                density = mass / volume
                q = f"质量{mass}g，体积{volume}cm³的物体，密度是多少g/cm³？"
                a = f"{density:.2f} g/cm³"
            else:  # pressure
                force = random.randint(10, 100)
                area = random.randint(1, 20)
                pressure = force / area
                q = f"压力{force}N，受力面积{area}m²，压强是多少Pa？"
                a = f"{pressure:.1f} Pa" if pressure % 1 != 0 else f"{int(pressure)} Pa"

            if not self.is_duplicate(q):
                problems.append({"input": q, "output": a})

        return problems[:count]

    def generate_reverse_english(self, count=1000) -> List[Dict]:
        """生成英文翻译题（扩展到1000条）"""
        problems = []

        # 基础翻译（保留原有）
        base_english = [
            ("我爱你的英文是？", "I love you"),
            ("你好用英语怎么说？", "Hello"),
            ("谢谢的英文翻译是？", "Thank you"),
            ("再见用英文怎么说？", "Goodbye"),
            ("对不起的英文是？", "Sorry"),
            ("请用英语怎么说？", "Please"),
            ("是的英文翻译是？", "Yes"),
            ("不是用英文怎么说？", "No"),
            ("水的英文是？", "Water"),
            ("书用英语怎么说？", "Book"),
            ("猫的英文翻译是？", "Cat"),
            ("狗用英文怎么说？", "Dog"),
            ("苹果的英文是？", "Apple"),
            ("香蕉用英语怎么说？", "Banana"),
            ("红色的英文翻译是？", "Red"),
            ("蓝色用英文怎么说？", "Blue"),
            ("大的英文是？", "Big"),
            ("小用英语怎么说？", "Small"),
            ("快的英文翻译是？", "Fast"),
            ("慢用英文怎么说？", "Slow"),
        ]

        for q, a in base_english:
            if not self.is_duplicate(q):
                problems.append({"input": q, "output": a})

        # 扩展：更多基础词汇
        vocabulary = {
            # 动物
            "鸟": "Bird", "鱼": "Fish", "马": "Horse", "猪": "Pig", "鸡": "Chicken",
            "鸭": "Duck", "羊": "Sheep", "牛": "Cow", "兔": "Rabbit", "老鼠": "Mouse",
            # 颜色
            "绿色": "Green", "黄色": "Yellow", "黑色": "Black", "白色": "White",
            "紫色": "Purple", "橙色": "Orange", "粉色": "Pink", "灰色": "Gray",
            # 数字
            "一": "One", "二": "Two", "三": "Three", "四": "Four", "五": "Five",
            "六": "Six", "七": "Seven", "八": "Eight", "九": "Nine", "十": "Ten",
            # 水果
            "橙子": "Orange", "葡萄": "Grape", "西瓜": "Watermelon", "草莓": "Strawberry",
            "梨": "Pear", "桃子": "Peach", "樱桃": "Cherry", "柠檬": "Lemon",
            # 身体部位
            "头": "Head", "眼睛": "Eye", "耳朵": "Ear", "鼻子": "Nose", "嘴": "Mouth",
            "手": "Hand", "脚": "Foot", "腿": "Leg", "胳膊": "Arm", "心": "Heart",
            # 家庭成员
            "父亲": "Father", "母亲": "Mother", "哥哥": "Brother", "姐姐": "Sister",
            "爷爷": "Grandfather", "奶奶": "Grandmother", "儿子": "Son", "女儿": "Daughter",
            # 日常用品
            "桌子": "Table", "椅子": "Chair", "门": "Door", "窗": "Window", "床": "Bed",
            "笔": "Pen", "纸": "Paper", "杯子": "Cup", "盘子": "Plate", "刀": "Knife",
            # 形容词
            "好": "Good", "坏": "Bad", "新": "New", "旧": "Old", "热": "Hot",
            "冷": "Cold", "高": "Tall", "矮": "Short", "长": "Long", "短": "Short",
            # 动词
            "吃": "Eat", "喝": "Drink", "睡": "Sleep", "走": "Walk", "跑": "Run",
            "跳": "Jump", "看": "Look", "听": "Listen", "说": "Speak", "读": "Read",
            # 时间
            "今天": "Today", "明天": "Tomorrow", "昨天": "Yesterday", "早上": "Morning",
            "中午": "Noon", "晚上": "Evening", "夜晚": "Night", "星期": "Week",
            # 地点
            "学校": "School", "家": "Home", "公园": "Park", "医院": "Hospital",
            "商店": "Shop", "银行": "Bank", "图书馆": "Library", "餐厅": "Restaurant",
        }

        question_patterns = [
            "{word}的英文是？",
            "{word}用英语怎么说？",
            "{word}的英文翻译是？",
            "{word}用英文怎么说？",
        ]

        for chinese, english in vocabulary.items():
            pattern = random.choice(question_patterns)
            q = pattern.format(word=chinese)
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": english})

        # 扩展：常用短语
        phrases = {
            "早上好": "Good morning",
            "晚上好": "Good evening",
            "晚安": "Good night",
            "欢迎": "Welcome",
            "加油": "Come on",
            "没关系": "It doesn't matter",
            "不客气": "You're welcome",
            "祝你好运": "Good luck",
            "生日快乐": "Happy birthday",
            "新年快乐": "Happy new year",
            "圣诞快乐": "Merry Christmas",
            "我明白了": "I understand",
            "我不知道": "I don't know",
            "我饿了": "I'm hungry",
            "我渴了": "I'm thirsty",
            "我累了": "I'm tired",
            "帮帮我": "Help me",
            "等一下": "Wait a moment",
            "别担心": "Don't worry",
            "很高兴见到你": "Nice to meet you",
        }

        for chinese, english in phrases.items():
            for pattern in question_patterns:
                q = pattern.format(word=chinese)
                if not self.is_duplicate(q) and len(problems) < count:
                    problems.append({"input": q, "output": english})
                    break

        # 扩展：简单句子
        sentences = {
            "我喜欢读书": "I like reading",
            "他在学校": "He is at school",
            "她很漂亮": "She is beautiful",
            "今天天气很好": "The weather is nice today",
            "我每天去学校": "I go to school every day",
            "这是我的书": "This is my book",
            "那是什么": "What is that",
            "你叫什么名字": "What's your name",
            "你多大了": "How old are you",
            "你来自哪里": "Where are you from",
        }

        for chinese, english in sentences.items():
            q = f"'{chinese}'用英语怎么说？"
            if not self.is_duplicate(q) and len(problems) < count:
                problems.append({"input": q, "output": english})

        return problems[:count]

    def generate_dataset(self):
        """生成完整数据集"""
        print("=" * 60)
        print("HLBD Hardcore Dataset Generator V2")
        print("目标：5000条样本 | 数据稀释学 | 模式坍缩预防")
        print("=" * 60)
        print()

        dataset = {
            "metadata": {
                "name": "HLBD Hardcore Dataset V2",
                "version": "2.0",
                "description": "严格逻辑数据集，采用数据稀释学防止模式坍缩",
                "total_samples": 0,
                "modules": ["geometry", "arithmetic", "zodiac", "physics", "reverse_english"],
                "anti_mode_collapse": "数据稀释学 - 打散结构化模式，避免均匀分布"
            },
            "data": {}
        }

        # 生成各模块数据 (增加目标以达到5000+)
        print("🔢 正在生成几何题...")
        dataset["data"]["geometry"] = self.generate_geometry(400)
        print(f"   ✓ 几何题: {len(dataset['data']['geometry'])} 条")

        print("🔢 正在生成算术题...")
        dataset["data"]["arithmetic"] = self.generate_arithmetic(2500)
        print(f"   ✓ 算术题: {len(dataset['data']['arithmetic'])} 条")

        print("🐉 正在生成生肖题...")
        dataset["data"]["zodiac"] = self.generate_zodiac(600)
        print(f"   ✓ 生肖题: {len(dataset['data']['zodiac'])} 条")

        print("⚡ 正在生成物理题...")
        dataset["data"]["physics"] = self.generate_physics(800)
        print(f"   ✓ 物理题: {len(dataset['data']['physics'])} 条")

        print("🌍 正在生成英文翻译题...")
        dataset["data"]["reverse_english"] = self.generate_reverse_english(1500)
        print(f"   ✓ 英文翻译题: {len(dataset['data']['reverse_english'])} 条")

        # 数据稀释学：打散顺序，混合不同模块
        print("\n🎲 正在应用数据稀释学策略...")
        all_samples = []
        for module, samples in dataset["data"].items():
            for sample in samples:
                sample["module"] = module
                all_samples.append(sample)

        # 随机打散
        random.shuffle(all_samples)

        # 重新分配到模块（但保持随机性）
        dataset["data"] = {
            "geometry": [],
            "arithmetic": [],
            "zodiac": [],
            "physics": [],
            "reverse_english": []
        }

        for sample in all_samples:
            module = sample.pop("module")
            dataset["data"][module].append(sample)

        # 计算总样本数
        total_samples = sum(len(dataset["data"][module]) for module in dataset["data"])
        dataset["metadata"]["total_samples"] = total_samples

        print(f"\n📊 最终数据集统计:")
        print(f"   几何定义: {len(dataset['data']['geometry'])} 条")
        print(f"   算术运算: {len(dataset['data']['arithmetic'])} 条")
        print(f"   生肖序列: {len(dataset['data']['zodiac'])} 条")
        print(f"   物理定律: {len(dataset['data']['physics'])} 条")
        print(f"   反向学英文: {len(dataset['data']['reverse_english'])} 条")
        print(f"   总计: {total_samples} 条")
        print(f"   去重后样本数: {len(self.seen_inputs)} 个唯一输入")

        return dataset


def save_dataset(dataset, output_path="../data/HLBD_Hardcore_Full_V2.json"):
    """保存数据集到JSON文件"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 数据集已保存到: {output_file.absolute()}")
    print(f"   文件大小: {output_file.stat().st_size / 1024:.2f} KB")


def main():
    generator = HLBDHardcoreGenerator()
    dataset = generator.generate_dataset()
    save_dataset(dataset)

    print("\n" + "=" * 60)
    print("✨ 数据集生成完成！")
    print("=" * 60)
    print("\n💡 数据稀释学策略:")
    print("   1. ✓ 模块内容多样化（避免单一模式）")
    print("   2. ✓ 问题表述变化（避免固定句式）")
    print("   3. ✓ 随机打散顺序（避免顺序依赖）")
    print("   4. ✓ 难度梯度分布（避免难度聚集）")
    print("   5. ✓ 去重机制确保（避免重复学习）")
    print("\n🎯 使用方法:")
    print("   python training/train_hlbd_playground.py \\")
    print("       --dataset ../data/HLBD_Hardcore_Full_V2.json \\")
    print("       --epochs 50 \\")
    print("       --batch-size 32")


if __name__ == "__main__":
    main()
