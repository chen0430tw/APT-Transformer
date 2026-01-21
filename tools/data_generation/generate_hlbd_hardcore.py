#!/usr/bin/env python3
"""
HLBD Hardcore Dataset Generator
生成严格逻辑数据集，防止模型"偷懒"学习

数据集特点：
1. 几何定义 - 数学性质唯一
2. 100以内四则运算 - 答案唯一
3. 十二生肖序列 - 绝对顺序
4. 物理定律 - 因果关系明确
"""

import json
import random
from pathlib import Path


def generate_math_problems(count=500):
    """
    自动生成500道数学题（加减法，0-50范围）

    格式：
    - 加法: "15 + 23 = ?" → "38"
    - 减法: "45 - 18 = ?" → "27"
    """
    problems = []

    for _ in range(count):
        # 随机选择加法或减法
        operation = random.choice(['+', '-'])

        if operation == '+':
            # 加法：确保结果在100以内
            a = random.randint(0, 50)
            b = random.randint(0, 50 - a)
            answer = a + b
            question = f"{a} + {b} = ?"

        else:  # 减法
            # 减法：确保结果非负
            a = random.randint(0, 50)
            b = random.randint(0, a)
            answer = a - b
            question = f"{a} - {b} = ?"

        problems.append({
            "input": question,
            "output": str(answer)
        })

    return problems


def create_hardcore_dataset():
    """创建HLBD Hardcore完整数据集"""

    dataset = {
        "metadata": {
            "name": "HLBD Hardcore Dataset",
            "version": "1.0",
            "description": "严格逻辑数据集，用于防止模型shortcut learning",
            "total_samples": 0,
            "modules": ["geometry", "arithmetic", "zodiac", "physics"]
        },
        "data": {
            "geometry": [
                # 几何定义 - 数学性质唯一
                {
                    "input": "三角形有几条边？",
                    "output": "3"
                },
                {
                    "input": "正方形有几个直角？",
                    "output": "4"
                },
                {
                    "input": "圆形的半径与直径的关系是？",
                    "output": "直径是半径的2倍"
                },
                {
                    "input": "平行四边形对边有什么关系？",
                    "output": "对边平行且相等"
                },
                {
                    "input": "三角形内角和是多少度？",
                    "output": "180度"
                },
                {
                    "input": "等边三角形每个角是多少度？",
                    "output": "60度"
                },
                {
                    "input": "矩形的对角线有什么特点？",
                    "output": "对角线相等且互相平分"
                },
                {
                    "input": "圆的周长公式是？",
                    "output": "C = 2πr"
                },
                {
                    "input": "圆的面积公式是？",
                    "output": "A = πr²"
                },
                {
                    "input": "正方形的面积公式是？",
                    "output": "A = a²"
                }
            ],

            "arithmetic": [
                # 100以内四则运算 - 答案唯一
                {"input": "12 + 34 = ?", "output": "46"},
                {"input": "56 - 23 = ?", "output": "33"},
                {"input": "7 × 8 = ?", "output": "56"},
                {"input": "48 ÷ 6 = ?", "output": "8"},
                {"input": "25 + 25 = ?", "output": "50"},
                {"input": "100 - 37 = ?", "output": "63"},
                {"input": "9 × 9 = ?", "output": "81"},
                {"input": "72 ÷ 8 = ?", "output": "9"},
                {"input": "15 + 28 = ?", "output": "43"},
                {"input": "90 - 45 = ?", "output": "45"},
                {"input": "6 × 7 = ?", "output": "42"},
                {"input": "81 ÷ 9 = ?", "output": "9"},
                {"input": "33 + 44 = ?", "output": "77"},
                {"input": "88 - 39 = ?", "output": "49"},
                {"input": "5 × 12 = ?", "output": "60"},
                {"input": "96 ÷ 12 = ?", "output": "8"},
                {"input": "47 + 28 = ?", "output": "75"},
                {"input": "65 - 27 = ?", "output": "38"},
                {"input": "8 × 11 = ?", "output": "88"},
                {"input": "99 ÷ 11 = ?", "output": "9"},
            ],

            "zodiac": [
                # 十二生肖序列 - 绝对顺序
                {"input": "鼠后面是什么生肖？", "output": "牛"},
                {"input": "牛后面是什么生肖？", "output": "虎"},
                {"input": "虎后面是什么生肖？", "output": "兔"},
                {"input": "兔后面是什么生肖？", "output": "龙"},
                {"input": "龙后面是什么生肖？", "output": "蛇"},
                {"input": "蛇后面是什么生肖？", "output": "马"},
                {"input": "马后面是什么生肖？", "output": "羊"},
                {"input": "羊后面是什么生肖？", "output": "猴"},
                {"input": "猴后面是什么生肖？", "output": "鸡"},
                {"input": "鸡后面是什么生肖？", "output": "狗"},
                {"input": "狗后面是什么生肖？", "output": "猪"},
                {"input": "猪后面是什么生肖？", "output": "鼠"},
                {"input": "十二生肖第一位是？", "output": "鼠"},
                {"input": "十二生肖第五位是？", "output": "龙"},
                {"input": "十二生肖最后一位是？", "output": "猪"},
            ],

            "physics": [
                # 物理定律 - 因果关系明确
                {
                    "input": "水在标准大气压下的沸点是多少摄氏度？",
                    "output": "100摄氏度"
                },
                {
                    "input": "水在标准大气压下的凝固点是多少摄氏度？",
                    "output": "0摄氏度"
                },
                {
                    "input": "地球表面的重力加速度约是多少m/s²？",
                    "output": "9.8 m/s²"
                },
                {
                    "input": "光在真空中的速度约是多少km/s？",
                    "output": "300000 km/s"
                },
                {
                    "input": "声音在空气中的传播速度约是多少m/s？",
                    "output": "340 m/s"
                },
                {
                    "input": "牛顿第一定律是什么？",
                    "output": "物体在不受外力或合外力为零时保持静止或匀速直线运动"
                },
                {
                    "input": "能量守恒定律是什么？",
                    "output": "能量既不会凭空产生，也不会凭空消失，只能从一种形式转化为另一种形式"
                },
                {
                    "input": "密度的公式是？",
                    "output": "ρ = m/V"
                },
                {
                    "input": "速度的公式是？",
                    "output": "v = s/t"
                },
                {
                    "input": "压强的公式是？",
                    "output": "P = F/S"
                },
            ],

            "reverse_english": [
                # 反向学英文 - 中文→英文翻译
                {"input": "我爱你的英文是？", "output": "I love you"},
                {"input": "你好用英语怎么说？", "output": "Hello"},
                {"input": "谢谢的英文翻译是？", "output": "Thank you"},
                {"input": "再见用英文怎么说？", "output": "Goodbye"},
                {"input": "对不起的英文是？", "output": "Sorry"},
                {"input": "请用英语怎么说？", "output": "Please"},
                {"input": "是的英文翻译是？", "output": "Yes"},
                {"input": "不是用英文怎么说？", "output": "No"},
                {"input": "水的英文是？", "output": "Water"},
                {"input": "书用英语怎么说？", "output": "Book"},
                {"input": "猫的英文翻译是？", "output": "Cat"},
                {"input": "狗用英文怎么说？", "output": "Dog"},
                {"input": "苹果的英文是？", "output": "Apple"},
                {"input": "香蕉用英语怎么说？", "output": "Banana"},
                {"input": "红色的英文翻译是？", "output": "Red"},
                {"input": "蓝色用英文怎么说？", "output": "Blue"},
                {"input": "大的英文是？", "output": "Big"},
                {"input": "小用英语怎么说？", "output": "Small"},
                {"input": "快的英文翻译是？", "output": "Fast"},
                {"input": "慢用英文怎么说？", "output": "Slow"},
            ]
        }
    }

    # 生成500道随机数学题
    print("🔢 正在生成500道随机数学题...")
    auto_generated_math = generate_math_problems(500)
    dataset["data"]["arithmetic"].extend(auto_generated_math)

    # 计算总样本数
    total_samples = sum(len(dataset["data"][module]) for module in dataset["data"])
    dataset["metadata"]["total_samples"] = total_samples

    # 统计信息
    print(f"\n📊 数据集统计:")
    print(f"   几何定义: {len(dataset['data']['geometry'])} 条")
    print(f"   算术运算: {len(dataset['data']['arithmetic'])} 条 (含{len(auto_generated_math)}道自动生成)")
    print(f"   生肖序列: {len(dataset['data']['zodiac'])} 条")
    print(f"   物理定律: {len(dataset['data']['physics'])} 条")
    print(f"   反向学英文: {len(dataset['data']['reverse_english'])} 条")
    print(f"   总计: {total_samples} 条")

    return dataset


def save_dataset(dataset, output_path="../data/HLBD_Hardcore_Full.json"):
    """保存数据集到JSON文件"""
    output_file = Path(output_path)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 数据集已保存到: {output_file.absolute()}")
    print(f"   文件大小: {output_file.stat().st_size / 1024:.2f} KB")


def main():
    print("=" * 60)
    print("HLBD Hardcore Dataset Generator")
    print("严格逻辑数据集生成器 - 防止模型「偷懒」")
    print("=" * 60)
    print()

    # 创建数据集
    dataset = create_hardcore_dataset()

    # 保存数据集
    save_dataset(dataset)

    print("\n" + "=" * 60)
    print("✨ 数据集生成完成！")
    print("=" * 60)
    print("\n💡 使用方法:")
    print("   在训练脚本中加载此数据集进行HLBD微调")
    print("   目标：防止模型输出通用答案（如'在公园中，享受自然风光。'）")
    print("   强制模型学习严格的输入-输出映射关系")


if __name__ == "__main__":
    main()
