#!/usr/bin/env python3
"""
verl-agent 数据准备脚本

此脚本用于创建 verl-agent 训练所需的 parquet 数据文件。
verl-agent 使用数据文件来指示模态（text 或 visual）和数据大小。
"""

import os
import argparse
import pandas as pd

def create_agent_data(output_dir, train_size=16, val_size=128):
    """
    创建 agent 训练数据文件

    Args:
        output_dir: 输出目录
        train_size: 训练数据大小
        val_size: 验证数据大小
    """
    os.makedirs(output_dir, exist_ok=True)

    # 创建训练数据
    train_data = []
    for i in range(train_size):
        train_data.append({
            "data_source": "text",
            "prompt": [{
                "role": "user",
                "content": "",  # 空字符串表示纯文本任务
            }],
            "ability": "agent",
            "extra_info": {
                'split': 'train',
                'index': i,
            }
        })

    # 创建验证数据
    val_data = []
    for i in range(val_size):
        val_data.append({
            "data_source": "text",
            "prompt": [{
                "role": "user",
                "content": "",  # 空字符串表示纯文本任务
            }],
            "ability": "agent",
            "extra_info": {
                'split': 'test',
                'index': i,
            }
        })

    # 保存为 parquet 文件
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_file = os.path.join(output_dir, 'train.parquet')
    val_file = os.path.join(output_dir, 'test.parquet')

    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)

    print(f"✅ 训练数据已保存到: {train_file} ({len(train_data)} 条)")
    print(f"✅ 验证数据已保存到: {val_file} ({len(val_data)} 条)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='创建 verl-agent 训练数据')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--train_size', type=int, default=16, help='训练数据大小')
    parser.add_argument('--val_size', type=int, default=128, help='验证数据大小')

    args = parser.parse_args()

    create_agent_data(args.output_dir, args.train_size, args.val_size)
