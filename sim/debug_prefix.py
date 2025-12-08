#!/usr/bin/env python3
"""
调试 prefix key 和 prompt 的脚本
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from sim.client_simulator import ShareGPTLoader, ChatTemplateFormatter

# 加载测试文件
conversations = ShareGPTLoader.load_from_file("sim/test_prefix_sharing_simple.json")

# 创建 formatter
formatter = ChatTemplateFormatter(
    model_name="meta-llama/Llama-3.2-1B-Instruct"
)

print("=" * 80)
print("调试 Prefix Key 和 Prompt")
print("=" * 80)

for i, conv in enumerate(conversations, 1):
    print(f"\n对话 {i} (ID: {conv.conversation_id}):")
    print(f"Messages: {conv.messages}")
    
    # 获取第一个 user message
    first_user_msg = next((m for m in conv.messages if m.get("role") == "user"), None)
    if first_user_msg:
        print(f"\n第一个 user message: {first_user_msg['content']}")
        
        # 计算 prefix key
        prefix_key = formatter.get_prefix_key(conv.messages, use_full_conversation=False)
        print(f"Prefix key: {prefix_key}")
        
        # 格式化 prompt
        context_messages = conv.messages[:1]  # 只取第一个 user message
        prompt = formatter.format_conversation(context_messages, add_generation_prompt=True)
        print(f"\nFormatted prompt (前200字符):")
        print(prompt[:200])
        print("...")
        
        # Tokenize
        if formatter.tokenizer:
            token_ids = formatter.tokenizer.encode(prompt, add_special_tokens=False)
            print(f"\nToken count: {len(token_ids)}")
            print(f"First 20 tokens: {token_ids[:20]}")
            
            # 检查两个对话的 prefix key 是否相同
            if i == 1:
                conv1_prefix_key = prefix_key
                conv1_tokens = len(token_ids)
            elif i == 2:
                conv2_prefix_key = prefix_key
                conv2_tokens = len(token_ids)
                print(f"\n{'='*80}")
                print("比较结果:")
                print(f"  Conv 1 prefix key: {conv1_prefix_key}")
                print(f"  Conv 2 prefix key: {conv2_prefix_key}")
                print(f"  Prefix keys 相同: {conv1_prefix_key == conv2_prefix_key}")
                print(f"  Conv 1 tokens: {conv1_tokens}")
                print(f"  Conv 2 tokens: {conv2_tokens}")
                print(f"  Token counts 相同: {conv1_tokens == conv2_tokens}")
                
                # 检查 token IDs 是否相同
                conv1_token_ids = formatter.tokenizer.encode(
                    formatter.format_conversation(conversations[0].messages[:1], add_generation_prompt=True),
                    add_special_tokens=False
                )
                conv2_token_ids = formatter.tokenizer.encode(
                    formatter.format_conversation(conversations[1].messages[:1], add_generation_prompt=True),
                    add_special_tokens=False
                )
                
                # 找出相同的 tokens
                min_len = min(len(conv1_token_ids), len(conv2_token_ids))
                matching_tokens = 0
                for j in range(min_len):
                    if conv1_token_ids[j] == conv2_token_ids[j]:
                        matching_tokens += 1
                    else:
                        break
                
                print(f"  匹配的 tokens: {matching_tokens}/{min_len}")
                print(f"  匹配比例: {matching_tokens/min_len*100:.2f}%")
                
                # Block 计算
                block_size = 16
                conv1_blocks = (conv1_tokens + block_size - 1) // block_size
                conv2_blocks = (conv2_tokens + block_size - 1) // block_size
                matching_blocks = matching_tokens // block_size
                
                print(f"\n  Block 分析 (block_size={block_size}):")
                print(f"    Conv 1 blocks: {conv1_blocks}")
                print(f"    Conv 2 blocks: {conv2_blocks}")
                print(f"    匹配的 blocks: {matching_blocks}")
                print(f"    预期 hit tokens: {matching_blocks * block_size}")
                print(f"    预期 hit ratio: {matching_blocks * block_size / conv2_tokens * 100:.2f}%")

print("\n" + "=" * 80)

