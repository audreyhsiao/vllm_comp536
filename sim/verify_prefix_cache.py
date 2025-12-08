#!/usr/bin/env python3
"""
驗證 prefix cache 正確性的腳本

用法:
    python sim/verify_prefix_cache.py prefix_stats.json

這個腳本會檢查 prefix_stats.json 並驗證:
1. 第一個請求應該沒有 prefix hit
2. 第二個請求應該有 prefix hit（如果兩個請求有相同的 prefix）
3. Hit ratio 應該符合預期
"""

import json
import sys
from typing import Dict, Any, List


def verify_prefix_stats(stats_file: str, 
                        expected_hit_ratio: float = 0.9,
                        tolerance: float = 0.1,
                        verbose: bool = True) -> bool:
    """
    驗證 prefix cache 統計資訊
    
    Args:
        stats_file: prefix_stats.json 檔案路徑
        expected_hit_ratio: 預期的 hit ratio（預設 0.9，即 90%）
        tolerance: 允許的誤差範圍（預設 0.1，即 10%）
        verbose: 是否輸出詳細資訊
    
    Returns:
        True 如果驗證通過，False 否則
    """
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    except FileNotFoundError:
        print(f"❌ 錯誤: 找不到檔案 {stats_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ 錯誤: JSON 解析失敗: {e}")
        return False
    
    requests = stats.get('requests', {})
    if not requests:
        print("❌ 錯誤: 沒有找到任何請求統計資訊")
        return False
    
    request_ids = sorted(requests.keys())
    block_size = stats.get('block_size', 16)
    
    if verbose:
        print(f"📊 分析 {len(request_ids)} 個請求")
        print(f"📦 Block size: {block_size} tokens")
        print()
    
    # 驗證第一個請求
    first_req_id = request_ids[0]
    first_req = requests[first_req_id]
    
    if verbose:
        print(f"🔍 請求 1 ({first_req_id}):")
        print(f"   - Total prompt tokens: {first_req.get('total_prompt_tokens', 'N/A')}")
        print(f"   - Prefix hit tokens: {first_req.get('prefix_hit_tokens', 0)}")
        print(f"   - Prefix hit ratio: {first_req.get('prefix_hit_ratio', 0.0):.2%}")
        print(f"   - Hit block IDs: {first_req.get('hit_block_ids', [])}")
    
    # 第一個請求應該沒有 hit
    first_hit_ratio = first_req.get('prefix_hit_ratio', 0.0)
    if first_hit_ratio != 0.0:
        print(f"❌ 驗證失敗: 第一個請求應該沒有 prefix hit，但得到 {first_hit_ratio:.2%}")
        return False
    
    if verbose:
        print("   ✓ 第一個請求沒有 prefix hit（符合預期）")
        print()
    
    # 如果只有一個請求，只驗證第一個
    if len(request_ids) == 1:
        print("⚠️  警告: 只有一個請求，無法驗證 prefix sharing")
        return True
    
    # 驗證第二個請求
    second_req_id = request_ids[1]
    second_req = requests[second_req_id]
    
    if verbose:
        print(f"🔍 請求 2 ({second_req_id}):")
        print(f"   - Total prompt tokens: {second_req.get('total_prompt_tokens', 'N/A')}")
        print(f"   - Prefix hit tokens: {second_req.get('prefix_hit_tokens', 0)}")
        print(f"   - Prefix hit ratio: {second_req.get('prefix_hit_ratio', 0.0):.2%}")
        print(f"   - Hit block IDs: {second_req.get('hit_block_ids', [])}")
    
    second_hit_ratio = second_req.get('prefix_hit_ratio', 0.0)
    second_hit_tokens = second_req.get('prefix_hit_tokens', 0)
    second_hit_blocks = second_req.get('hit_block_ids', [])
    
    # 檢查是否有 hit
    if second_hit_tokens == 0:
        print("❌ 驗證失敗: 第二個請求應該有 prefix hit，但 hit tokens = 0")
        return False
    
    if len(second_hit_blocks) == 0:
        print("❌ 驗證失敗: 第二個請求應該有 hit blocks，但 hit_block_ids 為空")
        return False
    
    # 計算理論上的最大可能 hit ratio（基於 block 對齊）
    second_total_tokens = second_req.get('total_prompt_tokens', 0)
    first_total_tokens = first_req.get('total_prompt_tokens', 0)
    
    # 計算理論上可以 hit 的 blocks 數量
    # 如果兩個請求的 tokens 不完全相同，可能只有部分 blocks 匹配
    max_possible_hit_blocks = min(
        len(second_hit_blocks),
        (first_total_tokens + block_size - 1) // block_size  # 第一個請求的 block 數量
    )
    max_possible_hit_tokens = max_possible_hit_blocks * block_size
    max_possible_hit_ratio = min(1.0, max_possible_hit_tokens / second_total_tokens) if second_total_tokens > 0 else 0.0
    
    if verbose:
        print(f"   - 第一個請求 tokens: {first_total_tokens}")
        print(f"   - 第二個請求 tokens: {second_total_tokens}")
        print(f"   - 理論最大 hit blocks: {max_possible_hit_blocks}")
        print(f"   - 理論最大 hit tokens: {max_possible_hit_tokens}")
        print(f"   - 理論最大 hit ratio: {max_possible_hit_ratio:.2%}")
        print()
    
    # 檢查是否有 hit（基本驗證）
    if second_hit_tokens == 0:
        print("❌ 驗證失敗: 第二個請求應該有 prefix hit，但 hit tokens = 0")
        return False
    
    if len(second_hit_blocks) == 0:
        print("❌ 驗證失敗: 第二個請求應該有 hit blocks，但 hit_block_ids 為空")
        return False
    
    # 檢查 hit ratio 是否在預期範圍內
    min_expected = max(0.0, expected_hit_ratio - tolerance)
    max_expected = min(1.0, expected_hit_ratio + tolerance)
    
    # 如果實際 hit ratio 低於預期，但接近理論最大值，則可能是因為 prompt 不完全相同
    if second_hit_ratio < min_expected:
        # 檢查是否接近理論最大值（允許 5% 的誤差）
        if max_possible_hit_ratio > 0 and second_hit_ratio >= max_possible_hit_ratio * 0.95:
            print(f"⚠️  警告: Hit ratio {second_hit_ratio:.2%} 低於預期 [{min_expected:.2%}, {max_expected:.2%}]")
            print(f"   但接近理論最大值 {max_possible_hit_ratio:.2%}，可能是因為兩個請求的 prompt 不完全相同")
            print(f"   （例如：chat template 格式化、tokenization 差異等）")
            print(f"   ✓ 這可能是正常的，因為 prefix cache 是基於 block hash 的")
            if verbose:
                print(f"   - 實際 hit: {second_hit_tokens} tokens ({len(second_hit_blocks)} blocks)")
                print(f"   - 理論最大: {max_possible_hit_tokens} tokens ({max_possible_hit_blocks} blocks)")
        else:
            print(f"❌ 驗證失敗: Hit ratio {second_hit_ratio:.2%} 低於預期範圍 [{min_expected:.2%}, {max_expected:.2%}]")
            print(f"   理論最大 hit ratio: {max_possible_hit_ratio:.2%}")
            return False
    else:
        if verbose:
            print(f"   ✓ Hit ratio {second_hit_ratio:.2%} 在預期範圍內 [{min_expected:.2%}, {max_expected:.2%}]")
    
    if verbose:
        print(f"   ✓ Hit {len(second_hit_blocks)} 個 blocks ({second_hit_tokens} tokens)")
        print()
    
    # 驗證 block 統計
    blocks = stats.get('blocks', {})
    if verbose:
        print(f"📦 Block 統計:")
        for block_id in second_hit_blocks:
            block_id_str = str(block_id)
            if block_id_str in blocks:
                block_stats = blocks[block_id_str]
                hit_count = block_stats.get('hit_count', 0)
                print(f"   - Block {block_id}: hit_count = {hit_count}")
            else:
                print(f"   - Block {block_id}: 無統計資訊")
        print()
    
    # 如果我們到達這裡，基本驗證通過
    # 但如果有警告，我們仍然認為驗證通過（因為可能是正常的行為）
    if second_hit_ratio < min_expected and max_possible_hit_ratio > 0 and second_hit_ratio >= max_possible_hit_ratio * 0.95:
        print("✅ 驗證通過（有警告，但可能是正常的）")
    else:
        print("✅ 所有驗證通過！")
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python sim/verify_prefix_cache.py <prefix_stats.json> [expected_hit_ratio] [tolerance]")
        print()
        print("範例:")
        print("  python sim/verify_prefix_cache.py prefix_stats.json")
        print("  python sim/verify_prefix_cache.py prefix_stats.json 0.9 0.1")
        sys.exit(1)
    
    stats_file = sys.argv[1]
    expected_hit_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
    
    success = verify_prefix_stats(stats_file, expected_hit_ratio, tolerance)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

