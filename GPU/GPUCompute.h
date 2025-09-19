//  GPU/GPUCompute.h

/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
// 8891689_MOD: 确保包含了 GPUEngine.h 以获取 ITEM_SIZE32 等定义
#include "GPUEngine.h" 

// CUDA Kernel main function
// We use affine coordinates for elliptic curve point (ie Z=1)

// 8891689_MOD: 定义新的搜索模式標誌，供內核使用
#define SEARCH_MODE_ADDRESS 0
#define SEARCH_MODE_PUBKEY  1

// 8891689_MOD: 為公鑰搜索模式定義常量內存
__device__ __constant__ uint64_t d_targetPubKeyX[4];
__device__ __constant__ uint32_t d_searchInfo; 

// 8891689_MOD: 新的公鑰匹配函數 ,https://github.com/8891689
__device__ __noinline__ void CheckPointPubKey(uint64_t *px, uint8_t py_is_odd, int32_t incr, uint32_t *out) {
    if (px[0] == d_targetPubKeyX[0] && px[1] == d_targetPubKeyX[1] &&
        px[2] == d_targetPubKeyX[2] && px[3] == d_targetPubKeyX[3]) {
        
        uint32_t target_y_is_odd = d_searchInfo & 2; // bit 1
        if ((py_is_odd << 1) == target_y_is_odd) {
            uint32_t pos = atomicAdd(out, 1);
            // 输出缓冲区的大小由主机端保证，这里只需写入即可
            
            uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            uint32_t* item_ptr = out + (pos * ITEM_SIZE32 + 1);
            
            item_ptr[0] = tid;
            item_ptr[1] = (uint32_t)(incr << 16) | (uint32_t)(1 << 15); // 标记为压缩模式
            
            // 8891689_FIX: 完整地写回 256-bit 的 X 坐标 (8 * 32-bit)
            // 虽然 ITEM_SIZE 限制了我们，但我们尽量多写一些以供调试
            // ITEM_SIZE32 是 7，所以 item_ptr[0]..[6] 可用。
            // item_ptr[0] 是 tid, item_ptr[1] 是 incr, 所以还剩 5 个 u32
            uint32_t* px_32 = (uint32_t*)px;
            item_ptr[2] = px_32[0]; // LSB of X
            item_ptr[3] = px_32[1];
            item_ptr[4] = px_32[2];
            item_ptr[5] = px_32[3];
            item_ptr[6] = px_32[4]; // 20 字节，与 hash160 长度相同
        }
    }
}

// 原始 CheckPoint 函數，已移除 maxFound 检查
__device__ __noinline__ void CheckPoint(uint32_t *_h, int32_t incr, address_t *address, uint32_t *lookup32, uint32_t *out) {

  uint32_t   off;
  addressl_t  l32;
  address_t   pr0;
  address_t   hit;
  uint32_t   st;
  uint32_t   ed;
  uint32_t   mi;
  uint32_t   lmi;
  
    pr0 = *(address_t *)(_h);
    hit = address[pr0];

    if (hit) {
        if (lookup32) {
            off = lookup32[pr0];
            l32 = _h[0];
            st = off;
            ed = off + hit - 1;
            while (st <= ed) {
                mi = (st + ed) / 2;
                lmi = lookup32[mi];
                if (l32 < lmi) {
                    ed = mi - 1;
                }
                else if (l32 == lmi) {
                    goto addItem; // 找到匹配，跳轉
                }
                else {
                    st = mi + 1;
                }
            }
            return; // 在查找表中未找到，返回
        }

    // 如果不使用 lookup32 或找到了匹配項，則執行寫入操作
    addItem:
        uint32_t pos = atomicAdd(out, 1);
        uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
        uint32_t* item_ptr = out + (pos * ITEM_SIZE32 + 1);
        
        item_ptr[0] = tid;
        item_ptr[1] = (uint32_t)(incr << 16) | (uint32_t)(1 << 15); // 總是壓縮模式
        item_ptr[2] = _h[0];
        item_ptr[3] = _h[1];
        item_ptr[4] = _h[2];
        item_ptr[5] = _h[3];
        item_ptr[6] = _h[4];
    }
}
