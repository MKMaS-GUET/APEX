#include "utils/bit_operations.hpp"

#include <iostream>

namespace bitop {

One::One(MMap<char>& bits, uint begin, uint end)
    : bits_(&bits), bits_in_memory_(nullptr), bit_offset_(begin), end_(end) {}
One::One(char* bits_in_memory, uint begin, uint end)
    : bits_(nullptr), bits_in_memory_(bits_in_memory), bit_offset_(begin), end_(end) {}

// next one in [begin, end)
uint One::Next() {
    if (bits_in_memory_ == nullptr) {
        while (bit_offset_ < end_) {
            uint byte_index = bit_offset_ / 8;
            uint bit_in_byte = bit_offset_ % 8;

            // 如果当前字节为0，快速跳过连续的零字节
            if (bits_->at(byte_index) == 0) {
                uint next_non_zero = skip_zero_bytes(bits_, bit_offset_, end_ - 1);
                bit_offset_ = next_non_zero;
                continue;
            }

            // 在非零字节中查找下一个1位
            unsigned char byte_val = bits_->at(byte_index);
            for (uint i = bit_in_byte; i < 8 && (byte_index * 8 + i) < end_; i++) {
                if (byte_val & (1 << (7 - i))) {
                    uint result = byte_index * 8 + i;
                    bit_offset_ = result + 1;
                    return result;
                }
            }

            // 跳到下一个字节的开始
            bit_offset_ = (byte_index + 1) * 8;
        }
    } else {
        while (bit_offset_ < end_) {
            uint byte_index = bit_offset_ / 8;
            uint bit_in_byte = bit_offset_ % 8;

            // 如果当前字节为0，快速跳过连续的零字节
            if (bits_in_memory_[byte_index] == 0) {
                uint next_non_zero = skip_zero_bytes(bits_in_memory_, bit_offset_, end_ - 1);
                bit_offset_ = next_non_zero;
                continue;
            }

            // 在非零字节中查找下一个1位
            unsigned char byte_val = bits_in_memory_[byte_index];
            for (uint i = bit_in_byte; i < 8 && (byte_index * 8 + i) < end_; i++) {
                if (byte_val & (1 << (7 - i))) {
                    uint result = byte_index * 8 + i;
                    bit_offset_ = result + 1;
                    return result;
                }
            }

            // 跳到下一个字节的开始
            bit_offset_ = (byte_index + 1) * 8;
        }
    }

    return end_;
}

// ones in [begin, end]
uint range_rank(MMap<char>& bits, uint begin, uint end) {
    uint cnt = 0;
    uint byte_begin = begin / 8;
    uint byte_end = end / 8;

    if (byte_begin == byte_end) {
        // 同一字节内的情况
        unsigned char byte_val = bits[byte_begin];
        for (uint i = begin % 8; i <= end % 8; i++) {
            if (byte_val & (1 << (7 - i))) {
                cnt++;
            }
        }
    } else {
        // 处理第一个字节的部分位
        unsigned char first_byte = bits[byte_begin];
        for (uint i = begin % 8; i < 8; i++) {
            if (first_byte & (1 << (7 - i))) {
                cnt++;
            }
        }

        // 处理中间的完整字节
        for (uint byte_idx = byte_begin + 1; byte_idx < byte_end; byte_idx++) {
            cnt += __builtin_popcount(static_cast<unsigned char>(bits[byte_idx]));
        }

        // 处理最后一个字节的部分位
        unsigned char last_byte = bits[byte_end];
        for (uint i = 0; i <= end % 8; i++) {
            if (last_byte & (1 << (7 - i))) {
                cnt++;
            }
        }
    }

    return cnt;
}

// ones in [begin, end]
uint range_rank(char* bits_in_memory_, uint begin, uint end) {
    uint cnt = 0;
    uint byte_begin = begin / 8;
    uint byte_end = end / 8;

    if (byte_begin == byte_end) {
        // 同一字节内的情况
        unsigned char byte_val = bits_in_memory_[byte_begin];
        for (uint i = begin % 8; i <= end % 8; i++) {
            if (byte_val & (1 << (7 - i))) {
                cnt++;
            }
        }
    } else {
        // 处理第一个字节的部分位
        unsigned char first_byte = bits_in_memory_[byte_begin];
        for (uint i = begin % 8; i < 8; i++) {
            if (first_byte & (1 << (7 - i))) {
                cnt++;
            }
        }

        // 处理中间的完整字节
        for (uint byte_idx = byte_begin + 1; byte_idx < byte_end; byte_idx++) {
            cnt += __builtin_popcount(static_cast<unsigned char>(bits_in_memory_[byte_idx]));
        }

        // 处理最后一个字节的部分位
        unsigned char last_byte = bits_in_memory_[byte_end];
        for (uint i = 0; i <= end % 8; i++) {
            if (last_byte & (1 << (7 - i))) {
                cnt++;
            }
        }
    }

    return cnt;
}

uint AccessBitSequence(MMap<uint>& bits, ulong bit_start, uint data_width) {
    uint uint_base = bit_start / 32;
    uint offset_in_uint = bit_start % 32;

    uint uint_cnt = (offset_in_uint + data_width + 31) / 32;
    uint data = 0;
    uint remaining_bits = data_width;

    for (uint uint_offset = uint_base; uint_offset < uint_base + uint_cnt; uint_offset++) {
        // 计算可以写入的位数
        uint bits_to_write = std::min(32 - offset_in_uint, remaining_bits);

        // 生成掩码
        uint shift_to_end = 32 - (offset_in_uint + bits_to_write);
        uint mask = ((1ull << bits_to_write) - 1) << shift_to_end;

        // 提取所需位并移位到目标位置
        uint extracted_bits = (bits[uint_offset] & mask) >> shift_to_end;
        data |= extracted_bits << (remaining_bits - bits_to_write);

        remaining_bits -= bits_to_write;
        offset_in_uint = 0;
    }

    return data;
}

uint AccessBitSequence(uint* bits, ulong bit_start, uint data_width) {
    uint uint_base = bit_start / 32;
    uint offset_in_uint = bit_start % 32;

    uint uint_cnt = (offset_in_uint + data_width + 31) / 32;
    uint data = 0;
    uint remaining_bits = data_width;

    for (uint uint_offset = uint_base; uint_offset < uint_base + uint_cnt; uint_offset++) {
        // 计算可以写入的位数
        uint bits_to_write = std::min(32 - offset_in_uint, remaining_bits);

        // 生成掩码
        uint shift_to_end = 32 - (offset_in_uint + bits_to_write);
        uint mask = ((1ull << bits_to_write) - 1) << shift_to_end;

        // 提取所需位并移位到目标位置
        uint extracted_bits = (bits[uint_offset] & mask) >> shift_to_end;
        data |= extracted_bits << (remaining_bits - bits_to_write);

        remaining_bits -= bits_to_write;
        offset_in_uint = 0;
    }

    return data;
}

}  // namespace bitop
