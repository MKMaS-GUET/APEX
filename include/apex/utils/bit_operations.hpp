#ifndef BIT_OPERATION_HPP
#define BIT_OPERATION_HPP

#include "apex/utils/mmap.hpp"

namespace bitop {
#define bit_set(bits, offset) ((bits)[(offset) / 8] |= (1 << (7 - (offset) % 8)))
#define bit_get(bits, offset) (((bits)[(offset) / 8] >> (7 - (offset) % 8)) & 1)

// 内联函数用于更好的性能
inline uint popcount_byte(unsigned char byte) {
    return __builtin_popcount(byte);
}

// 快速跳过零字节
inline uint skip_zero_bytes(MMap<char>* bits, uint start_bit, uint end_bit) {
    uint byte_idx = start_bit / 8;
    uint end_byte = end_bit / 8;

    while (byte_idx <= end_byte && bits->at(byte_idx) == 0) {
        byte_idx++;
    }

    return byte_idx * 8;
}

inline uint skip_zero_bytes(char* bits_in_memory_, uint start_bit, uint end_bit) {
    uint byte_idx = start_bit / 8;
    uint end_byte = end_bit / 8;

    while (byte_idx <= end_byte && bits_in_memory_[byte_idx] == 0) {
        byte_idx++;
    }

    return byte_idx * 8;
}

class One {
    MMap<char>* bits_;
    char* bits_in_memory_;

    uint bit_offset_;
    uint end_;

   public:
    One() = default;

    One(MMap<char>& bits, uint begin, uint end);

    One(char* bits_in_memory_, uint begin, uint end);

    // next one in [begin, end)
    uint Next();
};

// ones in [begin, end)
uint range_rank(MMap<char>& bits, uint begin, uint end);

uint range_rank(char* bits_in_memory_, uint begin, uint end);

uint AccessBitSequence(MMap<uint>& bits, ulong bit_start, uint data_width);

uint AccessBitSequence(uint* bits, ulong bit_start, uint data_width);

}  // namespace bitop

#endif