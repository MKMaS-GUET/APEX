

#ifndef RESULT_MAP_HPP
#define RESULT_MAP_HPP

#include <parallel_hashmap/phmap.h>
#include <sys/types.h>

#include <span>
#include <vector>

struct VectorHash {
    std::size_t operator()(const std::vector<uint>& vec) const {
        constexpr std::size_t FNV_offset_basis = 14695981039346656037ULL;
        constexpr std::size_t FNV_prime = 1099511628211ULL;

        std::size_t hash = FNV_offset_basis;
        const char* data = reinterpret_cast<const char*>(vec.data());
        std::size_t size = vec.size() * sizeof(uint);

        for (std::size_t i = 0; i < size; ++i) {
            hash ^= static_cast<unsigned char>(data[i]);
            hash *= FNV_prime;
        }
        return hash;
    }
};

using ResultMap = phmap::flat_hash_map<std::vector<uint>, std::span<uint>, VectorHash>;

#endif