#ifndef DAAS_HPP
#define DAAS_HPP

#include <memory>
#include <span>
#include <vector>

#include "index/characteristic_set.hpp"
#include "index/predicate_index.hpp"
#include "utils/mmap.hpp"

class DAAs {
   public:
    struct Structure {
        uint data_cnt;
        uint* levels;
        char* level_end;
        char* array_end;

        void create(std::vector<std::vector<uint>>& arrays);

       public:
        Structure(std::vector<std::vector<uint>>& arrays);
        ~Structure();
    };

   private:
    bool in_memory_;

    std::string file_path_;

    std::vector<ulong> daa_offsets_;

    uint daa_levels_width_;
    MMap<uint> daa_levels_;
    MMap<char> daa_level_end_;
    MMap<char> daa_array_end_;

    uint* daa_levels_in_memory_;
    char* daa_level_end_in_memory_;
    char* daa_array_end_in_memory_;

    void Preprocess(std::vector<std::vector<std::vector<uint>>>& entity_set);

    void BuildDAAs(std::vector<std::vector<std::vector<uint>>>& entity_set);

   public:
    DAAs();
    DAAs(std::string file_path);
    DAAs(std::string file_path, uint daa_levels_width, bool in_memory);

    void Build(std::vector<std::vector<std::vector<uint>>>& entity_set);

    std::vector<ulong>& daa_offsets();

    void Load();

    uint AccessLevels(ulong offset);

    std::vector<uint>* AccessDAA(uint daa_offset, uint daa_size, std::span<uint>& offset2id, uint index);

    std::vector<uint>* AccessDAAAllArrays(uint daa_offset, uint daa_size, std::vector<std::span<uint>>& offset2id);

    uint daa_levels_width();

    void Close();
};

#endif