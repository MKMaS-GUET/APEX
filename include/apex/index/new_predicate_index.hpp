#ifndef NEW_PREDICATE_INDEX_HPP
#define NEW_PREDICATE_INDEX_HPP

#include <parallel_hashmap/btree.h>
#include <parallel_hashmap/phmap.h>
#include <condition_variable>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "apex/utils/bitset.hpp"

class NewPredicateIndex {
   public:
    struct Index {
        uint min_subject;
        uint min_object;

        Bitset subject_set_bitmap;
        Bitset object_set_bitmap;

        phmap::flat_hash_map<uint, uint> sid2offset;
        phmap::flat_hash_map<uint, uint> oid2offset;

        void Build(std::vector<std::pair<uint, uint>>& so_pairs);

        void Clear();
    };

    std::vector<Index> index_;

   private:
    std::string file_path_;
    std::shared_ptr<phmap::flat_hash_map<uint, std::vector<std::pair<uint, uint>>>> pso_;

    uint max_predicate_id_;
    MMap<uint> predicate_index_;
    MMap<uint8_t> predicate_index_arrays_;
    std::vector<std::span<uint>> ps_sets_;
    std::vector<std::span<uint>> po_sets_;

    void BuildPredicateIndex();

    void SubBuildPredicateIndex(std::deque<uint>* task_queue,
                                std::mutex* task_queue_mutex,
                                std::condition_variable* task_queue_cv,
                                std::atomic<bool>* task_queue_empty);

   public:
    NewPredicateIndex();
    NewPredicateIndex(std::string file_path, uint max_predicate_id);
    NewPredicateIndex(std::shared_ptr<phmap::flat_hash_map<uint, std::vector<std::pair<uint, uint>>>> pso,
                      std::string file_path,
                      uint max_predicate_id);

    void Build();

    void Store();

    std::span<uint>& GetSSet(uint pid);

    std::span<uint>& GetOSet(uint pid);

    uint GetSSetSize(uint pid);

    uint GetOSetSize(uint pid);

    void Close();
};

#endif