#ifndef RESULT_MAP_TRAVERSER_HPP
#define RESULT_MAP_TRAVERSER_HPP

#include <span>

#include "result_map.hpp"
#include "utils/chunked_vector.hpp"

using CandidateMap = phmap::flat_hash_map<uint, std::span<uint>>;

class VariableGroup {
   private:
    int level_;

    bool at_end_ = false;

    std::vector<ResultMap*> result_map_;

    std::vector<std::vector<std::pair<uint, uint>>> result_relation_;

    std::vector<std::vector<uint>> result_map_keys_;

    std::vector<std::vector<uint>*> candidate_value_;

    std::vector<uint> candidate_idx_;

    std::vector<uint> current_result_;

    ChunkedVector results_;

    std::vector<uint> empty;

    void Up();

    void Down();

    void Next();

    bool UpdateCurrentResult();

    void GenCandidateValue();

    std::span<const uint> RowAt(size_t index) const;

    void TraverseRange(uint start,
                       uint end,
                       const std::vector<std::vector<uint>>& result_map_keys_template,
                       std::vector<uint>& out);

   public:
    std::vector<uint> var_offsets;
    std::vector<uint> key_offsets;

    // 每一个 var 需要 results_ 中的哪一列数据
    std::vector<uint> var_result_offset;

    struct Group {
        std::vector<std::vector<uint>> ancestors;
        std::vector<uint> var_offsets;
        std::vector<uint> key_offsets;
    };

    class iterator {
       private:
        const VariableGroup* var_group_;
        size_t index_;

       public:
        inline iterator() = default;

        inline iterator(const VariableGroup* var_group, size_t index) : var_group_(var_group), index_(index) {}

        inline std::span<const uint> operator*() const { return var_group_->RowAt(index_); }

        inline iterator& operator++() {
            ++index_;
            return *this;
        }

        inline iterator operator++(int) {
            iterator tmp = *this;
            ++index_;
            return tmp;
        }

        inline iterator operator+(std::ptrdiff_t n) const { return iterator(var_group_, index_ + n); }

        inline iterator& operator+=(std::ptrdiff_t n) {
            index_ += n;
            return *this;
        }

        // 优化比较操作符
        inline bool operator==(const iterator& other) const { return index_ == other.index_; }

        inline bool operator!=(const iterator& other) const { return index_ != other.index_; }
    };

    VariableGroup() = default;

    VariableGroup(std::vector<ResultMap>& result_map,
                  std::pair<uint, uint> range,
                  std::vector<std::vector<std::pair<uint, uint>>>& result_relation,
                  Group group,
                  uint max_threads);

    VariableGroup(Group group);

    VariableGroup(ResultMap& map, Group group);

    VariableGroup(ResultMap& map, std::pair<uint, uint> range, Group group);

    ~VariableGroup();

    iterator begin() const { return iterator(this, 0); }

    iterator end() const { return iterator(this, size()); }

    size_t size() const { return results_.size(); }
};

#endif
