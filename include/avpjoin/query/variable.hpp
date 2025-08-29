#ifndef VARIABLE_HPP
#define VARIABLE_HPP

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "avpjoin/index/index_retriever.hpp"

using Term = SPARQLParser::Term;

using Position = Term::Position;

struct Variable {
    std::string variable;

    Position position;

    uint triple_constant_id;

    Position triple_constant_pos;

    std::span<uint> pre_retrieve;

    int total_set_size;

    Variable* connection;

    bool is_none;

    bool is_single;

    int var_id;

    phmap::parallel_flat_hash_map<uint,
                                  std::vector<uint>*,
                                  phmap::Hash<uint>,
                                  std::equal_to<uint>,
                                  std::allocator<std::pair<const uint, std::vector<uint>*>>,
                                  4,
                                  std::mutex>
        cache;

    Variable();

    Variable(std::string variable, Position position, std::vector<uint>* pre_retrieve);

    Variable(std::string variable,
             Position position,
             uint triple_constant_id,
             Position triple_constant_pos,
             std::shared_ptr<IndexRetriever> index);

    ~Variable();

    std::vector<uint>* Retrieve(uint key);

    std::span<uint> PreRetrieve();

   private:
    std::shared_ptr<IndexRetriever> index_;
};

#endif