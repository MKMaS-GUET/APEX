#include "query/variable.hpp"

Variable::Variable()
    : position(SPARQLParser::Term::kShared),
      triple_constant_id(0),
      triple_constant_pos(SPARQLParser::Term::kShared),
      pre_retrieve(),
      total_set_size(-1),
      connection(nullptr),
      is_none(false),
      is_single(false),
      var_id(-1) {}

Variable::Variable(std::string variable, Position position, std::vector<uint>* pre_retrieve)
    : variable(variable),
      position(position),
      triple_constant_id(0),
      triple_constant_pos(SPARQLParser::Term::kShared),
      total_set_size(-1),
      connection(nullptr),
      is_none(false),
      is_single(true),
      var_id(-1) {
    this->pre_retrieve = std::span<uint>(*pre_retrieve);
}

Variable::Variable(std::string variable,
                   Position position,
                   uint triple_constant_id,
                   Position triple_constant_pos,
                   std::shared_ptr<IndexRetriever> index)
    : variable(variable),
      position(position),
      triple_constant_id(triple_constant_id),
      triple_constant_pos(triple_constant_pos),
      pre_retrieve(),
      total_set_size(-1),
      connection(nullptr),
      is_none(false),
      is_single(false),
      var_id(-1),
      index_(index) {}

std::vector<uint>* Variable::Retrieve(uint key) {
    auto it = cache.find(key);
    if (it != cache.end())
        return it->second;

    std::vector<uint>* result = nullptr;
    Position key_pos = connection->position;
    if (triple_constant_pos == SPARQLParser::Term::kSubject) {
        // s ?p ?o
        if (key_pos == SPARQLParser::Term::kPredicate)
            result = index_->GetBySP(triple_constant_id, key);
        else if (key_pos == SPARQLParser::Term::kObject)
            result = index_->GetBySO(triple_constant_id, key);
    } else if (triple_constant_pos == SPARQLParser::Term::kPredicate) {
        // ?s p ?o
        if (key_pos == SPARQLParser::Term::kSubject)
            result = index_->GetBySP(key, triple_constant_id);
        else if (key_pos == SPARQLParser::Term::kObject) {
            result = index_->GetByOP(key, triple_constant_id);
        }
    } else if (triple_constant_pos == SPARQLParser::Term::kObject) {
        // ?s ?p o
        if (key_pos == SPARQLParser::Term::kSubject)
            result = index_->GetBySO(key, triple_constant_id);
        else if (key_pos == SPARQLParser::Term::kPredicate)
            result = index_->GetByOP(triple_constant_id, key);
    }
    if (result != nullptr)
        cache.insert({key, result});
    return result;
}

std::span<uint> Variable::PreRetrieve() {
    if (pre_retrieve.size() == 0) {
        if (position == SPARQLParser::Term::kSubject) {
            if (triple_constant_pos == SPARQLParser::Term::kPredicate)
                pre_retrieve = index_->GetPreSSet(triple_constant_id);
            if (triple_constant_pos == SPARQLParser::Term::kObject) {
                auto vec_ptr = index_->GetByO(triple_constant_id);
                pre_retrieve = std::span<uint>(vec_ptr->data(), vec_ptr->size());
            }
        }
        if (position == SPARQLParser::Term::kPredicate) {
            if (triple_constant_pos == SPARQLParser::Term::kSubject)
                pre_retrieve = index_->GetSPreSet(triple_constant_id);
            if (triple_constant_pos == SPARQLParser::Term::kObject)
                pre_retrieve = index_->GetOPreSet(triple_constant_id);
        }
        if (position == SPARQLParser::Term::kObject) {
            if (triple_constant_pos == SPARQLParser::Term::kPredicate)
                pre_retrieve = index_->GetPreOSet(triple_constant_id);
            if (triple_constant_pos == SPARQLParser::Term::kSubject) {
                auto vec_ptr = index_->GetByS(triple_constant_id);
                pre_retrieve = std::span<uint>(vec_ptr->data(), vec_ptr->size());
            }
        }
    }
    return pre_retrieve;
}

Variable::~Variable() {
    if (cache.size()) {
        for (auto& [_, v] : cache)
            delete v;
        cache.clear();
    }
}