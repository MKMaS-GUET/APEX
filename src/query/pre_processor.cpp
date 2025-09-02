#include "avpjoin/query/pre_processor.hpp"

using TripplePattern = std::vector<std::array<SPARQLParser::Term, 3>>;

PreProcessor::PreProcessor(std::shared_ptr<IndexRetriever> index,
                           const std::vector<SPARQLParser::TriplePattern>& triple_partterns,
                           bool use_order_generator) {
    auto begin = std::chrono::high_resolution_clock::now();

    zero_result_ = false;

    use_order_generator_ = use_order_generator;

    TripplePattern one_variable_tp;
    TripplePattern two_variable_tp;
    TripplePattern three_variable_tp;

    for (const auto& triple_parttern : triple_partterns) {
        auto& s = triple_parttern.subject;
        auto& p = triple_parttern.predicate;
        auto& o = triple_parttern.object;

        if (!p.IsVariable() && index->Term2ID(p) == 0) {
            zero_result_ = true;
            return;
        }

        if (triple_parttern.variable_cnt == 1)
            one_variable_tp.push_back({{s, p, o}});
        if (triple_parttern.variable_cnt == 2)
            two_variable_tp.push_back({s, p, o});
        if (triple_parttern.variable_cnt == 3)
            three_variable_tp.push_back({s, p, o});
    }

    for (const auto& tp : one_variable_tp) {
        auto& [s, p, o] = tp;

        std::vector<uint>* set = nullptr;
        if (s.IsVariable()) {
            uint oid = index->Term2ID(o);
            uint pid = index->Term2ID(p);
            set = index->GetByOP(oid, pid);

            if (use_order_generator_)
                query_graph_.AddVertex({s.value, set->size()});

            variables_.insert(s.value);
            str2var_[s.value].emplace_back(s.value, s.position, set);
        }
        if (p.IsVariable()) {
            uint sid = index->Term2ID(s);
            uint oid = index->Term2ID(o);
            set = index->GetBySO(sid, oid);

            if (use_order_generator_)
                query_graph_.AddVertex({p.value, set->size()});

            variables_.insert(p.value);
            str2var_[p.value].emplace_back(p.value, p.position, set);
        }
        if (o.IsVariable()) {
            uint sid = index->Term2ID(s);
            uint pid = index->Term2ID(p);
            set = index->GetBySP(sid, pid);

            if (use_order_generator_)
                query_graph_.AddVertex({o.value, set->size()});

            variables_.insert(o.value);
            str2var_[o.value].emplace_back(o.value, o.position, set);
        }
        if (set == nullptr || set->size() == 0) {
            zero_result_ = true;
            return;
        }
    }

    for (const auto& tp : two_variable_tp) {
        auto& [s, p, o] = tp;

        if (s.IsVariable() && p.IsVariable()) {
            uint oid = index->Term2ID(o);

            if (use_order_generator_)
                query_graph_.AddEdge({s.value, index->GetByO(oid)->size()}, {p.value, index->GetOPreSet(oid).size()},
                                     {0, Position::kObject});

            variables_.insert(s.value);
            variables_.insert(p.value);
            Variable& s_var = str2var_[s.value].emplace_back(s.value, s.position, oid, o.position, index);
            Variable& p_var = str2var_[p.value].emplace_back(p.value, p.position, oid, o.position, index);
            s_var.connection = &p_var;
            p_var.connection = &s_var;
        }
        if (s.IsVariable() && o.IsVariable()) {
            uint pid = index->Term2ID(p);

            if (use_order_generator_)
                query_graph_.AddEdge({s.value, index->GetSSetSize(pid)}, {o.value, index->GetOSetSize(pid)},
                                     {pid, Position::kPredicate});

            variables_.insert(s.value);
            variables_.insert(o.value);
            Variable& s_var = str2var_[s.value].emplace_back(s.value, s.position, pid, p.position, index);
            Variable& o_var = str2var_[o.value].emplace_back(o.value, o.position, pid, p.position, index);
            s_var.connection = &o_var;
            o_var.connection = &s_var;
        }
        if (p.IsVariable() && o.IsVariable()) {
            uint sid = index->Term2ID(s);
            if (use_order_generator_)
                query_graph_.AddEdge({p.value, index->GetSPreSet(sid).size()}, {o.value, index->GetByS(sid)->size()},
                                     {0, Position::kSubject});

            variables_.insert(p.value);
            variables_.insert(o.value);
            Variable& p_var = str2var_[p.value].emplace_back(p.value, p.position, sid, s.position, index);
            Variable& o_var = str2var_[o.value].emplace_back(o.value, o.position, sid, s.position, index);
            p_var.connection = &o_var;
            o_var.connection = &p_var;
        }
    }
    if (use_order_generator_)
        query_graph_.Init();

    auto end = std::chrono::high_resolution_clock::now();
    process_cost_ = end - begin;
}

std::vector<std::pair<uint, Position>> PreProcessor::MappingVariable(const std::vector<std::string>& variables) {
    std::vector<std::pair<uint, Position>> ret;
    ret.reserve(variables.size());

    for (const auto& var : variables) {
        const auto& vars = str2var_[var];
        if (vars.empty())
            continue;
        uint var_id = vars.front().var_id;

        if (vars.size() == 1) {
            ret.emplace_back(var_id, vars.front().position);
            continue;
        }

        // Count positions
        int pos_count[3] = {0, 0, 0};  // subject, predicate, object
        for (const auto& var : vars) {
            switch (var.position) {
                case Position::kSubject:
                    pos_count[0]++;
                    break;
                case Position::kPredicate:
                    pos_count[1]++;
                    break;
                case Position::kObject:
                    pos_count[2]++;
                    break;
                default:
                    break;
            }
        }

        if (pos_count[1] > 0) {
            ret.emplace_back(var_id, Position::kPredicate);
        } else if (pos_count[0] > 0) {
            ret.emplace_back(var_id, Position::kSubject);
        } else if (pos_count[2] > 0) {
            ret.emplace_back(var_id, Position::kObject);
        }
    }
    return ret;
}

void PreProcessor::ResetQueryGraph() {
    query_graph_.Reset();
}

std::list<Variable>* PreProcessor::VarsOf(std::string variable) {
    return &str2var_[variable];
}

phmap::flat_hash_set<std::string> PreProcessor::variables() {
    return variables_;
}

uint PreProcessor::VariableCount() {
    return variables_.size();
}

void PreProcessor::UpdateQueryGraph(std::string variable, uint cur_est_size) {
    query_graph_.UpdateQueryGraph(variable, cur_est_size);
}

std::string PreProcessor::query_graph() {
    return query_graph_.ToString();
}

bool PreProcessor::use_order_generator() {
    return use_order_generator_;
}

double PreProcessor::process_cost() {
    return process_cost_.count();
}

bool PreProcessor::zero_result() {
    return zero_result_;
}