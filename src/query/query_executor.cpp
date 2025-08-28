#include <parallel_hashmap/phmap.h>

#include <queue>
#include "avpjoin/query/query_executor.hpp"

QueryExecutor::QueryExecutor(std::shared_ptr<IndexRetriever> index, SPARQLParser parser)
    : index_(index), parser_(parser) {
    zero_result_ = false;

    phmap::flat_hash_map<std::string, std::vector<std::string>> adjacency_list_ud;

    auto add_edge = [&](const std::string& var1, const std::string& var2) {
        adjacency_list_ud[var1].emplace_back(var2);
        adjacency_list_ud[var2].emplace_back(var1);
    };

    for (const auto& triple_pattern : parser.TriplePatterns()) {
        const auto& s = triple_pattern.subject;
        const auto& p = triple_pattern.predicate;
        const auto& o = triple_pattern.object;

        if (triple_pattern.variable_cnt == 1) {
            if (s.IsVariable())
                adjacency_list_ud.try_emplace(s.value);
            if (p.IsVariable())
                adjacency_list_ud.try_emplace(p.value);
            if (o.IsVariable())
                adjacency_list_ud.try_emplace(o.value);
        } else if (triple_pattern.variable_cnt == 2) {
            if (s.IsVariable() && p.IsVariable() && !o.IsVariable())
                add_edge(s.value, p.value);
            else if (s.IsVariable() && !p.IsVariable() && o.IsVariable())
                add_edge(s.value, o.value);
            else if (!s.IsVariable() && p.IsVariable() && o.IsVariable())
                add_edge(p.value, o.value);
        }
    }

    phmap::flat_hash_map<std::string, int> var_to_component;
    int component_id = 0;

    // 使用BFS找到所有连通分量
    for (const auto& node : adjacency_list_ud) {
        if (var_to_component.contains(node.first))
            continue;

        std::queue<std::string> q;
        q.push(node.first);
        var_to_component[node.first] = component_id;

        while (!q.empty()) {
            std::string current = q.front();
            q.pop();

            for (const auto& neighbor : adjacency_list_ud[current]) {
                if (!var_to_component.contains(neighbor)) {
                    var_to_component[neighbor] = component_id;
                    q.push(neighbor);
                }
            }
        }

        component_id++;
    }

    // 将三元组模式分组到对应的连通图中
    phmap::flat_hash_map<int, std::vector<SPARQLParser::TriplePattern>> component_to_triples;

    for (const auto& triple_pattern : parser.TriplePatterns()) {
        const auto& s = triple_pattern.subject;
        const auto& p = triple_pattern.predicate;
        const auto& o = triple_pattern.object;

        // 找到这个三元组模式中所有变量所属的连通分量
        std::set<int> components;

        if (s.IsVariable() && var_to_component.contains(s.value))
            components.insert(var_to_component[s.value]);
        if (p.IsVariable() && var_to_component.contains(p.value))
            components.insert(var_to_component[p.value]);
        if (o.IsVariable() && var_to_component.contains(o.value))
            components.insert(var_to_component[o.value]);

        // 如果三元组模式没有变量或变量不在任何连通分量中，创建一个新的组
        if (components.empty()) {
            sub_queries_.push_back({triple_pattern});
            continue;
        }

        // 如果三元组模式中的变量属于多个连通分量，需要合并这些分量
        if (components.size() > 1) {
            // 找到最小的组件ID作为主ID
            int main_component = *components.begin();

            // 将所有相关的组件合并到主组件
            for (int comp : components) {
                if (comp == main_component)
                    continue;

                // 重新标记所有属于comp的变量
                for (auto& [var, comp_id] : var_to_component) {
                    if (comp_id == comp)
                        comp_id = main_component;
                }

                // 将comp中的三元组合并到main_component
                if (component_to_triples.contains(comp)) {
                    auto& triples = component_to_triples[comp];
                    component_to_triples[main_component].insert(component_to_triples[main_component].end(),
                                                                triples.begin(), triples.end());
                    component_to_triples.erase(comp);
                }
            }

            // 将当前三元组添加到主组件
            component_to_triples[main_component].push_back(triple_pattern);
        } else {
            // 只有一个连通分量，直接添加
            component_to_triples[*components.begin()].push_back(triple_pattern);
        }
    }

    // 将组件中的三元组添加到最终结果
    for (auto& [comp_id, triples] : component_to_triples) {
        sub_queries_.push_back(std::move(triples));
    }

    // 按照子查询的大小从小到大排序
    std::sort(sub_queries_.begin(), sub_queries_.end(),
              [](const auto& a, const auto& b) { return a.size() < b.size(); });

    // 获取每个子查询包含的变量
    sub_query_vars_.reserve(sub_queries_.size());
    for (const auto& sub_query : sub_queries_) {
        std::set<std::string> vars_set;

        for (const auto& triple_pattern : sub_query) {
            const auto& s = triple_pattern.subject;
            const auto& p = triple_pattern.predicate;
            const auto& o = triple_pattern.object;

            if (s.IsVariable())
                vars_set.insert(s.value);
            if (p.IsVariable())
                vars_set.insert(p.value);
            if (o.IsVariable())
                vars_set.insert(o.value);
        }

        sub_query_vars_.emplace_back(vars_set.begin(), vars_set.end());
    }
}

void QueryExecutor::Query() {
    uint total_limit = parser_.Limit();
    for (auto& sub_query : sub_queries_) {
        auto executor = new SubQueryExecutor(index_, sub_query, total_limit, false);
        executors_.push_back(executor);
        executor->Query();
        if (executor->zero_result()) {
            zero_result_ = true;
            return;
        }
        auto results = executor->results();
        sub_query_results_.push_back(results);
        total_limit = (total_limit + results->size() - 1) / results->size();
    }
}

uint QueryExecutor::PrintResult() {
    if (zero_result_)
        return 0;

    uint limit = parser_.Limit();
    uint count = 0;

    std::vector<std::vector<std::vector<uint>>::const_iterator> iters;
    std::vector<std::vector<std::vector<uint>>::const_iterator> ends;

    for (const auto& result : sub_query_results_) {
        iters.push_back(result->begin());
        ends.push_back(result->end());
    }

    // 执行笛卡尔积
    while (true) {
        // 输出当前组合
        std::vector<uint> combined_row;

        // 收集所有子查询结果的当前行
        for (size_t i = 0; i < sub_query_results_.size(); i++) {
            const auto& current_row = *(iters[i]);
            combined_row.insert(combined_row.end(), current_row.begin(), current_row.end());
        }

        // 输出结果（根据实际情况调整）
        // for (uint value : combined_row) {
        //     std::cout << value << " ";
        // }
        // std::cout << std::endl;

        count++;

        // 检查是否达到限制
        if (limit > 0 && count >= limit)
            break;

        // 移动到下一个组合
        int idx = sub_query_results_.size() - 1;
        bool all_done = false;

        while (idx >= 0) {
            iters[idx]++;

            if (iters[idx] == ends[idx]) {
                // 当前迭代器到达末尾，重置并检查前一个
                if (idx == 0) {
                    // 这是第一个迭代器，所有组合都已遍历完毕
                    all_done = true;
                    break;
                }
                iters[idx] = sub_query_results_[idx]->begin();
                idx--;
            } else {
                // 当前迭代器还有更多元素，继续处理
                break;
            }
        }

        if (all_done || idx < 0) {
            break;
        }
    }

    return count;
}

double QueryExecutor::preprocess_cost() {
    double time = 0;
    for (auto e : executors_)
        time += e->preprocess_cost();
    return time;
}

double QueryExecutor::execute_cost() {
    double time = 0;
    for (auto e : executors_)
        time += e->execute_cost();
    return time;
}

double QueryExecutor::gen_result_cost() {
    if (zero_result_)
        return 0;
    double time = 0;
    for (auto e : executors_)
        time += e->gen_result_cost();
    return time;
}