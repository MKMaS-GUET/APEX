#include <parallel_hashmap/phmap.h>
#include <queue>

#include "avpjoin/query/query_executor.hpp"

QueryExecutor::QueryExecutor(std::shared_ptr<IndexRetriever> index, SPARQLParser parser, uint max_threads)
    : max_threads_(max_threads), index_(index), parser_(parser) {
    zero_result_ = false;
    gen_plan_cost_ = 0;

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

    is_cycle_ = IsCycleGraph(adjacency_list_ud);
    int component_id = 0;

    // 使用BFS找到所有连通分量
    for (const auto& node : adjacency_list_ud) {
        if (var_to_component_.contains(node.first))
            continue;

        std::queue<std::string> q;
        q.push(node.first);
        var_to_component_[node.first] = component_id;

        while (!q.empty()) {
            std::string current = q.front();
            q.pop();

            for (const auto& neighbor : adjacency_list_ud[current]) {
                if (!var_to_component_.contains(neighbor)) {
                    var_to_component_[neighbor] = component_id;
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

        if (s.IsVariable() && var_to_component_.contains(s.value))
            components.insert(var_to_component_[s.value]);
        if (p.IsVariable() && var_to_component_.contains(p.value))
            components.insert(var_to_component_[p.value]);
        if (o.IsVariable() && var_to_component_.contains(o.value))
            components.insert(var_to_component_[o.value]);

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
                for (auto& [var, comp_id] : var_to_component_) {
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
        // std::cout << "----------------" << std::endl;

        std::set<std::string> vars_set;

        for (const auto& triple_pattern : sub_query) {
            const auto& s = triple_pattern.subject;
            const auto& p = triple_pattern.predicate;
            const auto& o = triple_pattern.object;
            // std::cout << s.value << " " << p.value << " " << o.value << std::endl;

            if (s.IsVariable())
                vars_set.insert(s.value);
            if (p.IsVariable())
                vars_set.insert(p.value);
            if (o.IsVariable())
                vars_set.insert(o.value);
        }

        sub_query_vars_.emplace_back(vars_set.begin(), vars_set.end());
    }
    // std::cout << "----------------" << std::endl;
}

bool QueryExecutor::IsCycleGraph(const phmap::flat_hash_map<std::string, std::vector<std::string>>& adj_list) {
    if (adj_list.size() < 3)
        return false;  // 环至少需要3个节点

    // 检查每个节点的度数是否都为2
    for (const auto& [node, neighbors] : adj_list) {
        if (neighbors.size() != 2)
            return false;
    }

    // 检查是否为连通图（从任意节点开始DFS能访问所有节点）
    phmap::flat_hash_set<std::string> visited;
    std::function<void(const std::string&)> dfs = [&](const std::string& node) {
        visited.insert(node);
        for (const auto& neighbor : adj_list.at(node)) {
            if (!visited.contains(neighbor))
                dfs(neighbor);
        }
    };

    dfs(adj_list.begin()->first);
    return visited.size() == adj_list.size();
}

QueryExecutor::~QueryExecutor() {
    for (auto& executor : executors_)
        executor->~SubQueryExecutor();
}

void QueryExecutor::Query() {
    uint total_limit = parser_.Limit();
    for (auto& sub_query : sub_queries_) {
        auto executor = new SubQueryExecutor(index_, sub_query, is_cycle_, total_limit, false, max_threads_);
        executors_.push_back(executor);
        executor->Query();
        uint count = executor->ResultSize();
        if (count == 0) {
            zero_result_ = true;
            return;
        }
        if (total_limit != __UINT32_MAX__) {
            total_limit = (total_limit + count - 1) / count;
            if (total_limit < 1)
                total_limit = 1;
        }
    }
}

void QueryExecutor::Train(UDPService& service) {
    uint limit = parser_.Limit();

    for (auto& sub_query : sub_queries_) {
        std::cout << "-------------------------------------" << std::endl;
        SubQueryExecutor base_executor = SubQueryExecutor(index_, sub_query, is_cycle_, limit, false, max_threads_);
        SubQueryExecutor leaner_executor = SubQueryExecutor(index_, sub_query, is_cycle_, limit, true, max_threads_);
        if (base_executor.query_end()) {
            zero_result_ = true;
            continue;
        }

        double plan_time = 0;
        std::chrono::duration<double, std::milli> time;
        double base_exec_time = 0;
        double leaner_exec_time = 0;
        while (!base_executor.ordering_complete() && !leaner_executor.ordering_complete()) {
            if (base_executor.UpdateFirstVariableRange())
                break;
            if (leaner_executor.UpdateFirstVariableRange())
                break;

            service.sendMessage("start");
            while (!base_executor.query_end() && !leaner_executor.query_end()) {
                std::cout << "-------------------------------------" << std::endl;
                auto start = std::chrono::high_resolution_clock::now();
                std::string base_next_variable = base_executor.NextVarieble();
                int base_result_len = base_executor.ProcessNextVariable(base_next_variable);
                time = std::chrono::high_resolution_clock::now() - start;
                base_exec_time = time.count();
                std::cout << "Base Processing " << base_next_variable << " takes: " << base_exec_time << " ms"
                          << std::endl;

                service.sendMessage(leaner_executor.query_graph());

                start = std::chrono::high_resolution_clock::now();
                std::string next_variable = service.receiveMessage();
                time = std::chrono::high_resolution_clock::now() - start;
                plan_time += time.count();

                start = std::chrono::high_resolution_clock::now();
                int leaner_result_len = leaner_executor.ProcessNextVariable(next_variable);
                time = std::chrono::high_resolution_clock::now() - start;
                leaner_exec_time = time.count();
                std::cout << "Leaner Processing " << next_variable << " takes: " << leaner_exec_time << " ms"
                          << std::endl;

                std::ostringstream reward_stream;
                reward_stream << "[" << (base_result_len - leaner_result_len) << ","
                              << (base_exec_time - leaner_exec_time) << "]";
                service.sendMessage(reward_stream.str());
            }
            base_executor.Reset();
            leaner_executor.Reset();
            service.sendMessage("end");

            std::cout << "gen plan takes " << plan_time << " ms." << std::endl;
            std::cout << "base execute takes " << base_executor.execute_cost() << " ms." << std::endl;
            std::cout << "leaner execute takes " << leaner_executor.execute_cost() << " ms." << std::endl;
        }
    }
}

void QueryExecutor::Test(UDPService& service) {
    uint total_limit = parser_.Limit();
    for (auto& sub_query : sub_queries_) {
        auto executor = new SubQueryExecutor(index_, sub_query, is_cycle_, total_limit, true, max_threads_);
        executors_.push_back(executor);
        if (executor->query_end()) {
            zero_result_ = true;
            continue;
        }

        std::chrono::duration<double, std::milli> time;
        while (!executor->ordering_complete()) {
            if (executor->UpdateFirstVariableRange())
                break;
            service.sendMessage("start");
            while (!executor->query_end()) {
                auto start = std::chrono::high_resolution_clock::now();
                service.sendMessage(executor->query_graph());
                std::string next_variable = service.receiveMessage();
                time = std::chrono::high_resolution_clock::now() - start;
                gen_plan_cost_ += time.count();

                start = std::chrono::high_resolution_clock::now();
                executor->ProcessNextVariable(next_variable);
                time = std::chrono::high_resolution_clock::now() - start;
                std::cout << "Processing " << next_variable << " takes: " << time.count() << " ms" << std::endl;
            }
            executor->Reset();
            service.sendMessage("end");
        }
        executor->PostProcess();

        uint count = executor->ResultSize();
        if (count == 0) {
            zero_result_ = true;
            return;
        }
        total_limit = (total_limit + count - 1) / count;
        if (total_limit < 1)
            total_limit = 1;
    }
}

uint QueryExecutor::PrintResult() {
    if (zero_result_)
        return 0;

    phmap::flat_hash_map<std::string, std::pair<uint, Position>> var_to_priority_position;
    uint col_offset = 0;
    std::vector<uint> col_offsets;
    for (auto& executor : executors_) {
        auto vars = executor->variable_order();
        auto var_group_data = executor->MappingVariable(vars);
        for (uint v_id = 0; v_id < vars.size(); ++v_id) {
            auto [it, _] = var_to_priority_position.emplace(vars[v_id], var_group_data[v_id]);
            it->second.first += col_offset;
        }
        col_offsets.push_back(col_offset);
        col_offset += vars.size();
    }

    // 直接生成优先级和位置向量
    const auto& var_print_order = parser_.ProjectVariables();
    std::vector<std::pair<uint, Position>> var_priority_position;
    var_priority_position.reserve(var_print_order.size());
    for (const auto& var : var_print_order)
        var_priority_position.push_back(var_to_priority_position[var]);

    // 初始化迭代器
    std::vector<ResultGenerator::iterator> iters, begins, ends;
    for (auto& executor : executors_) {
        auto [begin, end] = executor->ResultsIter();
        iters.push_back(begin);
        begins.push_back(begin);
        ends.push_back(end);
    }

    for (uint i = 0; i < var_print_order.size(); i++)
        std::cout << var_print_order[i] << " ";
    std::cout << std::endl;

    uint limit = parser_.Limit();
    uint count = 0;
    const uint num_executors = executors_.size();
    std::vector<uint> result_row(col_offset);

    // 笛卡尔积遍历
    while (true) {
        // 收集所有执行器的当前行数据
        for (uint i = 0; i < num_executors; ++i) {
            const auto& current_row = *(iters[i]);
            std::copy(current_row->begin(), current_row->end(), result_row.begin() + col_offsets[i]);
        }

        // 输出结果
        for (const auto& [index, pos] : var_priority_position)
            std::cout << index_->ID2String(result_row[index], pos) << " ";
        std::cout << std::endl;

        if (++count >= limit)
            break;

        // 移动到下一个组合
        int idx = num_executors - 1;
        while (idx >= 0 && ++iters[idx] == ends[idx]) {
            iters[idx] = begins[idx];
            if (--idx < 0)
                return count;  // 所有组合遍历完成
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

double QueryExecutor::build_group_cost() {
    double time = 0;
    for (auto e : executors_)
        time += e->build_group_cost();
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

double QueryExecutor::gen_plan_cost() {
    return gen_plan_cost_;
}