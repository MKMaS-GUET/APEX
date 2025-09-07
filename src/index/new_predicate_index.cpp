#include <filesystem>
#include <iostream>
#include <thread>

#include "avpjoin/index/daas.hpp"
#include "avpjoin/index/new_predicate_index.hpp"

void NewPredicateIndex::Index::Build(std::vector<std::pair<uint, uint>>& so_pairs) {
    std::vector<uint> s_set;
    std::vector<uint> o_set;

    for (const auto& so : so_pairs) {
        s_set.push_back(so.first);
        o_set.push_back(so.second);
    }
    std::sort(s_set.begin(), s_set.end());
    s_set.erase(std::unique(s_set.begin(), s_set.end()), s_set.end());
    std::cout << s_set.size() << " " << s_set.back() << std::endl;

    std::sort(o_set.begin(), o_set.end());
    o_set.erase(std::unique(o_set.begin(), o_set.end()), o_set.end());
    std::cout << o_set.size() << " " << o_set.back() << std::endl;

    uint id = 0;
    min_subject = s_set.front();
    for (auto& s : s_set) {
        subject_set_bitmap.Set(s - min_subject);
        sid2offset[s] = id;
        id++;
    }
    std::vector<uint>().swap(s_set);

    id = 0;
    min_object = o_set.front();
    for (auto& o : o_set) {
        object_set_bitmap.Set(o - min_object);
        oid2offset[o] = id;
        id++;
    }
    std::vector<uint>().swap(o_set);
}

void NewPredicateIndex::Index::Clear() {
    phmap::flat_hash_map<uint, uint>().swap(sid2offset);
    phmap::flat_hash_map<uint, uint>().swap(oid2offset);
}

NewPredicateIndex::NewPredicateIndex() {}

NewPredicateIndex::NewPredicateIndex(std::string file_path, uint max_predicate_id)
    : file_path_(file_path), max_predicate_id_(max_predicate_id) {
    predicate_index_ = MMap<uint>(file_path_ + "predicate_index");

    std::string index_path = file_path_ + "predicate_index_arrays";
    predicate_index_arrays_ = MMap<uint8_t>(index_path);

    ps_sets_ = std::vector<std::span<uint>>(max_predicate_id_);
    po_sets_ = std::vector<std::span<uint>>(max_predicate_id_);

    std::vector<std::pair<uint, uint>> s_sizes, o_sizes;
    for (uint pid = 1; pid <= max_predicate_id_; pid++) {
        s_sizes.push_back({GetSSetSize(pid), pid});
        o_sizes.push_back({GetOSetSize(pid), pid});
    }

    std::sort(s_sizes.begin(), s_sizes.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::sort(o_sizes.begin(), o_sizes.end(), [](const auto& a, const auto& b) { return a.first > b.first; });

    uint cnt = 0;
    for (auto it = s_sizes.begin(); it != s_sizes.end() && cnt < max_predicate_id; ++it, cnt++)
        GetSSet(it->second);
    cnt = 0;
    for (auto it = o_sizes.begin(); it != o_sizes.end() && cnt < max_predicate_id; ++it, cnt++)
        GetOSet(it->second);
}

NewPredicateIndex::NewPredicateIndex(
    std::shared_ptr<phmap::flat_hash_map<uint, std::vector<std::pair<uint, uint>>>> pso,
    std::string file_path,
    uint max_predicate_id)
    : file_path_(file_path), pso_(pso), max_predicate_id_(max_predicate_id) {
    std::filesystem::path dir = std::filesystem::path(file_path_);
    if (!std::filesystem::exists(dir))
        std::filesystem::create_directories(dir);
}

void NewPredicateIndex::Build() {
    index_ = std::vector<Index>(max_predicate_id_);
    BuildPredicateIndex();
}

void NewPredicateIndex::BuildPredicateIndex() {
    std::vector<std::pair<uint, uint>> predicate_rank;
    predicate_rank.reserve(max_predicate_id_);
    for (uint pid = 1; pid <= max_predicate_id_; ++pid) {
        pso_->at(pid).shrink_to_fit();
        predicate_rank.emplace_back(pid, pso_->at(pid).size());
    }
    std::sort(predicate_rank.begin(), predicate_rank.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::deque<uint> task_queue;
    std::mutex task_queue_mutex;
    std::condition_variable task_queue_cv;
    std::atomic<bool> task_queue_empty{false};

    uint cpu_count = std::thread::hardware_concurrency();
    // uint cpu_count = 1;

    for (uint i = 0; i < max_predicate_id_; i++)
        task_queue.push_back(predicate_rank[i].first);

    std::vector<std::thread> threads;
    for (uint tid = 0; tid < cpu_count; tid++) {
        threads.emplace_back(std::bind(&NewPredicateIndex::SubBuildPredicateIndex, this, &task_queue, &task_queue_mutex,
                                       &task_queue_cv, &task_queue_empty));
    }
    for (auto& t : threads)
        t.join();
}

void NewPredicateIndex::SubBuildPredicateIndex(std::deque<uint>* task_queue,
                                               std::mutex* task_queue_mutex,
                                               std::condition_variable* task_queue_cv,
                                               std::atomic<bool>* task_queue_empty) {
    while (true) {
        std::unique_lock<std::mutex> lock(*task_queue_mutex);
        while (task_queue->empty() && !task_queue_empty->load())
            task_queue_cv->wait(lock);
        if (task_queue->empty())
            break;  // No more tasks

        uint pid = task_queue->front();
        task_queue->pop_front();
        if (task_queue->empty()) {
            task_queue_empty->store(true);
        }
        lock.unlock();
        index_[pid - 1].Build(pso_->at(pid));
    }
}

void NewPredicateIndex::Store() {
    auto beg = std::chrono::high_resolution_clock::now();

    ulong predicate_index_arrays_file_size = 0;
    uint set_byte_cnt;
    for (uint pid = 1; pid <= max_predicate_id_; pid++) {
        set_byte_cnt = (index_[pid - 1].subject_set_bitmap.Size() + 7) / 8;
        predicate_index_arrays_file_size += set_byte_cnt;
        set_byte_cnt = (index_[pid - 1].object_set_bitmap.Size() + 7) / 8;
        predicate_index_arrays_file_size += set_byte_cnt;
    }

    MMap<uint> predicate_index = MMap<uint>(file_path_ + "predicate_index", max_predicate_id_ * 6 * 4);
    MMap<uint8_t> predicate_index_arrays =
        MMap<uint8_t>(file_path_ + "predicate_index_arrays", predicate_index_arrays_file_size);

    ulong arrays_file_offset = 0;

    Bitset* ps_set;
    Bitset* po_set;

    uint8_t buffer = 0;
    uint buffer_offset = 7;
    for (uint pid = 1; pid <= max_predicate_id_; pid++) {
        ps_set = &index_[pid - 1].subject_set_bitmap;
        po_set = &index_[pid - 1].object_set_bitmap;

        buffer = 0;
        buffer_offset = 7;

        predicate_index[(pid - 1) * 6] = arrays_file_offset;
        predicate_index[(pid - 1) * 6 + 1] = index_[pid - 1].min_subject;
        predicate_index[(pid - 1) * 6 + 2] = index_[pid - 1].sid2offset.size();
        for (uint i = 0; i < ps_set->Size(); i++) {
            if (ps_set->Get(i) == 1)
                buffer |= 1 << buffer_offset;
            if (buffer_offset == 0) {
                arrays_file_offset++;
                predicate_index_arrays.Write(buffer);
                buffer = 0;
                buffer_offset = 8;
            }
            buffer_offset--;
        }
        if (buffer != 0) {
            arrays_file_offset++;
            predicate_index_arrays.Write(buffer);
        }

        buffer = 0;
        buffer_offset = 7;

        predicate_index[(pid - 1) * 6 + 3] = arrays_file_offset;
        predicate_index[(pid - 1) * 6 + 4] = index_[pid - 1].min_object;
        predicate_index[(pid - 1) * 6 + 5] = index_[pid - 1].oid2offset.size();
        for (uint i = 0; i < po_set->Size(); i++) {
            if (po_set->Get(i) == 1)
                buffer = buffer | 1 << buffer_offset;
            if (buffer_offset == 0) {
                arrays_file_offset++;
                predicate_index_arrays.Write(buffer);
                buffer = 0;
                buffer_offset = 8;
            }
            buffer_offset--;
        }
        if (buffer != 0) {
            arrays_file_offset++;
            predicate_index_arrays.Write(buffer);
        }
    }

    predicate_index.CloseMap();
    predicate_index_arrays.CloseMap();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - beg;
    std::cout << "store predicate index takes " << diff.count() << " ms.               " << std::endl;
}

std::span<uint>& NewPredicateIndex::GetSSet(uint pid) {
    if (ps_sets_[pid - 1].size() == 0) {
        auto beg = std::chrono::high_resolution_clock::now();

        uint s_set_offset = predicate_index_[(pid - 1) * 6];
        uint min_subject = predicate_index_[(pid - 1) * 6 + 1];
        uint o_set_offset = predicate_index_[(pid - 1) * 6 + 3];
        uint byte_cnt = o_set_offset - s_set_offset;

        // 从位图中解码实际值
        std::vector<uint> actual_values;
        for (uint i = 0; i < byte_cnt; i++) {
            uint bitmap_word = predicate_index_arrays_[s_set_offset + i];
            if (bitmap_word != 0) {
                for (int bit = 7; bit >= 0; bit--) {
                    if ((bitmap_word >> bit) & 1) {
                        uint position = (i * 8) + (7 - bit);
                        actual_values.push_back(position + min_subject);
                    }
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - beg;
        std::cout << pid << " " << diff.count() << " ms." << std::endl;

        // 创建实际值的数组
        uint* set = new uint[actual_values.size()];
        for (size_t i = 0; i < actual_values.size(); i++)
            set[i] = actual_values[i];

        ps_sets_[pid - 1] = std::span<uint>(set, actual_values.size());
    }
    return ps_sets_[pid - 1];
};

std::span<uint>& NewPredicateIndex::GetOSet(uint pid) {
    if (po_sets_[pid - 1].size() == 0) {
        auto beg = std::chrono::high_resolution_clock::now();

        uint o_set_offset = predicate_index_[(pid - 1) * 6 + 3];
        uint min_object = predicate_index_[(pid - 1) * 6 + 4];
        uint byte_cnt;

        if (pid != max_predicate_id_)
            byte_cnt = predicate_index_[pid * 6] - o_set_offset;
        else
            byte_cnt = predicate_index_arrays_.size_ - o_set_offset;

        // 从位图中解码实际值
        std::vector<uint> actual_values;
        for (uint i = 0; i < byte_cnt; i++) {
            uint bitmap_word = predicate_index_arrays_[o_set_offset + i];
            if (bitmap_word != 0) {
                for (int bit = 7; bit >= 0; bit--) {
                    if ((bitmap_word >> bit) & 1) {
                        uint position = (i * 8) + (7 - bit);
                        actual_values.push_back(position + min_object);
                    }
                }
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - beg;
        std::cout << pid << " " << diff.count() << " ms." << std::endl;

        // 创建实际值的数组
        uint* set = new uint[actual_values.size()];

        for (size_t i = 0; i < actual_values.size(); i++)
            set[i] = actual_values[i];

        po_sets_[pid - 1] = std::span<uint>(set, actual_values.size());
    }
    return po_sets_[pid - 1];
}

uint NewPredicateIndex::GetSSetSize(uint pid) {
    return predicate_index_[(pid - 1) * 6 + 2];
}

uint NewPredicateIndex::GetOSetSize(uint pid) {
    return predicate_index_[(pid - 1) * 6 + 5];
}

void NewPredicateIndex::Close() {
    predicate_index_.CloseMap();
    predicate_index_arrays_.CloseMap();

    std::vector<std::span<uint>>().swap(ps_sets_);
    std::vector<std::span<uint>>().swap(po_sets_);
}