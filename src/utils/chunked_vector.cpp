#include "utils/chunked_vector.hpp"

#include <algorithm>
#include <span>
#include <cstdint>

ChunkedVector::ChunkedVector(size_t row_width, size_t chunk_rows)
    : row_width_(row_width), row_count_(0), chunk_rows_(chunk_rows ? chunk_rows : 10240) {
    chunks_.reserve(1);
}

void ChunkedVector::Reset(size_t row_width) {
    row_width_ = row_width;
    row_count_ = 0;
    chunks_.clear();
    chunks_.reserve(1);
}

void ChunkedVector::Clear() {
    row_count_ = 0;
    chunks_.clear();
}

void ChunkedVector::AppendRowSpan(const uint* data) {
    if (row_width_ == 0 || data == nullptr)
        return;

    AppendRows(data, 1);
}

void ChunkedVector::AppendRows(const uint* data, size_t row_count) {
    if (row_width_ == 0 || data == nullptr || row_count == 0)
        return;

    size_t remaining_rows = row_count;
    const uint* cursor = data;

    while (remaining_rows > 0) {
        if (chunks_.empty() || (chunks_.back().size() / row_width_) >= chunk_rows_) {
            chunks_.emplace_back();
            chunks_.back().reserve(chunk_rows_ * row_width_);
        }

        auto& chunk = chunks_.back();
        const size_t filled_rows = chunk.size() / row_width_;
        const size_t capacity_rows = chunk_rows_ - filled_rows;
        const size_t rows_to_copy = std::min(remaining_rows, capacity_rows);

        const uint* end_ptr = cursor + rows_to_copy * row_width_;
        chunk.insert(chunk.end(), cursor, end_ptr);

        cursor = end_ptr;
        remaining_rows -= rows_to_copy;
        row_count_ += rows_to_copy;
    }
}

void ChunkedVector::AppendFlat(const std::vector<uint>& flat) {
    if (row_width_ == 0 || flat.empty())
        return;
    const size_t rows = flat.size() / row_width_;
    AppendRows(flat.data(), rows);
}

std::span<const uint> ChunkedVector::RowAt(size_t index) const {
    if (row_width_ == 0 || index >= row_count_)
        return {};
    size_t chunk_idx = index / chunk_rows_;
    size_t offset_row = index % chunk_rows_;
    const auto& chunk = chunks_[chunk_idx];
    return std::span<const uint>(chunk.data() + offset_row * row_width_, row_width_);
}
