#ifndef CHUNKED_VECTOR_HPP
#define CHUNKED_VECTOR_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

using uint = unsigned int;

class ChunkedVector {
   public:
    explicit ChunkedVector(size_t row_width = 0, size_t chunk_rows = 1024);

    void Reset(size_t row_width);

    void Clear();

    void AppendRowSpan(const uint* data);

    void AppendFlat(const std::vector<uint>& flat);

    std::span<const uint> RowAt(size_t index) const;

    size_t size() const { return row_count_; }

    size_t row_width() const { return row_width_; }

   private:
    std::vector<std::vector<uint>> chunks_;
    size_t row_width_;
    size_t row_count_;
    size_t chunk_rows_;
};

#endif
