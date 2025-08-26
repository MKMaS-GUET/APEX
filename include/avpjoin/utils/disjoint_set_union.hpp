#ifndef DISJOINT_SET_UNION_HPP
#define DISJOINT_SET_UNION_HPP

#include <sys/types.h>
#include <numeric>
#include <vector>

struct DSU {
    std::vector<int> p, r;
    
    explicit DSU(uint n) : p(n), r(n, 0) { std::iota(p.begin(), p.end(), 0); }

    int Find(int x) { return p[x] == x ? x : p[x] = Find(p[x]); }

    void Unite(int a, int b) {
        a = Find(a);
        b = Find(b);
        if (a == b)
            return;
        if (r[a] < r[b])
            std::swap(a, b);
        p[b] = a;
        if (r[a] == r[b])
            r[a]++;
    }
};

#endif