// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2025 Tiago de Paula Peixoto <tiago@skewed.de>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU Lesser General Public License as published by the Free
// Software Foundation; either version 3 of the License, or (at your option) any
// later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
// details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#ifndef GRAPH_UTIL_HH
#define GRAPH_UTIL_HH

// allow boost::python to bind to std::functions
#include <boost/mpl/vector.hpp>
namespace boost { namespace python { namespace detail {
    template <class R_, class... PS_, class T=void>
    boost::mpl::vector<R_, PS_...>
    get_signature(std::function<R_ (PS_...)>&, T* = nullptr)
    {
        return boost::mpl::vector<R_, PS_...>();
    }
} } }

#include <ranges>
#include <random>
#include <string>

#include <boost/graph/graph_traits.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/range/iterator_range.hpp>

#include <boost/functional/hash.hpp>

#include "graph_selectors.hh"
#include "graph_filtered.hh"

namespace graph_tool
{

//
// Metaprogramming
// ===============

// check if is an instance of a template
template<class T, template<class...> class U>
inline constexpr bool is_instance_of_v = std::false_type{};

template<template<class...> class U, class... Vs>
inline constexpr bool is_instance_of_v<U<Vs...>,U> = std::true_type{};

// useful metafunction to determine whether a graph is directed or not

struct is_directed_
{
    template <class Graph>
    struct apply
    {
        typedef std::is_convertible<typename boost::graph_traits<Graph>::directed_category,
                                    boost::directed_tag> type;
    };
};

template <class Graph>
constexpr bool is_directed_v = is_directed_::apply<Graph>::type::value;

// This will actually count the number of vertices on a graph. Always O(V).
template <class Graph>
size_t hard_num_vertices(Graph& g)
{
    size_t n = 0;
    #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
        reduction(+:n)
    parallel_vertex_loop_no_spawn(g, [&](auto){ ++n; });
    return n;
};

// This will count "by hand" the number of edges on a graph. Always O(E).
template <class Graph>
size_t hard_num_edges(Graph& g)
{
    size_t n = 0;
    #pragma omp parallel if (num_vertices(g) > get_openmp_min_thresh()) \
        reduction(+:n)
    parallel_edge_loop_no_spawn(g, [&](auto){ ++n; });
    return n;
};

// returns true if vertices u and v are adjacent. This is O(k(u)).
template <class Graph>
bool is_adjacent(typename boost::graph_traits<Graph>::vertex_descriptor u,
                 typename boost::graph_traits<Graph>::vertex_descriptor v,
                 const Graph& g )
{
    for (const auto& e : out_edges_range(u, g))
    {
        if (target(e, g) == v)
            return true;
    }
    return false;
}

// computes the out-degree of a graph, ignoring self-edges
template <class Graph>
inline size_t
out_degree_no_loops(typename boost::graph_traits<Graph>::vertex_descriptor v,
                    const Graph &g)
{
    size_t k = 0;
    typename boost::graph_traits<Graph>::adjacency_iterator a,a_end;
    for (std::tie(a,a_end) = adjacent_vertices(v,g); a != a_end; ++a)
        if (*a != v)
            k++;
    return k;
}

// computes the out-degree of a graph, ignoring self-edges
template <class Graph, class Weights>
inline typename boost::property_traits<Weights>::value_type
out_degree_no_loops_weighted(typename boost::graph_traits<Graph>::vertex_descriptor v,
                             Weights w, const Graph &g)
{
    typename boost::property_traits<Weights>::value_type k = 0;
    typename boost::graph_traits<Graph>::out_edge_iterator e, e_end;
    for (std::tie(e, e_end) = out_edges(v, g); e != e_end; ++e)
        if (target(*e, g) != v)
            k += get(w, *e);
    return k;
}


template <class GraphOrig, class GraphTarget>
void graph_copy(const GraphOrig& g, GraphTarget& gt)
{
    typename boost::property_map<GraphOrig, boost::vertex_index_t>::type index = get(boost::vertex_index, g);
    typedef typename boost::graph_traits<GraphTarget>::vertex_descriptor tvertex_t;
    std::vector<tvertex_t> vmap(num_vertices(g));
    typename boost::graph_traits<GraphOrig>::vertex_iterator v, v_end;
    for (std::tie(v, v_end) = vertices(g); v != v_end; ++v)
        vmap[index[*v]] = add_vertex(gt);

    typename boost::graph_traits<GraphOrig>::edge_iterator e, e_end;
    for (std::tie(e, e_end) = edges(g); e != e_end; ++e)
        add_edge(vmap[index[source(*e, g)]], vmap[index[target(*e, g)]], gt);
}


//
// Parallel loops
// ==============

template <class Graph, class F, class Except = void>
auto parallel_vertex_loop_no_spawn(const Graph& g, F&& f,
                                   [[maybe_unused]] Except* ex=(void*)(nullptr))
{
    std::string err;
    bool skip = false;

    size_t N = num_vertices(g);
    #pragma omp for schedule(runtime)
    for (size_t i = 0; i < N; ++i)
    {
        if constexpr (!std::is_same_v<Except, void>)
        {
            if (skip)
                continue;
        }

        auto v = vertex(i, g);
        if (!is_valid_vertex(v, g))
            continue;

        if constexpr (std::is_same_v<Except, void>)
        {
            f(v);
        }
        else
        {
            try
            {
                f(v);
            }
            catch (const Except& ex)
            {
                skip = true;
                #pragma omp critical
                err = ex.what();
            }
        }
    }

    return std::make_tuple(skip, err);
}

template <class Graph, class F, class Except = void>
void parallel_vertex_loop(const Graph& g, F&& f,
                          size_t thres = get_openmp_min_thresh(),
                          [[maybe_unused]] Except* ex=(void*)(nullptr))
{
    std::tuple<bool, std::string> ret;
    #pragma omp parallel if (num_vertices(g) > thres)
    ret = parallel_vertex_loop_no_spawn<Graph, F>(g, std::forward<F>(f), ex);

    if constexpr (!std::is_same_v<Except, void>)
    {
        if (get<0>(ret))
            throw Except(get<1>(ret));
    }
}

template <class Graph>
const auto& get_dir(const Graph& g, std::true_type)
{ return g; }

template <class Graph>
const auto& get_dir(const Graph& g, std::false_type)
{ return g.original_graph(); }

template <class Graph, class EPred, class VPred>
auto get_dir(const boost::filt_graph<Graph, EPred, VPred>& g,
             std::false_type)
{
    typedef typename
        std::remove_reference<decltype(g._g.original_graph())>::type g_t;
    return boost::filt_graph<g_t, EPred, VPred>(g._g.original_graph(),
                                                g._edge_pred,
                                                g._vertex_pred);
}

template <class Graph, class F, class Except = void>
auto parallel_edge_loop_no_spawn(const Graph& g, F&& f,
                                 [[maybe_unused]] Except* ex = (void*)(nullptr))
{
    auto&& u = get_dir(g, typename is_directed_::apply<Graph>::type());
    typedef typename std::remove_const
        <typename std::remove_reference<decltype(u)>::type>::type graph_t;
    static_assert(is_directed_::apply<graph_t>::type::value,
                  "graph must be directed at this point");

    auto dispatch =
        [&](auto v)
        {
            for (auto e : out_edges_range(v, u))
                f(e);
        };

    return parallel_vertex_loop_no_spawn(u, dispatch, ex);
}

template <class Graph, class F, class Except = void>
void parallel_edge_loop(const Graph& g, F&& f,
                        size_t thres = get_openmp_min_thresh(),
                        [[maybe_unused]] Except* ex = (void*)(nullptr))
{
    std::tuple<bool, std::string> ret;
    #pragma omp parallel if (num_vertices(g) > thres)
    ret = parallel_edge_loop_no_spawn<Graph, F>(g, std::forward<F>(f), ex);

    if constexpr (!std::is_same_v<Except, void>)
    {
        if (get<0>(ret))
            throw Except(get<1>(ret));
    }
}

template <class Mutex>
struct canary_mutex
    : public Mutex
{
    size_t _canary;
};

#ifdef _LIBCPP_VERSION
/// work around libc++ poor C++23 support
template <class R>
inline auto enumerate(R&& rg)
{
    return std::views::zip(std::views::iota(size_t(0),
                                            size_t(std::ranges::size(rg))),
                           std::forward<R>(rg));
}
#else
using std::views::enumerate;
#endif

template <class Range, class F, class Except = void>
void parallel_loop_no_spawn(Range&& vals, F&& f,
                            [[maybe_unused]] Except* ex = (void*)(nullptr))
{
    std::conditional_t<std::is_same_v<Except, void>,
                       int, Except> e;
    bool skip = false;

    #pragma omp for schedule(runtime)
    for (auto&& [i, v] : enumerate(std::forward<Range>(vals)))
    {
        if constexpr (!std::is_same_v<Except, void>)
        {
            if (skip)
                continue;
        }

        if constexpr (std::is_same_v<Except, void>)
        {
            f(i, v);
        }
        else
        {
            try
            {
                f(i, v);
            }
            catch (const Except& ex)
            {
                skip = true;
                #pragma omp critical
                e = ex;
            }
        }
    }

    if constexpr (!std::is_same_v<Except, void>)
    {
        if (skip)
            throw e;
    }
}

template <class Range, class F, class Except=void>
void parallel_loop(Range&& vals, F&& f, size_t thres = get_openmp_min_thresh(),
                   Except* ex = (void*)(nullptr))
{
    #pragma omp parallel if (std::ranges::size(vals) > thres)
    parallel_loop_no_spawn<Range, F>(std::forward<Range>(vals),
                                     std::forward<F>(f), ex);
}

} // namespace graph_tool

//
// Data type string representation
// ===============================
//
// String representation of individual data types. We have to take care
// specifically that no information is lost with floating point I/O.
//
// These are implemented in graph_io.cc.

namespace boost
{

template <>
std::string lexical_cast<std::string,uint8_t>(const uint8_t& val);
template <>
uint8_t lexical_cast<uint8_t,std::string>(const std::string& val);
template <>
std::string lexical_cast<std::string,double>(const double& val);
template <>
double lexical_cast<double,std::string>(const std::string& val);
template <>
std::string lexical_cast<std::string,long double>(const long double& val);
template <>
long double lexical_cast<long double,std::string>(const std::string& val);
}

// std::vector<> stream i/o
namespace std
{
template <class Type>
ostream& operator<<(ostream& out, const std::vector<Type>& vec)
{
    for (size_t i = 0; i < vec.size(); ++i)
    {
        out << boost::lexical_cast<std::string>(vec[i]);
        if (i < vec.size() - 1)
            out << ", ";
    }
    return out;
}

template <class Type>
istream& operator>>(istream& in, std::vector<Type>& vec)
{
    using namespace boost;
    using namespace boost::algorithm;

    vec.clear();
    std::string data;
    getline(in, data);
    if (data == "")
        return in; // empty std::strings are OK
    std::vector<std::string> split_data;
    split(split_data, data, is_any_of(","));
    for (size_t i = 0; i < split_data.size(); ++i)
    {
        trim(split_data[i]);
        vec.push_back(lexical_cast<Type>(split_data[i]));
    }
    return in;
}

// std::string vectors need special attention, since separators must be properly
// escaped.
template <>
ostream& operator<<(ostream& out, const std::vector<std::string>& vec);

template <>
istream& operator>>(istream& in, std::vector<std::string>& vec);

} // std namespace

// This will iterate over a random permutation of a random access sequence, by
// swapping the values of the sequence as it iterates
template <class RandomAccessIterator, class RNG,
          class RandomDist = std::uniform_int_distribution<size_t>>
class random_permutation_iterator
{
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = typename RandomAccessIterator::value_type;
    using difference_type = typename RandomAccessIterator::difference_type;
    using pointer = typename RandomAccessIterator::pointer;
    using reference = typename RandomAccessIterator::reference;

    random_permutation_iterator(RandomAccessIterator begin,
                                RandomAccessIterator end, RNG& rng)
        : _i(begin), _end(end), _rng(&rng)
    {
        if(_i != _end)
        {
            RandomDist random(0,  _end - _i - 1);
            std::iter_swap(_i, _i + random(*_rng));
        }
    }

    typename std::iterator_traits<RandomAccessIterator>::reference operator*()
    {
        return *_i;
    }

    random_permutation_iterator& operator++()
    {
        ++_i;
        if(_i != _end)
        {
            RandomDist random(0,  _end - _i - 1);
            std::iter_swap(_i, _i + random(*_rng));
        }
        return *this;
    }

    bool operator==(const random_permutation_iterator& ri)
    {
        return _i == ri._i;
    }

    bool operator!=(const random_permutation_iterator& ri)
    {
        return _i != ri._i;
    }

    size_t operator-(const random_permutation_iterator& ri)
    {
        return _i - ri._i;
    }

private:
    RandomAccessIterator _i, _end;
    RNG* _rng;
};

template <class Iter, class RNG>
auto random_permutation_range(Iter begin, Iter end, RNG& rng)
{
    random_permutation_iterator<Iter, RNG> rbegin(begin, end, rng);
    random_permutation_iterator<Iter, RNG> rend(end, end, rng);
    return boost::make_iterator_range(rbegin, rend);
}

template <class V, class RNG>
auto random_permutation_range(V&& v, RNG& rng)
{
    return random_permutation_range(v.begin(), v.end(), rng);
}

// This will iterate over a forward sequence of size N and select a subset of
// size M <= N uniformly at random
template <class Iterator, class RNG>
class random_subset_iterator
{
public:
    using iterator_category = std::input_iterator_tag;
    using value_type = typename Iterator::value_type;
    using difference_type = typename Iterator::difference_type;
    using pointer = typename Iterator::pointer;
    using reference = typename Iterator::reference;

    random_subset_iterator(Iterator begin, Iterator end, size_t N, size_t M, RNG& rng)
        : _i(begin), _end(end), _N(N), _M(M), _m(0), _t(0), _rng(&rng)
    {
        skip();
    }

    random_subset_iterator(Iterator begin, Iterator end, size_t M, RNG& rng)
        : random_subset_iterator(begin, end, end - begin, M, rng)
    {}

    void skip()
    {
        if (_m == _M || _i == _end)
        {
            _i = _end;
            return;
        }

        while ((_i != _end) && ((_N - _t) * _u(*_rng) >= _M - _m))
        {
            ++_i;
            ++_t;
        }

        ++_t;
        ++_m;
    }

    typename std::iterator_traits<Iterator>::reference operator*()
    {
        return *_i;
    }

    random_subset_iterator& operator++() // prefix
    {
        ++_i;
        skip();
        return *this;
    }

    bool operator==(const random_subset_iterator& ri)
    {
        return _i == ri._i;
    }

    bool operator!=(const random_subset_iterator& ri)
    {
        return _i != ri._i;
    }

private:
    Iterator _i, _end;
    size_t _N;
    size_t _M;
    size_t _m;
    size_t _t;
    RNG* _rng;
    std::uniform_real_distribution<> _u;
};

template <class Iter, class RNG>
auto random_subset_range(Iter begin, Iter end, size_t N, size_t M, RNG& rng)
{
    random_subset_iterator<Iter, RNG> rbegin(begin, end, N, M, rng);
    random_subset_iterator<Iter, RNG> rend(end, end, N, M, rng);
    return boost::make_iterator_range(rbegin, rend);
}

template <class Iter, class RNG>
auto random_subset_range(Iter begin, Iter end, size_t M, RNG& rng)
{
    return random_subset_range(begin, end, end - begin, M, rng);
}

template <class V, class RNG>
auto random_subset_range(V&& v, size_t N, size_t M, RNG& rng)
{
    return random_subset_range(v.begin(), v.end(), N, M, rng);
}

template <class V, class RNG>
auto random_subset_range(V&& v, size_t M, RNG& rng)
{
    return random_subset_range(v, v.size(), M, rng);
}


// random acess iterator adaptor; useful for boost::multi_array<>, who insists
// to predent it's not random access, even though it is.

template <class Iter>
class random_access_adaptor
  : public boost::iterator_adaptor<
      random_access_adaptor<Iter>,         // Derived
      Iter,                                // Base
      boost::use_default,                  // Value
      std::random_access_iterator_tag>     // CategoryOrTraversal
{
 public:
    template <class... Args>
    random_access_adaptor(Args&&... args)
        : random_access_adaptor::iterator_adaptor_(std::forward<Args>(args)...) {}

    std::iter_reference_t<random_access_adaptor<Iter>>
    operator[](auto n) const
    {
        return *(*this + n);
    }
};

template <class Iter>
auto make_random_access_adaptor(Iter iter)
{
    static_assert(std::random_access_iterator<random_access_adaptor<Iter>>);
    return random_access_adaptor<Iter>(iter);
}

template <class Range>
auto make_random_access_adaptor_range(Range&& range)
{
    auto begin = make_random_access_adaptor(std::ranges::begin(range));
    auto end = make_random_access_adaptor(std::ranges::end(range));
    auto rg = graph_tool::mk_range(std::make_pair(begin, end));
    static_assert(std::ranges::random_access_range<decltype(rg)>);
    return rg;
}

//
// Useful hash<> specializations
//

namespace std
{

template <class Val>
void _hash_combine(size_t& seed, const Val& hash)
{
    seed ^= std::hash<Val>()(hash) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <size_t pos, class... T>
struct tuple_combine
{
    void operator()(size_t& seed, const std::tuple<T...>& v) const
    {
        std::_hash_combine(seed, std::get<pos-1>(v));
        tuple_combine<pos-1, T...>()(seed, v);
    }
};

template <class... T>
struct tuple_combine<0, T...>
{
    void operator()(size_t&, const std::tuple<T...>&) const {}
};

template <class... T>
struct hash<std::tuple<T...>>
{
    size_t operator()(std::tuple<T...> const& v) const
    {
        std::size_t seed = 0;
        tuple_combine<sizeof...(T), T...>()(seed, v);
        return seed;
    }
};

template <class T1, class T2>
struct hash<std::pair<T1, T2>>
{
    size_t operator()(std::pair<T1, T2> const& v) const
    {
        std::size_t seed = 0;
        std::_hash_combine(seed, v.first);
        std::_hash_combine(seed, v.second);
        return seed;
    }
};

template <class Value, class Allocator>
struct hash<std::vector<Value, Allocator>>
{
    size_t operator()(const std::vector<Value, Allocator>& v) const
    {
        size_t seed = 0;
        for (const auto& x : v)
            std::_hash_combine(seed, x);
        return seed;
    }
};

template <class Value, size_t N>
struct hash<std::array<Value, N>>
{
    size_t operator()(const std::array<Value, N>& x) const
    {
        size_t seed = 0;
        for (size_t i = 0; i < N; ++i)
            std::_hash_combine(seed, x[i]);
        return seed;
    }
};

template <class Value>
struct hash<std::complex<Value>>
{
    size_t operator()(const std::complex<Value>& v) const
    {
        size_t seed = hash<Value>()(v.real());
        std::_hash_combine(seed, hash<Value>()(v.imag()));
        return seed;
    }
};


}

#endif // GRAPH_UTIL_HH
