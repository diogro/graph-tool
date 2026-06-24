// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2026 Tiago de Paula Peixoto <tiago@skewed.de>
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

#ifndef HALF_EDGE_ADAPTOR
#define HALF_EDGE_ADAPTOR

namespace graph_tool
{

struct half_edge
{
    half_edge(const GraphInterface::edge_t& e, size_t endpoint,
              bool self_source = true)
        : _e(e), _endpoint(endpoint)
    {
        if (_e.s == _e.t)
        {
            if (self_source)
                _e.t = _null;
            else
                _e.s = _null;
        }

        if (endpoint == _null)
            std::println("endpoint is null: {} {}", _e.s, _e.t);

        if (e.s != endpoint && e.t != endpoint)
            std::println("invalid endpoint! {} {} {}", _e.s, _e.t, endpoint);
    }

    half_edge() = default;
    //half_edge(const half_edge&) = default;

    bool is_source() const
    {
        return _e.s == _endpoint;
    }

    size_t get_source() const
    {
        return _e.s;
    }

    size_t get_target() const
    {
        return _e.t;
    }

    auto&& get_edge() const
    {
        return _e;
    }

    auto get_endpoint() const
    {
        return _endpoint;
    }

    bool is_self() const
    {
        return _e.s == _null || _e.t == _null;
    }

    bool operator==(const half_edge& o) const
    {
        return _endpoint == o._endpoint && _e == o._e && ((_e.s == _null) == (o._e.s == _null));
    }

    bool operator>(const half_edge& o) const
    {
        if (_endpoint == o._endpoint)
            return _e.s > o._e.s;
        return _endpoint > o._endpoint;
    }

    bool operator<(const half_edge& o) const
    {
        if (_endpoint == o._endpoint)
            return _e.s < o._e.s;
        return _endpoint < o._endpoint;
    }

    void swap_endpoint()
    {
        if (_e.s == _null || _e.t == _null)
        {
            std::swap(_e.s, _e.t);
        }
        else
        {
            if (_endpoint == _e.s)
                _endpoint = _e.t;
            else
                _endpoint = _e.s;
        }
    }

    constexpr static inline size_t _null = std::numeric_limits<size_t>::max();

private:
    GraphInterface::edge_t _e;
    size_t _endpoint = std::numeric_limits<size_t>::max();
};

template<class T>
   requires std::is_same_v<T, half_edge>
size_t hash_value(const T& v)
{
    size_t seed = 0;
    boost::hash_combine(seed, hash_value(v.get_edge()));
    boost::hash_combine(seed, std::hash<size_t>{}(v.get_endpoint()));
    return seed;
}

template <class Graph>
struct half_edge_adaptor
    : public Graph
{
    half_edge_adaptor() = delete;

    const Graph& base() const
    {
        return reinterpret_cast<const Graph&>(*this);
    }

    Graph& base()
    {
        return reinterpret_cast<Graph&>(*this);
    }

    const auto& original_graph() const
    {
        return base();
    }

    auto& original_graph()
    {
        return base();
    }

    typedef Graph base_t;
    typedef half_edge vertex_t;
    typedef typename boost::graph_traits<Graph>::edge_descriptor edge_t;

    template <class EIter>
    class viter_t
      : public boost::iterator_facade<viter_t<EIter>,
                                      vertex_t,
                                      boost::forward_traversal_tag,
                                      vertex_t>
    {
    public:
        viter_t() = default;
//        viter_t(const viter_t&) = default;

        template <class G>
        viter_t(EIter eiter, G& g) : _iter(eiter), _g(&g.base()) {}

        bool equal(const viter_t& o) const
        {
            return _iter == o._iter && _source == o._source;
        }

        auto dereference() const
        {
            auto e = *_iter;
            if (_source)
                return vertex_t(e, source(e, *_g), true);
            else
                return vertex_t(e, target(e, *_g), false);
        }

        void increment()
        {
            if (_source)
            {
                _source = false;
            }
            else
            {
                _source = true;
                _iter++;
            }
        }

    private:
        EIter _iter;
        bool _source = true;
        const Graph* _g = nullptr;
    };

    struct eiter_t
      : public boost::iterator_facade<eiter_t,
                                      edge_t,
                                      boost::forward_traversal_tag,
                                      edge_t>
    {
    public:
        eiter_t() = default;
        eiter_t (const edge_t& e)
            : _e(e) {}

        bool equal(const eiter_t& o) const
        {
            return _e == o._e;
        }

        auto dereference() const
        {
            return _e;
        }

        void increment()
        {
            _e = edge_t();
        }

    private:
        edge_t _e;
    };


    class aiter_t
          : public boost::iterator_facade<aiter_t,
                                          vertex_t,
                                          boost::forward_traversal_tag,
                                          vertex_t>
    {
    public:
        aiter_t() = default;
        aiter_t(const half_edge& v)
            : _v(v) { _v.swap_endpoint(); }

        bool equal(const aiter_t& o) const
        {
            return _v == o._v;
        }

        auto dereference() const
        {
            return _v;
        }

        void increment()
        {
            _v = half_edge();
        }

    private:
        half_edge _v;
    };

};

template <class Graph>
auto& make_half_edge_adaptor(Graph& g)
{
    return reinterpret_cast<half_edge_adaptor<Graph>&>(g);
}

} // namespace graph_tool

namespace boost
{
using namespace graph_tool;
template <class Graph>
struct graph_traits<half_edge_adaptor<Graph>>
    : public graph_traits<Graph>
{
    typedef half_edge_adaptor<Graph>::vertex_t vertex_descriptor;
    typedef half_edge_adaptor<Graph>::eiter_t out_edge_iterator;
    typedef half_edge_adaptor<Graph>::eiter_t in_edge_iterator;
    static constexpr vertex_descriptor null_vertex() { return vertex_descriptor(); }
};

template <class Graph>
auto num_vertices(const half_edge_adaptor<Graph>& g)
{
    return 2 * num_edges(g.base());
}

template <class Graph>
auto num_edges(const half_edge_adaptor<Graph>& g)
{
    return num_edges(g.base());
}

template <class Graph>
auto source(const typename half_edge_adaptor<Graph>::edge_t& e,
            const half_edge_adaptor<Graph>&)
{
    if (e.s == half_edge::_null || e.t == half_edge::_null)
        return half_edge(e, std::min(e.s, e.t), true);
    else
        return half_edge(e, e.s, true);
}

template <class Graph>
auto target(const typename half_edge_adaptor<Graph>::edge_t& e,
            const half_edge_adaptor<Graph>&)
{
    if (e.s == half_edge::_null || e.t == half_edge::_null)
    {
        auto v = half_edge(e, std::min(e.s, e.t), true);
        v.swap_endpoint();
        return v;
    }
    else
    {
        return half_edge(e, e.t, false);
    }
}

template <class Graph>
auto vertices(const half_edge_adaptor<Graph>& g)
{
    auto [begin, end] = edges(g);
    typedef decltype(begin) eiter_t;
    typedef typename half_edge_adaptor<Graph>::template viter_t<eiter_t> viter_t;
    return std::make_pair(viter_t(begin, g), viter_t(end, g));
}

template <class Graph>
auto edge(auto&& u, auto&& v, const half_edge_adaptor<Graph>&)
{
    if (u.get_edge() == v.get_edge() && u.is_source() != v.is_source())
    {
        if (u.is_source())
            return std::make_pair(u.get_edge(), true);

        if constexpr (is_directed_v<Graph>)
        {
            return std::make_pair(GraphInterface::edge_t(), false);
        }
        else
        {
            auto e = u.get_edge();
            std::swap(e.s, e.t);
            return std::make_pair(e, true);
        }
    }
    return std::make_pair(GraphInterface::edge_t(), false);
}

template <class Graph>
auto add_edge(const auto& u, const auto& v, half_edge_adaptor<Graph>& g)
{
    return add_edge(u.get_endpoint(), v.get_endpoint(), g.base());
}

template <class Graph>
void remove_edge(auto e, half_edge_adaptor<Graph>& g)
{
    if (e.s == half_edge::_null)
        e.s = e.t;
    if (e.t == half_edge::_null)
        e.t = e.s;
    return remove_edge(e, g.base());
}

template <class Graph>
auto out_edges(const auto& v, const half_edge_adaptor<Graph>& g)
{
    auto e = v.get_edge();

    if constexpr (is_directed_v<Graph>)
    {
        if (!v.is_source())
            e = GraphInterface::edge_t();
    }
    else
    {
        if (!v.is_self() && !v.is_source())
            std::swap(e.s, e.t);
    }

    if (e != GraphInterface::edge_t())
    {
        if (!(target(e, g) != v))
            std::println("foo 1 {} {} | {} {} | {} {}", e.s, e.t, target(e, g).get_source(), target(e, g).get_target(), v.get_source(), v.get_target());
        if (!(source(e, g) == v))
            std::println("foo 2 {} {} | {} {} | {} {}", e.s, e.t, source(e, g).get_source(), source(e, g).get_target(), v.get_source(), v.get_target());
    }

    return std::make_pair(typename half_edge_adaptor<Graph>::eiter_t(e),
                          typename half_edge_adaptor<Graph>::eiter_t());
}

template <class Graph>
auto in_edges(const auto& v, const half_edge_adaptor<Graph>& g)
{
    auto e = v.get_edge();

    if constexpr (is_directed_v<Graph>)
    {
        if (v.is_source())
            e = GraphInterface::edge_t();
        else if (v.is_self())
            std::swap(e.s, e.t);
    }
    else
    {
        e = GraphInterface::edge_t();
    }

    if (e != GraphInterface::edge_t())
    {
        if (!(target(e, g) == v))
            std::println("foo 3 {} {} | {} {} | {} {}", e.s, e.t, target(e, g).get_source(), target(e, g).get_target(), v.get_source(), v.get_target());
        if (!(source(e, g) != v))
            std::println("foo 4 {} {} | {} {} | {} {}", e.s, e.t, source(e, g).get_source(), source(e, g).get_target(), v.get_source(), v.get_target());
    }

    return std::make_pair(typename half_edge_adaptor<Graph>::eiter_t(e),
                          typename half_edge_adaptor<Graph>::eiter_t());
}

template <class Graph>
auto all_edges(const auto& v, const half_edge_adaptor<Graph>& g)
{
    std::println("all edges");
    auto e = v.get_edge();
    if (v.is_source())
        return out_edges(v, g);
    else
        return in_edges(v, g);
}

template <class Graph>
auto _all_edges_in(const auto& v, const half_edge_adaptor<Graph>&)
{
    std::println("all edges in");
    auto e = v.get_edge();
    if (v.is_source())
        std::swap(e.s, e.t);
    return std::make_pair(typename half_edge_adaptor<Graph>::eiter_t(e),
                          typename half_edge_adaptor<Graph>::eiter_t());
}

template <class Graph>
auto _all_edges_out(const auto& v, const half_edge_adaptor<Graph>&)
{
    std::println("all edges out");
    auto e = v.get_edge();
    if (!v.is_source())
        std::swap(e.s, e.t);
    return std::make_pair(typename half_edge_adaptor<Graph>::eiter_t(e),
                          typename half_edge_adaptor<Graph>::eiter_t());
}


template <class Graph>
auto all_neighbors(const auto& v, const half_edge_adaptor<Graph>&)
{
    typedef typename half_edge_adaptor<Graph>::aiter_t aiter_t;
    return std::make_pair(aiter_t(v), aiter_t());
}

template <class Graph>
auto out_neighbors(const auto& v, const half_edge_adaptor<Graph>&)
{
    typedef typename half_edge_adaptor<Graph>::aiter_t aiter_t;
    if constexpr (is_directed_v<Graph>)
    {
        if (!v.is_source())
            return std::make_pair(aiter_t(), aiter_t());
    }
    return std::make_pair(aiter_t(v), aiter_t());
}

template <class Graph>
auto in_neighbors(const auto& v, const half_edge_adaptor<Graph>&)
{
    typedef typename half_edge_adaptor<Graph>::aiter_t aiter_t;
    if constexpr (is_directed_v<Graph>)
    {
        if (v.is_source())
            return std::make_pair(aiter_t(), aiter_t());
        else
            return std::make_pair(aiter_t(v), aiter_t());
    }
    else
    {
        return std::make_pair(aiter_t(), aiter_t());
    }
}

struct odeg_t
{
    size_t _v;
    int64_t _k;
    bool _in;

    auto& operator+=(int64_t c)
    {
        _k += c;
        return *this;
    }

    auto& operator-=(int64_t c)
    {
        _k -= c;
        return *this;
    }

    auto& operator*=(int64_t c)
    {
        _k *= c;
        return *this;
    }

    auto operator+(int64_t c) const
    {
        odeg_t r(*this);
        r += c;
        return r;
    }

    auto operator-(int64_t c) const
    {
        odeg_t r(*this);
        r -= c;
        return r;
    }

    auto operator-() const
    {
        odeg_t r(*this);
        r._k *= -1;
        return r;
    }

    auto& operator=(int64_t c)
    {
        _k = c;
        return *this;
    }

    template <class T>
        requires std::is_scalar_v<T>
    bool operator>(T c) const
    {
        return _k > c;
    }

    template <class T>
        requires std::is_scalar_v<T>
    bool operator==(T c) const
    {
        return _k == c;
    }

    operator int64_t (void) const { return _k; }
};

template <class T>
    requires std::is_scalar_v<T>
decltype(auto) operator*(T c, odeg_t d)
{
    d *= c;
    return d;
}

template <class Graph>
int _in_degree(const auto& v, const half_edge_adaptor<Graph>&)
{
    if constexpr (is_directed_v<Graph>)
        return !v.is_source() ? 1 : 0;
    else
        return 0;
}

template <class Graph>
int _out_degree(const auto& v, const half_edge_adaptor<Graph>&)
{
    if constexpr (is_directed_v<Graph>)
        return v.is_source() ? 1 : 0;
    else
        return 1;
}

template <class Graph>
odeg_t in_degree(const auto& v, const half_edge_adaptor<Graph>& g)
{
    return odeg_t({v.get_endpoint(), _in_degree(v, g), true});
}

template <class Graph>
odeg_t out_degree(const auto& v, const half_edge_adaptor<Graph>& g)
{
    return odeg_t({v.get_endpoint(), _out_degree(v, g), false});
}

template <class Graph, class EWeight>
    requires (std::is_scalar_v<typename property_traits<EWeight>::value_type>)
odeg_t in_degree(const auto& v, const half_edge_adaptor<Graph>& g, const EWeight& ew)
{
    odeg_t k = {v.get_endpoint(), 0, true};
    for (auto e : in_edges_range(v, g))
        k += ew[e];
    return k;
}

template <class Graph, class EWeight>
    requires (std::is_scalar_v<typename property_traits<EWeight>::value_type>)
odeg_t out_degree(const auto& v, const half_edge_adaptor<Graph>& g, const EWeight& ew)
{
    odeg_t k = {v.get_endpoint(), 0, false};
    for (auto e : out_edges_range(v, g))
        k += ew[e];
    return k;
}

} // boost namespace

namespace graph_tool
{

template <class Graph, class F, class Except = void>
auto parallel_vertex_loop_no_spawn(const half_edge_adaptor<Graph>& g, F&& f,
                                   Except* ex=(void*)(nullptr))
{
    std::string err;
    bool skip = false;
    parallel_edge_loop_no_spawn
        (g.base(),
         [&](auto e)
         {
             half_edge v(e, source(e, g.base()));
             f(v);
             v.swap_endpoint();
             f(v);
         }, ex);
    return std::make_tuple(skip, err);
}

template <class Graph, class F, class Except = void>
void parallel_vertex_loop(half_edge_adaptor<Graph>& g, F&& f,
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
auto get_neighborhood(const half_edge& v, const half_edge_adaptor<Graph>& g)
{
    std::vector<half_edge> vs;
    gt_hash_set<GraphInterface::edge_t> selfs;
    for (auto e : out_edges_range(v.get_endpoint(), g.base()))
    {
        if constexpr (!is_directed_v<Graph>)
        {
            auto iter = selfs.find(e);
            if (iter != selfs.end())
                continue;
            else
                selfs.insert(e);
        }
        vs.emplace_back(e, v.get_endpoint(), true);
        vs.emplace_back(e, target(e, g.base()), false);
    }

    for (auto e : in_edges_range(v.get_endpoint(), g.base()))
    {
        vs.emplace_back(e, v.get_endpoint(), true);
        vs.emplace_back(e, source(e, g.base()), false);
    }

    return vs;
}

template <class Graph, class RNG>
auto random_neighbor(const half_edge& v, const half_edge_adaptor<Graph>& g, RNG& rng)
{
    return uniform_sample(get_neighborhood(v, g), rng);
}

template <class Prop>
class VPropAdaptor
    : public boost::put_get_helper
                <typename boost::property_traits<Prop>::value_type::value_type,
                 VPropAdaptor<Prop>>
{
public:
    typedef property_traits<Prop>::value_type::value_type value_type;
    typedef value_type& reference;
    typedef half_edge key_type;
    typedef boost::read_write_property_map_tag category;

    typedef VPropAdaptor<std::decay_t<decltype(boost::get_checked(Prop()))>> checked_t;
    typedef VPropAdaptor<std::decay_t<decltype(boost::get_unchecked(Prop()))>> unchecked_t;

    typedef std::conditional_t<std::is_same_v<VPropAdaptor, checked_t>,
                               unchecked_t, checked_t> other_t;

    VPropAdaptor() = default;
    VPropAdaptor(const VPropAdaptor&) = default;
    VPropAdaptor(const Prop& p) : _p(p) {}
    VPropAdaptor(const other_t& p)
        : VPropAdaptor(const_cast<other_t&>(p).flip_checked()) {}

    reference operator[](auto&& v)
    {
        auto& val = _p[v.get_edge()];

        if constexpr (_is_checked)
            val.resize(2);

        if (v.get_target() == half_edge::_null)
        {
            return val[0];
        }
        else if (v.get_source() == half_edge::_null)
        {
            return val[1];
        }

        if (v.get_endpoint() == std::min(v.get_source(), v.get_target()))
            return val[0];
        else
            return val[1];
    }

    auto& get_checked()
    {
        if constexpr (_is_checked)
        {
            return *this;
        }
        else
        {
            typedef typename Prop::checked_t cprop_t;
            return *reinterpret_cast<VPropAdaptor<cprop_t>*>(this);
        }
    }

    auto& get_unchecked()
    {
        if constexpr (_is_checked)
        {
            typedef typename Prop::unchecked_t cprop_t;
            return *reinterpret_cast<VPropAdaptor<cprop_t>*>(this);
        }
        else
        {
            return *this;
        }
    }

    auto& flip_checked()
    {
        if constexpr (_is_checked)
            return get_unchecked();
        else
            return get_checked();
    }

    void resize(size_t M, value_type fill = value_type())
    {
        std::vector<value_type> vfill = {fill, fill};
        _p.resize(M, vfill);
    }

    void reserve(size_t M, value_type fill = value_type())
    {
        std::vector<value_type> vfill = {fill, fill};
        _p.reserve(M, vfill);
    }

    void soft_reserve(size_t M, value_type fill = value_type())
    {
        std::vector<value_type> vfill = {fill, fill};
        _p.soft_reserve(M, vfill);
    }

    void shrink_to_fit()
    {
        _p.shrink_to_fit();
    }

    auto get_prop()
    {
        return _p;
    }

private:
    Prop _p;

    typedef std::decay_t<decltype(boost::get_checked(_p))> cprop_t;
    constexpr static bool _is_checked = std::is_same_v<cprop_t, Prop>;
};

} // namespace graph_tool

namespace boost
{
using namespace graph_tool;

template <class Prop>
struct uncheck<VPropAdaptor<Prop>>
{
    typedef typename VPropAdaptor<Prop>::unchecked_t type;

    auto operator()(auto&& p)
    {
        return p.get_unchecked();
    }
};
} // namespace boost


namespace std
{
using namespace graph_tool;
template <>
struct formatter<half_edge>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const half_edge& v, format_context& ctx) const
    {
        string temp;
        format_to(back_inserter(temp), "{} {} {}", v.get_source(),
                  v.get_target(), v.get_endpoint());
        return formatter<string_view>().format(temp, ctx);
    }
};
}


#endif // HALF_EDGE_ADAPTOR
