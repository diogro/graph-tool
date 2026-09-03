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

#ifndef GRAPH_BLOCKMODEL_WEIGHTED__TYPES_HH
#define GRAPH_BLOCKMODEL_WEIGHTED__TYPES_HH

#include "config.h"

#include <cstdint>
#include <type_traits>
#include <utility>
#include <limits>
#include <vector>
#include <iterator>
#include <boost/container/small_vector.hpp>

namespace graph_tool
{

enum weight_type
{
    NONE,
    COUNT,
    REAL_EXPONENTIAL,
    REAL_NORMAL,
    DISCRETE_GEOMETRIC,
    DISCRETE_POISSON,
    DISCRETE_BINOMIAL
};

// pre-allocated covariates

#define WVEC_N 11

template <typename T>
concept WeightCounts = requires(T t)
{
    { t.visit_weights([](auto, auto){}) };
};

struct WeightContext
{
    WeightContext() = default;

    WeightContext(const auto& types)
    {
        for (size_t i = 0; i < types.size(); ++i)
        {
            _types.push_back(weight_type(types[i]));
            if (types[i] == weight_type::REAL_NORMAL)
                _normals.push_back(i);
        }
    }

    std::vector<weight_type> _types;
    std::vector<size_t> _normals;
};

static const WeightContext __global_wc;

template <class Key, class Prop>
struct WeightCountRef
{
    WeightCountRef(const Key& key, Prop& p)
        : _key(key), _p(p) {}

    template <WeightCounts C>
    void resize(const C& c)
    {
        if (c.get_wc() == &__global_wc)
            *this = 0;
    }

    decltype(auto) visit_count(auto&& f)
    {
        return f(_p._count[_key]);
    }

    decltype(auto) visit_count(auto&& f) const
    {
        return const_cast<WeightCountRef*>(this)->visit_count([&](const auto& a) { return f(a); });
    }

    void visit_weights(auto&& f)
    {
        for (size_t i = 0; i < _p._weights.size(); ++i)
            f(i, _p._weights[i][_key]);
    }

    void visit_weights(auto&& f) const
    {
        const_cast<WeightCountRef*>(this)->visit_weights([&](auto i, const auto& a) { f(i, a); });
    }

    decltype(auto) visit_weight(size_t i, auto&& f)
    {
        return f(_p._weights[i][_key]);
    }

    template <class F>
    decltype(auto) visit_weight(size_t i, F&& f) const
    {
        return const_cast<WeightCountRef*>(this)->visit_weight(i, [&](const auto& a) { return f(a); });
    }

    void visit_devs(auto&& f)
    {
        for (auto i : _p._wc._normals)
            f(i, _p._devs[i][_key]);
    }

    template <class F>
    void visit_devs(F&& f) const
    {
        const_cast<WeightCountRef*>(this)->visit_devs([&](auto i, const auto& a) { f(i, a); });
    }

    decltype(auto) visit_dev(size_t i, auto&& f)
    {
        return f(_p._devs[i][_key]);
    }

    template <class F>
    decltype(auto) visit_dev(size_t i, F&& f) const
    {
        return const_cast<WeightCountRef*>(this)->visit_dev(i, [&](const auto& a) { return f(a); });
    }

    const auto& get_types() const
    {
        return _p._wc._types;
    }

    const auto& get_normals() const
    {
        return _p._wc._normals;
    }

    const auto* get_wc() const
    {
        return &_p._wc;
    }

    template <class C>
    WeightCountRef& operator=(const C& c)
    {
        set_weight(*this, c);
        return *this;
    }

    typedef typename Prop::cprop_t::value_type cval_t;
    typedef typename Prop::prop_t::value_type wval_t;

    operator cval_t (void) const { return _p._count[_key]; }

    const Key& _key;
    Prop& _p;
};

template <class WVal, class CVal, size_t N>
struct WeightCount
{
    typedef WVal wval_t;
    typedef CVal cval_t;

    WeightCount()
        : _wc(&__global_wc)
    {}

    WeightCount(const WeightCount& c) = default;

    template <WeightCounts C>
    WeightCount(const C& c)
        : _wc(c.get_wc())
    {
        *this = c;
    }

    WeightCount(WeightCount&& c) noexcept
        : _count(std::exchange(c._count, 0)), _weights(std::move(c._weights)),
          _devs(std::move(c._devs)), _wc(c._wc)
    {}

    WeightCount& operator=(const WeightCount& c) = default;

    template <class C>
    WeightCount& operator=(const C& c)
    {
        set_weight(*this, c);
        return *this;
    }

    WeightCount& operator=(WeightCount&& c) noexcept = default;

    operator CVal (void) const { return _count; }

    template <WeightCounts C>
    void resize(const C& c)
    {
        resize(*c.get_wc());
    }

    void resize(const WeightContext& wc)
    {
        if (_wc == &__global_wc || wc._types.size() != _wc->_types.size())
        {
            _wc = &wc;
            _weights.resize(_wc->_types.size());
            if (!_wc->_normals.empty())
                _devs.resize(_wc->_types.size());
        }
    }

    decltype(auto) visit_count(auto&& f)
    {
        return f(_count);
    }

    template <class F>
    decltype(auto) visit_count(F&& f) const
    {
        return const_cast<WeightCount*>(this)->visit_count([&](const auto& a) { return f(a); });
    }

    void visit_weights(auto&& f)
    {
        for (size_t i = 0; i < _weights.size(); ++i)
            f(i, _weights[i]);
    }

    template <class F>
    void visit_weights(F&& f) const
    {
        const_cast<WeightCount*>(this)->visit_weights([&](auto i, const auto& a) { f(i, a); });
    }

    decltype(auto) visit_weight(size_t i, auto&& f)
    {
        return f(_weights[i]);
    }

    template <class F>
    decltype(auto) visit_weight(size_t i, F&& f) const
    {
        return const_cast<WeightCount*>(this)->visit_weight(i, [&](const auto& a) { return f(a); });
    }

    void visit_devs(auto&& f)
    {
        for (auto i : _wc->_normals)
            f(i, _devs[i]);
    }

    template <class F>
    void visit_devs(F&& f) const
    {
        const_cast<WeightCount*>(this)->visit_devs([&](auto i, const auto& a) { f(i, a); });
    }

    decltype(auto) visit_dev(size_t i, auto&& f)
    {
        return f(_devs[i]);
    }

    template <class F>
    decltype(auto) visit_dev(size_t i, F&& f) const
    {
        return const_cast<WeightCount*>(this)->visit_dev(i, [&](const auto& a) { return f(a); });
    }

    const auto& get_types() const
    {
        return _wc->_types;
    }

    const auto& get_normals() const
    {
        return _wc->_normals;
    }

    const auto* get_wc() const
    {
        return _wc;
    }

    CVal _count = 0;
    typedef std::conditional_t<(N > 0),
                               boost::container::small_vector<WVal, N>,
                               std::vector<WVal>> vec_t;
    vec_t _weights;
    vec_t _devs;
    const WeightContext* _wc = nullptr;
};

template <WeightCounts X>
decltype(auto) get_count(const X& x)
{
    return x.visit_count([](auto& w) { return w; });
}

template <WeightCounts X>
decltype(auto) get_weight(const X& x, size_t i)
{
    return x.visit_weight(i, [](auto& w) { return w; });
}

template <WeightCounts X>
decltype(auto) get_dev(const X& x, size_t i)
{
    return x.visit_dev(i, [](auto& w) { return w; });
}

template <WeightCounts X>
void set_count(X&& x,auto&& y)
{
    return x.visit_count([&](auto& w) { w = y; });
}

template <WeightCounts X>
void set_weight(X&& x, size_t i, auto&& y)
{
    return x.visit_weight(i, [&](auto& w) { w = y; });
}

template <WeightCounts X>
void set_dev(X&& x, size_t i, auto&& y)
{
    return x.visit_dev(i, [&](auto& w) { w = y; });
}

template <WeightCounts X, class F>
decltype(auto) gen_op(X&& x, F&& op)
{
    x.visit_count
        ([&](auto& cx)
         {
             op(cx);
         });

    x.visit_weights
        ([&](auto, auto& wx)
         {
             op(wx);
         });

    x.visit_devs
        ([&](auto, auto& dx)
         {
             op(dx);
         });

    return std::forward<X>(x);
}

template <WeightCounts X, WeightCounts Y, class F>
decltype(auto) gen_bin_op(X&& x, const Y& y, F&& op)
{
    if constexpr (!std::is_const_v<std::remove_reference_t<X>>)
        x.resize(y);

    y.visit_count
        ([&](auto& cy)
         {
             x.visit_count
                 ([&](auto&& cx)
                  {
                      op(cx, cy);
                  });
         });

    y.visit_weights
        ([&](auto i, auto& wy)
         {
             x.visit_weight
                 (i,
                  [&](auto&& wx)
                  {
                      op(wx, wy);
                  });
             });

    y.visit_devs
        ([&](auto i, auto& dy)
         {
             x.visit_dev
                 (i,
                  [&](auto&& dx)
                  {
                      op(dx, dy);
                  });
             });

    return std::forward<X>(x);
}


template <bool enable=true, WeightCounts X, WeightCounts Y>
decltype(auto) atomic_inc(X&& x, const Y& y)
{
    return gen_bin_op(std::forward<X>(x), y,
                      [](auto&& a, auto&& b){ atomic_inc<enable>(a, b); });
}

template <WeightCounts X, WeightCounts Y>
decltype(auto) operator+=(X&& x, const Y& y)
{
    return atomic_inc<false>(std::forward<X>(x), y);
}

template <bool enable=true, WeightCounts X, WeightCounts Y>
decltype(auto) atomic_dec(X&& x, const Y& y)
{
    return gen_bin_op(std::forward<X>(x), y,
                      [](auto&& a, auto&& b){ atomic_dec<enable>(a, b); });
}

template <WeightCounts X, WeightCounts Y>
decltype(auto) operator-=(X&& x, const Y& y)
{
    return atomic_dec<false>(std::forward<X>(x), y);
}

template <bool enable=true, WeightCounts X, WeightCounts Y>
decltype(auto) atomic_div(X&& x, const Y& y)
{
    return gen_bin_op(std::forward<X>(x), y,
                      [](auto&& a, auto&& b){ atomic_div<enable>(a, b); });
}

template <WeightCounts X, WeightCounts Y>
decltype(auto) operator/=(X&& x, const Y& y)
{
    return atomic_div<false>(std::forward<X>(x), y);
}

template <bool enable=true, WeightCounts X, WeightCounts Y>
decltype(auto) atomic_prod(X&& x, const Y& y)
{
    return gen_bin_op(std::forward<X>(x), y,
                      [](auto&& a, auto&& b){ atomic_prod<enable>(a, b); });
}

template <WeightCounts X, WeightCounts Y>
decltype(auto) operator*=(X&& x, const Y& y)
{
    return atomic_prod<false>(std::forward<X>(x), y);
}

template <WeightCounts X, WeightCounts Y>
void set_weight(X& x, const Y& y)
{
    gen_bin_op(x, y, [](auto&& a, auto&& b){ a = b; });
}

template <WeightCounts X, class Y>
    requires std::is_scalar_v<Y>
void set_weight(X& x, Y y)
{
    gen_op(x, [&](auto&& a){ a = y; });
}

template <WeightCounts X, WeightCounts Y>
bool operator==(const X& x, const Y& y)
{
    bool eq = true;
    gen_bin_op(x, y, [&](auto&& a, auto&& b){ eq &= (a == b); });
    return eq;
}

template <WeightCounts X, class Y>
    requires std::is_scalar_v<Y>
bool operator==(const X& x, Y y)
{
    bool eq = true;
    gen_op(x, [&](auto&& a){ eq &= (a == y); });
    return eq;
}

template <WeightCounts X, WeightCounts Y>
auto operator+(X x, const Y& y)
{
    x += y;
    return x;
}

template <WeightCounts X, WeightCounts Y>
auto operator-(X x, const Y& y)
{
    x -= y;
    return x;
}

template <bool enable=true, WeightCounts X, class T>
    requires std::is_scalar_v<T>
decltype(auto) atomic_div(X&& x, const T& c)
{
    return gen_op(x, [&](auto&& y) { atomic_div<enable>(y, c); });
}

template <WeightCounts X, class T>
   requires std::is_scalar_v<T>
decltype(auto) operator/=(X& x, T c)
{
    return atomic_div<true>(x, c);
}

template <bool enable=true, WeightCounts X, class T>
    requires std::is_scalar_v<T>
auto& atomic_prod(X& x, T c)
{
    return gen_op(x, [&](auto&& y) { atomic_prod<enable>(y, c); });
}

template <WeightCounts X, class T>
    requires std::is_scalar_v<T>
auto& operator*=(X& x, T c)
{
    return gen_op(x, [&](auto&& y) { y *= c; });
}

template <WeightCounts X>
auto operator-(X x)
{
    x *= -1;
    return x;
}

template <class CProp, class Props, size_t N = WVEC_N>
class WeightsPropMap
{
public:
    template <class Types>
    WeightsPropMap(const CProp& count, const Props& ps, const Props& pds,
                   const Types& types)
        : _count(count), _weights(ps), _devs(pds), _wc(types)
    {
    }

    WeightsPropMap() = delete;

    typedef CProp cprop_t;
    typedef typename Props::value_type prop_t;

    typedef typename boost::property_traits<CProp>::key_type key_type;
    typedef WeightCount<typename prop_t::value_type,
                        typename cprop_t::value_type, N> value_type;
    typedef WeightCountRef<key_type, WeightsPropMap> reference;
    typedef boost::read_write_property_map_tag category;

    template <class K>
    reference operator[](K&& key) const
    {
        return reference(key, *const_cast<WeightsPropMap*>(this));
    }

    auto& get_checked()
    {
        typedef std::decay_t<decltype(boost::get_checked(_count))> c_prop_t;
        if constexpr (std::is_same_v<c_prop_t, CProp>)
        {
            return *this;
        }
        else
        {
            typedef typename CProp::checked_t c_cprop_t;
            typedef typename Props::value_type::checked_t wprop_t;
            return *reinterpret_cast<WeightsPropMap<c_cprop_t, std::vector<wprop_t>>*>(this);
        }
    }

    void sync_size()
    {
        soft_reserve(_count.get_storage().size());
    }

    void reserve(size_t M)
    {
        _count.reserve(M);
        for (auto& p : _weights)
            p.reserve(M);
        for (auto i : _wc._normals)
            _devs[i].reserve(M);
    }

    void soft_reserve(size_t M)
    {
        _count.soft_reserve(M);
        for (auto& p : _weights)
            p.soft_reserve(M);
        for (auto i : _wc._normals)
            _devs[i].soft_reserve(M);
    }

    void resize(size_t M)
    {
        _count.resize(M);
        for (auto& p : _weights)
            p.resize(M);
        for (auto i : _wc._normals)
            _devs[i].resize(M);
    }

    void shrink_to_fit()
    {
        _count.shrink_to_fit();
        for (auto& p : _weights)
            p.shrink_to_fit();
        for (auto i : _wc._normals)
            _devs[i].shrink_to_fit();
    }

//private:
    friend reference;
    friend value_type;

    CProp _count;
    Props _weights;
    Props _devs;
    const WeightContext _wc;
};

template <class CProp, class Props, size_t N>
auto get(const WeightsPropMap<CProp, Props, N>& p, auto&& key)
{
    return p[key];
}

template <class CProp, class Props, size_t N>
void put(WeightsPropMap<CProp, Props, N>& p, auto&& key, auto&& val)
{
    p[key] = val;
}

template <size_t N, class CProp, class Props, class Types>
auto make_weights_prop(const CProp& count, const Props& ps, const Props& pds,
                       const Types& types)
{
    return WeightsPropMap<CProp, Props, N>(count, ps, pds, types);
}

} // namespace graph_tool

namespace std
{
using namespace graph_tool;

template <WeightCounts W>
struct formatter<W>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const W& c, format_context& ctx) const
    {
        string temp;
        c.visit_count
            ([&](auto x)
             {
                 format_to(back_inserter(temp), "(c:{}, ", x);
             });
        format_to(back_inserter(temp), "w:");
        c.visit_weights
            ([&](auto, auto& x)
             {
                 format_to(back_inserter(temp), "{}, ", x);
             });
        format_to(back_inserter(temp), " d: ");
        c.visit_devs
            ([&](auto, auto& x)
             {
                 format_to(back_inserter(temp), "{}, ", x);
             });
        format_to(back_inserter(temp), ")");
        return formatter<string_view>().format(temp, ctx);
    }
};

}

#endif // GRAPH_BLOCKMODEL_WEIGHTED__TYPES_HH
