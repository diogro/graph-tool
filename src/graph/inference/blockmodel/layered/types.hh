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

#ifndef GRAPH_LAYERED_BLOCKMODEL_TYPES_HH
#define GRAPH_LAYERED_BLOCKMODEL_TYPES_HH

#include "config.h"

#include <cstdint>
#include <type_traits>
#include <utility>
#include <limits>
#include <vector>
#include <flat_map>
#include <shared_mutex>
#include <boost/container/small_vector.hpp>

#include "../../../parallel_util.hh"

#include "../entropy.hh"
#include "vec_ref.hh"


// TODO: experiment with regular vector and dedicated memory pool!

// pre-allocated layers

#define LVEC_N 10

// layer label type

typedef int64_t lval_t;

namespace graph_tool
{

template <class CVal>
using lmap_t = std::flat_map<lval_t, CVal, std::less<lval_t>,
                             boost::container::small_vector<lval_t, LVEC_N>,
                             boost::container::small_vector<CVal, LVEC_N>>;

template <class CVal>
using rlmap_t = std::flat_map<lval_t, CVal, std::less<lval_t>,
                              VecRef<std::vector<lval_t>>,
                              VecRef<std::vector<CVal>>>;

auto lc_get(const auto& x, auto l)
{
    auto iter = x.find(l);
    if (iter == x.end())
        return std::decay_t<decltype(iter->second)>{};
    return iter->second;
}

void lc_set(auto& x, auto l, const auto& val)
{
    if (val == 0)
        x.erase(l);
    else
        x[l] = val;
}

template <bool enable_lock=true, class Lock>
bool lc_visit(auto& h, auto&& l, auto&& f, Lock&& lock)
{
    constexpr bool enable =
        enable_lock && !std::is_same_v<std::decay_t<Lock>, dummy_mutex>;

    auto iter = h.find(l);

    if (iter != h.end())
    {
        f(*iter);

        auto stamp = get_canary(lock);

        if (iter->second == 0)
        {
            upgrade_lock<{.la={.enable=enable}}>
                ([&]()
                 {
                     if constexpr (!enable)
                     {
                         h.erase(iter);
                     }
                     else
                     {
                         auto& canary = get_canary(lock);
                         if (stamp == canary) [[likely]]
                         {
                             if (iter->second == 0) // race still possible
                                 h.erase(iter);
                         }
                         else                 [[unlikely]]
                         {
                             // lost race; search again
                             iter = h.find(l);
                             if (iter != h.end() && iter->second == 0)
                                 h.erase(iter);
                         }
                         canary++;
                     }
                 }, lock);
        }
        return true;
    }
    return false;
};


template <bool enable_lock=true, class Lock>
void lc_insert_or_visit(auto& h, const auto& x, auto&& f, Lock&& lock)
{
    constexpr bool enable =
        enable_lock && !std::is_same_v<std::decay_t<Lock>, dummy_mutex>;

    if (lc_visit<enable>(h, x.first, f, lock))
        return;

    assert(x.second != 0);

    auto stamp = get_canary(lock);

    upgrade_lock<{.la={.enable=enable}}>
        ([&]()
         {
             if constexpr (!enable)
             {
                 h.insert(x);
             }
             else
             {
                 auto& canary = get_canary(lock);
                 if (stamp == canary) [[likely]]
                 {
                     h.insert(x);
                 }
                 else                 [[unlikely]]
                 {
                     // lost race; search again
                     auto iter = h.find(x.first);
                     if (iter != h.end())  [[unlikely]]
                         f(*iter);
                     else                  [[likely]]
                         h.insert(x);
                 }
                 canary++;
             }
         }, lock);
};

template <bool enable_lock=true, class T1, class T2, class Lock>
void lc_sum(T1& to, const T2& from, Lock&& lock)
{
    constexpr bool enable =
        enable_lock && !std::is_same_v<std::decay_t<Lock>, dummy_mutex>;

    if constexpr (std::is_same_v<T1, T2>)
    {
        if (&to == &from)
        {
            for (auto&& [l, c] : to)
                atomic_inc<enable>(c, c);
            return;
        }
    }

    for (const auto& x : from)
        lc_insert_or_visit<enable>
            (to, x,
             [&](auto&& y)
             {
                 atomic_inc<enable>(y.second, x.second);
             }, lock);
}

template <bool enable_lock=true, class T1, class T2, class Lock>
void lc_sub(T1& to, const T2& from, Lock&& lock)
{
    constexpr bool enable =
        enable_lock && !std::is_same_v<std::decay_t<Lock>, dummy_mutex>;

    if constexpr (std::is_same_v<T1, T2>)
    {
        if (&to == &from)
        {
            upgrade_lock<{.la={.enable=enable}}>
                ([&]()
                 {
                     to.clear();
                 }, lock);
            return;
        }
    }

    for (const auto& [l, c] : from)
    {
        lc_insert_or_visit<enable>
            (to, std::make_pair(l, -c),
             [&](auto&& y)
             {
                 atomic_dec<enable>(y.second, c);
             }, lock);
    }
}

template <bool enable_lock=true, class T1, class T2, class Lock>
void lc_div(T1& to, const T2& from, Lock&& lock)
{
    constexpr bool enable =
        enable_lock && !std::is_same_v<std::decay_t<Lock>, dummy_mutex>;

    if constexpr (std::is_same_v<T1, T2>)
    {
        if (&to == &from)
        {
            for (auto&& [l, c] : to)
            {
                assert(c != 0);
                atomic_div<enable>(c, c);
            }
            return;
        }
    }

    for (const auto& [l, c] : from)
        lc_visit<enable>
            (to, l,
             [&](auto&& y)
             {
                 assert(c != 0);
                 atomic_div<enable>(y.second, c);
             }, lock);
}

bool lc_equal(const auto& a, const auto& b)
{
    if (a.size() != b.size())
        return false;
    return std::ranges::all_of(std::views::zip(a, b),
                               [](const auto& x)
                               { return get<0>(x) == get<1>(x); });
}

bool lc_cmp(const auto& a, const auto& b, auto&& cmp)
{
    return (std::ranges::all_of(a,
                                [&](const auto& x)
                                {
                                    const auto& [l, c] = x;
                                    return cmp(c, lc_get(b, l));
                                }) &&
            std::ranges::all_of(b,
                                [&](const auto& x)
                                {
                                    const auto& [l, c] = x;
                                    return cmp(lc_get(a, l), c);
                                }));
}

auto lc_sum(const auto& h)
{
    typedef std::decay_t<decltype(h.begin()->second)> c_t;
    c_t c = c_t();
    for (const auto& x : h)
        c += x.second;
    return c;
}

template <typename T>
concept LayerCounts = requires(T t)
{
    { t.layer_dispatch([](auto){ return 0.; }) };
};

template <class CVal, bool ref, bool layered>
struct LayerCount
{
    typedef CVal cval_t;

    LayerCount()
    {
        if constexpr (!ref)
            _concurrent = false;
    }

    LayerCount(const LayerCount& c)
        : _concurrent(c._concurrent), _h(c._h)
    {}

    template <class Prop, class Key>
    LayerCount(Prop& p, const Key& key)
        : _concurrent(p._concurrent), _mutex(*p._mutex[key]),
          _lock(_concurrent ? std::shared_lock<shared_canary_mutex>(_mutex) :
                              std::shared_lock<shared_canary_mutex>()),
          _h(std::sorted_unique_t(), vec_ref(p._ec[key]), vec_ref(p._w[key]))
    {}

    template <class CV, bool r, bool l>
       requires (!std::is_same_v<LayerCount, LayerCount<CV, r, l>>)
    LayerCount(const LayerCount<CV, r, l>& c)
    {
        set_concurrent(false);
        *this = c;
    }

    template <bool disable_lock=false, bool unique=false>
    void dispatch(auto&& f)
    {
        if constexpr (disable_lock)
        {
            f(_h, dummy_mutex());
        }
        else
        {
            if (_concurrent)
            {
                if constexpr (ref)
                {
                    f(_h, _lock);
                }
                else
                {
                    if constexpr (unique)
                        do_ulock<{.enable=!disable_lock, .pass_lock=true}>
                            ([&](auto&& lock){ f(_h, lock); }, _mutex);
                    else
                        do_slock<{.enable=!disable_lock, .pass_lock=true}>
                            ([&](auto&& lock){ f(_h, lock); }, _mutex);
                }
            }
            else
            {
                f(_h, dummy_mutex());
            }
        }
    }

    template <bool disable_lock=false, bool unique=false>
    void cdispatch(auto&& f) const
    {
        typedef std::decay_t<decltype(f)> f_t;
        const_cast<LayerCount*>(this)->dispatch<disable_lock,unique>(std::forward<f_t>(f));
    }

    auto layer_dispatch(auto&& f) const
    {
        decltype(f(cval_t())) ret = 0;
        cdispatch<true>([&](auto&& h, auto)
                        {
                            for (auto&& [l, c] : h)
                                ret += f(c);
                        });
        return ret;
    }

    template <class LC>
    void set(const LC& c)
    {
        dispatch<false, true>
            ([&](auto&& h, auto&&)
             {
                 c.template cdispatch<true>
                     ([&](auto&& hc, auto)
                      {
                          if constexpr (std::is_same_v<std::decay_t<decltype(h)>,
                                        std::decay_t<decltype(hc)>>)
                          {
                              h = hc;
                          }
                          else
                          {
                              h.clear();
                              for (const auto& x : hc)
                                  h.insert(h.end(), x);
                          }
                      });
             });
    }

    LayerCount& operator=(const LayerCount& c)
    {
        set(c);
        return *this;
    }

    template <LayerCounts LC>
    LayerCount& operator=(const LC& c)
    {
        set(c);
        return *this;
    }

    template <class C>
        requires (std::is_same_v<C, CVal> || std::is_scalar_v<C>)
    LayerCount& operator=(C c)
    {
        dispatch<false, true>
            ([&](auto&& hc, auto&&)
             {
                 if (c == 0)
                 {
                     hc.clear();
                 }
                 else
                 {
                     for (auto&& x : hc)
                         x.second = c;
                 }
             });
        return *this;
    }

    void clean()
    {
        std::erase_if(_h,
                      [&](const auto& lc)
                      {
                          return lc.second == 0;
                      });
    }

    operator cval_t () const
    {
        return sum();
    }

    operator std::conditional_t<std::is_scalar_v<cval_t>, std::string, int64_t> () const
    {
        return int64_t(sum());
    }

    auto sum() const
    {
        cval_t ret = cval_t();
        cdispatch([&](auto&& hc, auto&&){ ret = lc_sum(hc); });
        return ret;
    }

    auto get(auto l) const
    {
        return lc_get(_h, l);
    }

    void set(auto l, const auto& c)
    {
        lc_set(_h, l, c);
    }

    void set_concurrent(bool enable)
    {
        static_assert(!ref);
        _concurrent = enable;
    }

    // for non-scalar counts
    template <class T>
    void resize(const T& n)
    {
        cdispatch<true>([&](auto&& h, auto)
                        {
                            for (auto&& [l, c] : h)
                                c.resize(n);
                        });
    }

    void print() //useful for debugging
    {
        std::println("{}", *this);
    }

    std::conditional_t<ref, bool&, bool> _concurrent;
    std::conditional_t<ref, shared_canary_mutex&, shared_canary_mutex> _mutex;
    std::conditional_t<ref, std::shared_lock<shared_canary_mutex>, dummy_mutex> _lock;

    typedef std::conditional_t<ref, rlmap_t<CVal>, lmap_t<CVal>> h_t;
    h_t _h;

    constexpr static bool _layered = layered;
    constexpr static bool _ref = ref;
};

template <class CVal>
struct VLayerCount:
    public LayerCount<CVal, false, true>
{
    typedef LayerCount<CVal, false, true> base_t;

    VLayerCount() : _total{} {}

    template <class T>
    VLayerCount(const VLayerCount<T>& c)
        : base_t(c), _total(c._total) {}

    template <class T, bool ref, bool l>
    explicit VLayerCount(const LayerCount<T, ref, l>& c)
        : base_t(c), _total{} {}

    template <class C>
    VLayerCount& operator=(const C& c)
    {
        static_cast<base_t&>(*this) = c;
        if constexpr (is_specialization_of_v<C, VLayerCount>)
            _total = c._total;
        else if constexpr (std::is_same_v<C, CVal> || std::is_scalar_v<C>)
            _total = c;
        else
            static_assert(false, "invalid operation");
        return *this;
    }

    template <class C>
    VLayerCount& operator+=(const C& c)
    {
        static_cast<base_t&>(*this) += c;
        if constexpr (is_specialization_of_v<C, VLayerCount>)
            atomic_inc(_total, c._total);
        else
            static_assert(false, "invalid operation");
        assert(_total < 10000000);
        return *this;
    }

    template <class C>
    VLayerCount& operator-=(const C& c)
    {
        static_cast<base_t&>(*this) -= c;
        if constexpr (is_specialization_of_v<C, VLayerCount>)
            atomic_dec(_total, c._total);
        else
            static_assert(false, "invalid operation");
        assert(_total < 10000000);
        return *this;
    }

    template <class CV>
    bool operator==(const VLayerCount<CV>& c) const
    {
        return _total == c._total && (static_cast<const base_t&>(*this) ==
                                      static_cast<const typename VLayerCount<CV>::base_t&>(c));
    }

    template <class C>
        requires std::is_scalar_v<C>
    bool operator==(const C& c) const
    {
        return _total == c && (static_cast<const base_t&>(*this) == c);
    }

    operator CVal (void) const
    {
        return _total;
    }

    CVal _total;
};

template <LayerCounts LC, LayerCounts C>
    requires (!is_specialization_of_v<LC, VLayerCount> &&
              !is_specialization_of_v<C, VLayerCount>)
bool operator==(const LC& x, const C& c)
{
    bool equal = false;
    x.cdispatch([&](auto&& hc, auto&&)
                {
                    c.template cdispatch<true>
                        ([&](auto&& c, auto)
                         {
                             equal = lc_equal(hc, c);
                         });
                });
    return equal;
}

template <LayerCounts LC, class C>
    requires (std::is_scalar_v<C> &&
              !is_specialization_of_v<LC, VLayerCount>)
bool operator==(const LC& x, C c)
{
    bool equal = true;
    x.cdispatch([&](auto&& hc, auto&&)
                {
                    for (auto&& x : hc)
                    {
                        if (quiet_not_equal(x.second, c))
                        {
                            equal = false;
                            break;
                        }
                    }
                });
    return equal;
}

// component-wise !!
template <LayerCounts LC, class C>
    requires std::is_scalar_v<C>
auto operator>(const LC& x, C c)
{
    LayerCount<bool, false, x._layered> ret;

    x.cdispatch([&](auto&& hc, auto&&)
                {
                    for (auto&& x : hc)
                    {
                        if (quiet_greater(x.second, c))
                            ret._h[x.first] = true;
                    }
                });

    return ret;
}

// component-wise !!
template <LayerCounts LC, class C>
    requires std::is_scalar_v<C>
auto operator<(const LC& x, C c)
{
    LayerCount<bool, false, x._layered> ret;

    x.cdispatch([&](auto&& hc, auto&&)
                {
                    for (auto&& x : hc)
                    {
                        if (quiet_less(x.second, c))
                            ret._h[x.first] = true;
                    }
                });

    return ret;
}

// component-wise !!
template <LayerCounts LC, class C>
    requires std::is_scalar_v<C>
auto operator>=(const LC& x, C c)
{
    LayerCount<bool, false, x._layered> ret;

    x.cdispatch([&](auto&& hc, auto&&)
                {
                    for (auto&& x : hc)
                    {
                        if (quiet_greater_equal(x.second, c))
                            ret._h[x.first] = true;
                    }
                });

    return ret;
}

// component-wise !!
template <LayerCounts LC, class C>
    requires std::is_scalar_v<C>
auto operator<=(const LC& x, C c)
{
    LayerCount<bool, false, x._layered> ret;

    x.cdispatch([&](auto&& hc, auto&&)
                {
                    for (auto&& x : hc)
                    {
                        if (quiet_less_equal(x.second, c))
                            ret._h[x.first] = true;
                    }
                });

    return ret;
}

template <bool enable=true, LayerCounts T1, LayerCounts T2>
decltype(auto) atomic_inc(T1&& x, T2&& c)
{
    x.template dispatch<!enable>
        ([&](auto&& h, auto&& lock)
         {
             c.template cdispatch<true>
                 ([&](auto&& hc, auto)
                  {
                      lc_sum<enable>(h, hc, lock);
                  });
         });
    return std::forward<T1>(x);
}

template <LayerCounts LC, LayerCounts C>
decltype(auto) operator+=(LC&& x, const C& c)
{
    return atomic_inc<false>(x, c);
}

template <bool enable=true, LayerCounts T1, LayerCounts T2>
decltype(auto) atomic_dec(T1&& x, T2&& c)
{
    x.template dispatch<!enable>
        ([&](auto&& h, auto&& lock)
         {
             c.template cdispatch<true>
                 ([&](auto&& hc, auto)
                  {
                      lc_sub<enable>(h, hc, lock);
                  });
         });
    return std::forward<T1>(x);
}

template <LayerCounts LC, LayerCounts C>
decltype(auto) operator-=(LC&& x, const C& c)
{
    return atomic_dec<false>(x, c);
}

template <bool enable=true, LayerCounts T1, LayerCounts T2>
decltype(auto) atomic_div(T1&& x, T2&& c)
{
    x.template dispatch<!enable>
        ([&](auto&& h, auto&& lock)
         {
             c.template cdispatch<true>
                 ([&](auto&& hc, auto)
                  {
                      lc_div<enable>(h, hc, lock);
                  });
         });
    return std::forward<T1>(x);
}

template <LayerCounts LC, LayerCounts C>
decltype(auto) operator/=(LC&& x, const C& c)
{
    return atomic_div<false>(x, c);
}

template <LayerCounts LC, LayerCounts C>
auto operator+(LC x, const C& c)
{
    x += c;
    return x;
}

template <LayerCounts LC, LayerCounts C>
LC operator-(LC x, const C& c)
{
    x -= c;
    return x;
}

template <LayerCounts LC, class C>
    requires std::is_scalar_v<C>
decltype(auto) operator/=(LC&& x, C c)
{
    x.dispatch([&](auto&& hc, auto&&)
               {
                   for (auto&& x : hc)
                       atomic_div(x.second, c);
               });
    return std::forward<LC>(x);
}

template <LayerCounts LC, class C>
    requires std::is_scalar_v<C>
decltype(auto) operator*=(LC&& x, C c)
{
    x.dispatch([&](auto&& hc, auto&&)
               {
                   for (auto&& x : hc)
                       atomic_prod(x.second, c);
               });
    x.clean();
    return std::forward<LC>(x);
}

template <LayerCounts LC, LayerCounts C>
decltype(auto) operator*=(LC&& x, const C& c)
{
    x.dispatch([&](auto&& h, auto&&)
               {
                   c.cdispatch([&](auto&& hc, auto&&)
                               {
                                   std::erase_if(h,
                                                 [&](const auto& lc)
                                                 {
                                                     return lc_get(hc, get<0>(lc)) == 0;
                                                 });

                                   for (auto&& [l, c] : h)
                                       c *= lc_get(hc, l);
                               });
               });
    return std::forward<LC>(x);
}

template <LayerCounts LC, class C>
LC operator*(LC x, const C& c)
{
    x *= c;
    return x;
}

template <class C, LayerCounts LC>
   requires is_scalar_v<C>
LC operator*(C c, const LC& x)
{
    return x * c;
}

template <LayerCounts LC>
LC operator-(LC x)
{
    x *= -1;
    return x;
}

template <LayerCounts LC, class C>
LC operator/(LC x, const C& c)
{
    x /= c;
    return x;
}


template <LayerCounts C>
void set_concurrent(C& c, bool activate=true)
{
    c.set_concurrent(activate);
}

template <class C>
constexpr void set_concurrent(C&) {}

template <LayerCounts T1, LayerCounts T2>
struct thresh_diff_imp<T1, T2>
{
    auto operator()(const T1& before, const T2& after, int t = 0)
    {
        std::conditional_t<is_specialization_of_v<std::decay_t<T1>, VLayerCount>,
                           VLayerCount<typename std::decay_t<T1>::cval_t>,
                           LayerCount<typename std::decay_t<T1>::cval_t, false, true>>
            diff;
        if constexpr (is_specialization_of_v<std::decay_t<T1>, VLayerCount>)
            diff._total = thresh_diff(before._total, after._total, t);
        for (auto&& [l, a] : after._h)
        {
            auto b = before.get(l);
            auto d = thresh_diff(b, a, t);
            if (d != 0)
                diff._h.insert(diff._h.end(), std::make_pair(l, d));
        }
        return diff;
    }
};

template <class T>
class def_shared_ptr:
    public std::shared_ptr<T>
{
public:
    def_shared_ptr() : std::shared_ptr<T>(std::make_shared<T>()) {}
};

template <class EC, class W, bool layered,
          class MMap = eprop_map_t<def_shared_ptr<shared_canary_mutex>>::unchecked_t>
class LayersPropMap
{
public:
    typedef typename boost::property_traits<EC>::key_type key_type;
    typedef typename boost::property_traits<W>::value_type::value_type w_val_t;
    typedef LayerCount<w_val_t, false, layered> value_type;
    typedef LayerCount<w_val_t, true, layered> reference;
    typedef boost::read_write_property_map_tag category;

    LayersPropMap() = default;

    template <class Graph>
    LayersPropMap(const EC& ec, const W& w, const Graph& g)
        : _ec(ec), _w(w)
    {
        auto M = base_graph(g).get_edge_index_range();
        resize(M);
    }

    template <class K>
    reference operator[](K&& key) const
    {
        return reference(*const_cast<LayersPropMap*>(this), key);
    }

    auto& get_checked()
    {
        if constexpr (_is_checked)
        {
            return *this;
        }
        else
        {
            typedef typename EC::checked_t ecprop_t;
            typedef typename W::checked_t wprop_t;
            typedef typename MMap::checked_t mmap_t;
            return *reinterpret_cast<LayersPropMap<ecprop_t, wprop_t, layered,
                                                   mmap_t>*>(this);
        }
    }

    void set_mutex(auto&& m)
    {
        _mutex  = m;
    }

    auto get_mutex() const
    {
        return _mutex;
    }

    void sync_size()
    {
        soft_reserve(_ec.get_storage().size());
    }

    void resize(size_t M)
    {
        _ec.resize(M);
        _w.resize(M);
        _mutex.resize(M);
    }

    void reserve(size_t M)
    {
        _ec.reserve(M);
        _w.reserve(M);
        _mutex.reserve(M);
    }

    void soft_reserve(size_t M)
    {
        _ec.soft_reserve(M);
        _w.soft_reserve(M);
        _mutex.soft_reserve(M);
    }

    void shrink_to_fit()
    {
        _ec.shrink_to_fit();
        _w.shrink_to_fit();
        _mutex.shrink_to_fit();
    }

    void set_concurrent(bool enable)
    {
        _concurrent = enable;
    }

    LayersPropMap copy()
    {
        auto ec = _ec.copy();
        auto w = _w.copy();
        auto map = LayersPropMap();
        map._ec = ec;
        map._w = w;
        map._mutex.resize(_mutex.get_storage().size());
        map._concurrent = _concurrent;
        return map;
    }

    constexpr static bool _layered = layered;

private:
    friend reference;

    EC _ec;
    W _w;

    MMap _mutex;
    bool _concurrent = false;

public:
    typedef std::decay_t<decltype(boost::get_checked(_ec))> cprop_t;
    constexpr static bool _is_checked = std::is_same_v<cprop_t, EC>;
};

template <class EC, class W, bool layered, class... Ts>
auto get(const LayersPropMap<EC, W, layered, Ts...>& p, auto&& key)
{
    return p[key];
}

template <class EC, class W, bool layered, class... Ts>
void put(LayersPropMap<EC, W, layered, Ts...>& p, auto&& key, auto&& val)
{
    p[key] = val;
}

template <bool layered, class EC, class W, class Graph, class... Ts>
auto make_layers_prop(const EC& ec, const W& w, const Graph& g)
{
    return LayersPropMap<EC, W, layered, Ts...>(ec, w, g);
}

} // namespace graph_tool

namespace std
{
using namespace graph_tool;
template <LayerCounts C>
struct formatter<C>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const C& c, format_context& ctx) const
    {
        string temp;
        format_to(back_inserter(temp), "[");
        c.template cdispatch<true>
            ([&](auto&& h, auto)
             {
                 for (auto&& [l, c] : h)
                     format_to(back_inserter(temp), "{}: {}, ", l, c);
             });
        if constexpr (is_specialization_of_v<C, VLayerCount>)
            format_to(back_inserter(temp), "total: {} ", c._total);
        format_to(back_inserter(temp), "]");
        return formatter<string_view>().format(temp, ctx);
    }
};


template <LayerCounts C>
C pow(const C& x, auto n)
{
    C y = x;
    for (auto&& [l, c] : y._h)
        c = std::pow(c, n);
    return y;
}

}

#endif // GRAPH_LAYERED_BLOCKMODEL_TYPES_HH
