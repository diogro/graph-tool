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

#ifndef PARALLEL_UTIL_HH
#define PARALLEL_UTIL_HH

#include <mutex>
#include <shared_mutex>
#include <tuple>
#include <variant>

#include <boost/scope/scope_exit.hpp>

namespace graph_tool
{

// Mutex with a "canary" counter, for lock upgrades

template <class Mutex>
struct canary_mutex
    : public Mutex
{
    size_t _canary = 0;
};

using shared_canary_mutex = canary_mutex<std::shared_mutex>;

constexpr size_t get_canary(std::monostate&)
{
    return 0;
}

auto& get_canary(auto& lock)
{
    return lock.mutex()->_canary;
}


// Joint locking patterns

template <template<class> class Lock, bool enable, bool pass_lock,
          class F, class... Mutex>
decltype(auto) do_lock(F&& f, Mutex&... mutex)
{
    if constexpr (enable)
    {
        std::tuple<Lock<Mutex>...>
            locks(Lock<Mutex>(mutex, std::defer_lock_t())...);
        if constexpr (sizeof...(Mutex) > 1)
            std::apply([&](auto&... as){ std::lock(as...); }, locks);
        else
            std::get<0>(locks).lock();
        if constexpr (pass_lock)
            return std::apply(f, locks);
        else
            return f();
    }
    else
    {
        if constexpr (pass_lock)
            return f(mutex...);
        else
            return f();
    }
}

template <bool enable=true, bool pass_lock=false>
decltype(auto) do_slock(auto&& f, auto&... mutex)
{
    return do_lock<std::shared_lock, enable, pass_lock>
        (std::forward<std::decay_t<decltype(f)>>(f), mutex...);
}

template <bool pass_lock=false>
decltype(auto) do_slock_d(auto&& f, bool enable, auto&... mutex)
{
    if (enable)
        return do_slock<true, pass_lock>
            (std::forward<std::decay_t<decltype(f)>>(f), mutex...);
    else
        return do_slock<false, pass_lock>
            (std::forward<std::decay_t<decltype(f)>>(f), mutex...);
}

template <bool enable=true, bool pass_lock=false>
decltype(auto) do_ulock(auto&& f, auto&... mutex)
{
    return do_lock<std::unique_lock, enable, pass_lock>
        (std::forward<std::decay_t<decltype(f)>>(f), mutex...);
}

template <bool pass_lock=false>
decltype(auto) do_ulock_d(auto&& f, bool enable, auto&... mutex)
{
    if (enable)
        return do_ulock<true, pass_lock>
            (std::forward<std::decay_t<decltype(f)>>(f), mutex...);
    else
        return do_ulock<false, pass_lock>
            (std::forward<std::decay_t<decltype(f)>>(f), mutex...);
}

template <bool enable=true, bool pass_lock=false, bool relock=true>
decltype(auto) upgrade_lock(auto&& f, auto&... locks)
{
    if constexpr (!enable)
    {
        if constexpr (pass_lock)
            return f((locks, std::monostate())...);
        else
            return f();
    }
    else
    {
        (locks.unlock(), ...);

        auto lguard =
            boost::scope::make_scope_exit
            ([&]()
             {
                 if constexpr (relock)
                 {
                     if constexpr (sizeof...(locks) > 1)
                         std::lock(locks...);
                     else
                         (locks.lock(), ...);
                 }
             });

        return do_ulock<enable, pass_lock>
            (std::forward<std::decay_t<decltype(f)>>(f),
             *locks.mutex()...);
    }
}

template <bool pass_lock=false, bool relock=true>
decltype(auto) upgrade_lock_d(auto&& f, bool enable, auto&... locks)
{
    if (enable)
        return upgrade_lock<true, pass_lock, relock>
            (std::forward<std::decay_t<decltype(f)>>(f), locks...);
    else
        return upgrade_lock<false, pass_lock, relock>
            (std::forward<std::decay_t<decltype(f)>>(f), locks...);
}

template <bool enable=true>
decltype(auto) do_ulock_pair(auto&& f, auto u, auto v, auto& mu, auto& mv)
{
    if (u != v)
        return do_ulock<enable>
            (std::forward<std::decay_t<decltype(f)>>(f), mu, mv);
    else
        return do_ulock<enable>
            (std::forward<std::decay_t<decltype(f)>>(f), mu);
}

template <bool enable=true>
decltype(auto) do_slock_pair(auto&& f, auto u, auto v, auto& mu, auto& mv)
{
    if (u != v)
        return do_slock<enable>
            (std::forward<std::decay_t<decltype(f)>>(f), mu, mv);
    else
        return do_slock<enable>
            (std::forward<std::decay_t<decltype(f)>>(f), mu);
}

// visit or emplace

template <bool pass_lock=false>
void visit_or_emplace(auto&& m, auto&& k, auto&& f, auto&& g,
                      auto& lock)
{
    auto dispatch =
        [&](auto&& h, auto&& x, auto&& lock)
        {
            if constexpr (pass_lock)
                h(x, lock);
            else
                h(x);
        };

    auto iter = m.find(k);
    if (iter == m.end())
    {
        auto& canary = get_canary(lock);
        auto stamp = canary;
        upgrade_lock<true, true>
            ([&](auto& ulock)
             {
                 if (stamp == canary) [[likely]]
                 {
                     iter = m.try_emplace(k).first;
                     dispatch(g, iter->second, ulock);
                 }
                 else                 [[unlikely]]
                 {
                     // lost race; search again
                     iter = m.find(k);
                     if (iter == m.end()) [[likely]]
                     {
                         iter = m.try_emplace(k).first;
                         dispatch(g, iter->second, ulock);
                     }
                     else                 [[unlikely]]
                     {
                         dispatch(f, iter->second, ulock);
                     }
                 }
                 canary++;
             }, lock);
    }
    else
    {
        dispatch(f, iter->second, lock);
    }
}

// atomic operations

template <bool enable=true, class T1, class T2>
void atomic_inc(T1&& x, T2&& d)
{
    typedef std::decay_t<T1> TX;
    typedef std::decay_t<T2> TD;
    if constexpr (std::is_scalar_v<TX> &&
                  std::is_scalar_v<TD> && enable)
    {
        #pragma omp atomic
        x += d;
    }
    else if constexpr (std::is_scalar_v<TX> && enable)
    {
        #pragma omp atomic
        x += TX(d);
    }
    else
    {
        x += d;
    }
}

template <bool enable=true, class T1, class T2>
void atomic_dec(T1&& x, T2&& d)
{
    typedef std::decay_t<T1> TX;
    typedef std::decay_t<T2> TD;
    if constexpr (std::is_scalar_v<TX> &&
                  std::is_scalar_v<TD> && enable)
    {
        #pragma omp atomic
        x -= d;
    }
    else if constexpr (std::is_scalar_v<TX> && enable)
    {
        #pragma omp atomic
        x -= TX(d);
    }
    else
    {
        x -= d;
    }
}

template <bool enable=true, class T1, class T2>
void atomic_prod(T1&& x, T2&& d)
{
    typedef std::decay_t<T1> TX;
    typedef std::decay_t<T2> TD;
    if constexpr (std::is_scalar_v<TX> &&
                  std::is_scalar_v<TD> && enable)
    {
        #pragma omp atomic
        x *= d;
    }
    else if constexpr (std::is_scalar_v<TX> && enable)
    {
        #pragma omp atomic
        x *= TX(d);
    }
    else
    {
        x *= d;
    }
}

template <bool enable=true, class T1, class T2>
void atomic_div(T1&& x, T2&& d)
{
    typedef std::decay_t<T1> TX;
    typedef std::decay_t<T2> TD;
    if constexpr (std::is_scalar_v<TX> &&
                  std::is_scalar_v<TD> && enable)
    {
        #pragma omp atomic
        x /= d;
    }
    else if constexpr (std::is_scalar_v<TX> && enable)
    {
        #pragma omp atomic
        x /= TX(d);
    }
    else
    {
        x /= d;
    }
}


} // namespace graph_tool

#endif // PARALLEL_UTIL_HH
