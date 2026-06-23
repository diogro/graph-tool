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

template <class Mutex>
auto& get_canary(canary_mutex<Mutex>& lock)
{
    return lock._canary;
}

template <class Lock>
auto& get_canary(Lock& lock)
{
    return lock.mutex()->_canary;
}

struct dummy_mutex
{
    void lock() {}
    void unlock() {}
    auto* mutex() { return this; }
    constexpr static inline size_t _canary = 0;
};

constexpr size_t get_canary(const dummy_mutex&)
{
    return 0;
}

// Joint locking patterns

struct lock_args
{
    bool enable = true;
    bool pass_lock = false;
};

template <lock_args largs, class... Lock>
decltype(auto) do_lock(auto&& f, auto&... mutex)
{
    if constexpr (largs.enable)
    {
        std::tuple<Lock...>
            locks(Lock(mutex, std::defer_lock_t())...);
        if constexpr (sizeof...(Lock) > 1)
            std::apply([&](auto&... as){ std::lock(as...); }, locks);
        else
            std::get<0>(locks).lock();
        if constexpr (largs.pass_lock)
            return std::apply(f, locks);
        else
            return f();
    }
    else
    {
        if constexpr (largs.pass_lock)
            return f(mutex...);
        else
            return f();
    }
}

template <lock_args largs = {}>
decltype(auto) do_slock(auto&& f, auto&... mutex)
{
    return do_lock<largs, std::shared_lock<std::decay_t<decltype(mutex)>>...>
        (std::forward<decltype(f)>(f), mutex...);
}

template <lock_args largs = {}>
decltype(auto) do_slock_d(auto&& f, bool enable, auto&... mutex)
{
    if (enable)
        return do_slock<largs>
            (std::forward<decltype(f)>(f), mutex...);
    else
        return do_slock<largs>
            (std::forward<decltype(f)>(f), mutex...);
}

template <lock_args largs = {}>
decltype(auto) do_ulock(auto&& f, auto&... mutex)
{
    return do_lock<largs,
                   std::unique_lock<std::decay_t<decltype(mutex)>>...>
        (std::forward<decltype(f)>(f), mutex...);
}

template <lock_args largs = {}>
decltype(auto) do_ulock_d(auto&& f, bool enable, auto&... mutex)
{
    if (enable)
        return do_ulock<largs>
            (std::forward<decltype(f)>(f), mutex...);
    else
        return do_ulock<largs>
            (std::forward<decltype(f)>(f), mutex...);
}

struct upgrade_lock_args
{
    lock_args la;
    bool relock = true;
};

template <upgrade_lock_args largs = {}>
decltype(auto) upgrade_lock(auto&& f, auto&... locks)
{
    if constexpr (!largs.la.enable)
    {
        if constexpr (largs.la.pass_lock)
            return f((locks, dummy_mutex())...);
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
                 if constexpr (largs.relock)
                 {
                     if constexpr (sizeof...(locks) > 1)
                         std::lock(locks...);
                     else
                         (locks.lock(), ...);
                 }
             });

        return do_ulock<largs.la>
            (std::forward<decltype(f)>(f),
             *locks.mutex()...);
    }
}

template <upgrade_lock_args largs = {}>
decltype(auto) upgrade_lock_d(auto&& f, bool enable, auto&... locks)
{
    if (enable)
        return upgrade_lock<largs>
            (std::forward<decltype(f)>(f), locks...);
    else
        return upgrade_lock<{.la={.enable=false, .pass_lock=largs.la.pass_lock},
                             .relock=largs.relock}>
            (std::forward<decltype(f)>(f), locks...);
}

struct upgrade_canary_lock_args
{
    upgrade_lock_args ula;
    bool any = true;
};

template <upgrade_canary_lock_args largs = {}>
decltype(auto) upgrade_canary_lock(auto&& f, auto&... locks)
{
    if constexpr (largs.ula.la.enable)
    {
        std::array<size_t, sizeof...(locks)> stamp = {get_canary(locks)...};
        return upgrade_lock<largs.ula>
            ([&](auto&&... ls)
             {
                 auto lguard =
                     boost::scope::make_scope_exit
                     ([&]()
                      {
                          auto inc = [](auto&& x)
                                     {
                                         if constexpr (requires { x++; })
                                             x++;
                                     };
                          (inc(get_canary(locks)), ...);
                      });
                 std::array<size_t, sizeof...(locks)> canary = {get_canary(locks)...};
                 bool succ;
                 if constexpr (largs.any)
                     succ = std::ranges::any_of(std::views::zip(stamp, canary),
                                                [](auto&& sc){ return std::get<0>(sc) == std::get<1>(sc); });
                 else
                     succ = std::ranges::all_of(std::views::zip(stamp, canary),
                                                [](auto&& sc){ return std::get<0>(sc) == std::get<1>(sc); });
                return f(ls..., succ);
             }, locks...);
    }
    else
    {
        return upgrade_lock<largs.ula>
            ([&](auto&&... ls)
             {
                return f(ls..., true);
             }, locks...);
    }
}

template <upgrade_canary_lock_args largs = {}>
decltype(auto) upgrade_canary_lock_d(auto&& f, bool enable, auto&... locks)
{
    if (enable)
        return upgrade_canary_lock<largs>(std::forward<decltype(f)>(f), locks...);
    else
        return f(true);
}


template <lock_args largs = {}>
decltype(auto) do_ulock_pair(auto&& f, auto u, auto v, auto& mu, auto& mv)
{
    if (u != v)
        return do_ulock<largs>
            (std::forward<decltype(f)>(f), mu, mv);
    else
        return do_ulock<largs>
            (std::forward<decltype(f)>(f), mu);
}

template <lock_args largs = {}>
decltype(auto) do_slock_pair(auto&& f, auto u, auto v, auto& mu, auto& mv)
{
    if (u != v)
        return do_slock<largs>
            (std::forward<decltype(f)>(f), mu, mv);
    else
        return do_slock<largs>
            (std::forward<decltype(f)>(f), mu);
}

// visit or emplace

template <upgrade_canary_lock_args largs = {}>
void visit_or_emplace(auto&& m, auto&& k, auto&& f, auto&& g,
                      auto& lock)
{
    auto iter = m.find(k);
    if (iter == m.end())
    {
        upgrade_canary_lock<largs>
            ([&](auto&&... ls, bool succ)
             {
                 if (succ)           [[likely]]
                 {
                     iter = m.try_emplace(k).first;
                     g(iter->second, ls...);
                 }
                 else                [[unlikely]]
                 {
                     // lost race; search again
                     iter = m.find(k);
                     if (iter == m.end()) [[likely]]
                     {
                         iter = m.try_emplace(k).first;
                         g(iter->second, ls...);
                     }
                     else                 [[unlikely]]
                     {
                         f(iter->second, ls...);
                     }
                 }
             }, lock);
    }
    else
    {
        if constexpr (largs.ula.la.pass_lock)
            f(iter->second, lock);
        else
            f(iter->second);
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
