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

#include <shared_mutex>
#include <utility>

namespace graph_tool
{

template <class Mutex>
class slock
{
public:
    slock(Mutex& mutex, bool lock=true)
        : _lock(mutex, std::defer_lock_t())
    {
        if (lock)
            _lock.lock();
    }

private:
    std::shared_lock<Mutex> _lock;
};

template <class Mutex>
class ulock
{
public:
    ulock(Mutex& mutex, bool lock=true)
        : _lock(mutex, std::defer_lock_t())
    {
        if (lock)
            _lock.lock();
    }

private:
    std::unique_lock<Mutex> _lock;
};


template <class F, class Mutex>
auto do_slock(F&& f, Mutex& mutex, bool lock=true)
{
    slock<Mutex> lock_guard(mutex, lock);
    return f();
}

template <class F, class Mutex>
auto do_ulock(F&& f, Mutex& mutex, bool lock=true)
{
    ulock<Mutex> lock_guard(mutex, lock);
    return f();
}

} // namespace graph_tool

#endif // PARALLEL_UTIL_HH
