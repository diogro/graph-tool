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

#ifndef LOCK_FREE_STACK_HH
#define LOCK_FREE_STACK_HH

#include <atomic>
#include <deque>
#include <vector>

#include "openmp.hh"

// Nonblocking stack using Treiber's algorithm, see:
// https://en.wikipedia.org/wiki/Treiber_stack

// This version uses thread based storage and delayed deallocation to solve the
// ABA problem (https://en.wikipedia.org/wiki/ABA_problem).
//
// TODO: replace with a <hazard_pointer> implementatiion when C++26 becomes
// available.

template <class T>
class lock_free_stack
{
 public:
    lock_free_stack() : _storage(get_num_threads()) {}

    lock_free_stack(const lock_free_stack& o)
    {
        std::vector<T> tmp;

        auto node = o._head.load();
        while (node)
        {
            tmp.push_back(node->data);
            node = node->next;
        }

        while (!tmp.empty())
        {
            push(tmp.back());
            tmp.pop_back();
        }
    }

    lock_free_stack& operator=(lock_free_stack o)
    {
        _head = o._head.load();
        o._head = nullptr;
        _storage.swap(o._storage);
        return *this;
    }

    void push(const T& val)
    {
        auto& pool = _storage[get_thread_num()];
        pool.emplace_back(val);
        auto new_node = &pool.back();

        // replace head with new node; loop if another thread comes first
        new_node->next = _head;
        while (!_head.compare_exchange_strong(new_node->next, new_node));

        _size++;
    }

    bool pop(T& val)
    {
        node_t* old_head = nullptr;

        // replace head with next; loop if another thread comes first
        old_head = _head.load();
        do
        {
            if (!old_head)
                return false; // stack is empty
        }
        while (!_head.compare_exchange_strong(old_head, old_head->next));

        val = old_head->data;

        _size--;

        return true;
    }

    size_t size()
    {
        return _size;
    }

    void clear()
    {
        _storage.clear();
        _storage.resize(get_num_threads());
        _head = nullptr;
        _size = 0;
    }

 private:
    struct node_t
    {
        node_t(T const& d): data(d)  {}

        T data;
        node_t* next = nullptr;
    };

    std::vector<std::deque<node_t>> _storage;

    std::atomic<node_t*> _head = nullptr;
    std::atomic<size_t> _size = 0;
};

#endif // LOCK_FREE_STACK_HH
