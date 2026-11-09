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

#pragma once

#include "config.h"

#include "graph_tool.hh"

namespace graph_tool
{

struct dcount_t
    : public std::tuple<int64_t, int64_t>
{
    template <class... As>
    dcount_t(As&&... as)
        : std::tuple<int64_t, int64_t>(std::forward<As>(as)...) {}

    auto& operator+=(const dcount_t& o)
    {
        get<0>(*this) += get<0>(o);
        get<1>(*this) += get<1>(o);
        return *this;
    }

    auto& operator-=(const dcount_t& o)
    {
        get<0>(*this) -= get<0>(o);
        get<1>(*this) -= get<1>(o);
        return *this;
    }

    auto& operator*=(int x)
    {
        get<0>(*this) *= x;
        get<1>(*this) *= x;
        return *this;
    }

    bool operator==(int x) const
    {
        return get<0>(*this) == x && get<1>(*this) == x;
    }

    // component-wise p and n >= 0, such that *this == p - n
    dcount_t positive_part() const
    {
        return {std::max(get<0>(*this), int64_t(0)),
                std::max(get<1>(*this), int64_t(0))};
    }

    dcount_t negative_part() const
    {
        return {std::max(-get<0>(*this), int64_t(0)),
                std::max(-get<1>(*this), int64_t(0))};
    }
};

struct ovweight_t
    : public gt_hash_map<int64_t, dcount_t>
{
    ovweight_t() = default;

    ovweight_t(const odeg_t& d)
    {
        (*this) += d;
    }

    dcount_t get_count(auto k) const
    {
        auto iter = this->find(k);
        if (iter == this->end())
            return dcount_t();
        return iter->second;
    }

    ovweight_t& operator=(int64_t c)
    {
        if (c == 1) // vertex weight setting for nested SBM
        {
            if (this->empty())
                 throw ValueException(std::format("cannot unit set empty ovweight_t"));
            auto x = *this->begin();
            boost::unordered::erase_if
                (*this,
                 [&](auto& y)
                 {
                     return x != y;
                 });
            return *this;
        }

        if (c == 0)
            this->clear();
        else
            throw ValueException(std::format("invalid initialization of overlapping weight: {} {}", c, int64_t(*this)));

        return *this;
    }

    ovweight_t& operator+=(const ovweight_t& o)
    {
        for (const auto& [k, v] : o)
        {
            auto iter = this->find(k);
            if (iter == this->end())
            {
                if (v != 0)
                    iter = this->emplace(k, v).first;
            }
            else
            {
                iter->second += v;
                if (iter->second == 0)
                    this->erase(iter);
            }
        }
        return *this;
    }

    ovweight_t& operator-=(const ovweight_t& o)
    {
        for (auto& [k, v] : o)
        {
            auto iter = this->find(k);
            if (iter == this->end())
                iter = this->emplace(k, dcount_t()).first;
            iter->second -= v;
            if (iter->second == 0)
                this->erase(iter);
        }
        return *this;
    }

    ovweight_t operator-() const
    {
        ovweight_t o(*this);
        for (auto& [k, v] : o)
            v *= -1;
        return o;
    }

    ovweight_t operator+(const ovweight_t& o) const
    {
        ovweight_t r(*this);
        r += o;
        return r;
    }

    ovweight_t operator-(const ovweight_t& o) const
    {
        ovweight_t r(*this);
        r -= o;
        return r;
    }

    ovweight_t& operator+=(const odeg_t& d)
    {
        auto& k = (*this)[d._v];
        if (d._in)
            get<0>(k) += d._k;
        else
            get<1>(k) += d._k;
        return *this;
    }

    ovweight_t& operator*=(int64_t c)
    {
        for (auto& [k, v] : *this)
            v *= c;
        return *this;
    }

    bool operator==(const ovweight_t& o) const
    {
        if (this->size() != o.size())
            return false;

        for (const auto& [k, v] : o)
        {
            if (v != get_count(k))
                return false;
        }

        return true;
    }

    // bool operator==(size_t w) const
    // {
    //     return this->size() == w;
    // }

    operator int64_t (void) const { return this->size(); }
};

// NOTE: these are deliberately not atomic!

template <bool enable=true>
decltype(auto) atomic_inc(ovweight_t& x, const auto& y)
{
    x += y;
    return x;
}

template <bool enable=true>
decltype(auto) atomic_dec(ovweight_t& x, const auto& y)
{
    x -= y;
    return x;
}

template <bool enable=true>
decltype(auto) atomic_prod(ovweight_t& x, const auto& y)
{
    x *= y;
    return x;
}

template <bool enable=true>
decltype(auto) atomic_div(ovweight_t& x, const auto& y)
{
    x /= y;
    return x;
}


} // namespace graph_tool

namespace std
{
using namespace graph_tool;

template<>
struct formatter<ovweight_t>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const ovweight_t& w, format_context& ctx) const
    {
        string temp;
        format_to(back_inserter(temp), "[");
        for (auto& [k, v] : w)
            format_to(back_inserter(temp), "{}: ({}, {}),", k,
                      get<0>(v), get<1>(v));
        temp.resize(temp.size()-1);
        format_to(back_inserter(temp), "]");
        return formatter<string_view>().format(temp, ctx);
    }
};

}
