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

#include "hash_map_wrap.hh"

template <class Map,
          class SMap = gt_hash_map<typename Map::key_type,
                                   typename Map::mapped_type>>
class scoped_hash_map
{
public:
    scoped_hash_map(const Map& m) : _m(m){}

    typedef typename Map::key_type key_type;
    typedef typename Map::value_type value_type;
    typedef typename Map::mapped_type mapped_type;

    typedef typename Map::const_iterator m_iter_t;
    typedef typename SMap::iterator s_iter_t;
    typedef typename SMap::const_iterator s_citer_t;


    template <bool is_const>
    class scoped_iterator
        : public boost::iterator_facade<scoped_iterator<is_const>,
                                        std::conditional_t<is_const,
                                                           const value_type,
                                                           value_type>,
                                        boost::forward_traversal_tag>
    {
    public:
        scoped_iterator() = default;

        template <class SIter>
        scoped_iterator(const scoped_hash_map& m, m_iter_t m_iter, SIter s_iter)
            : _s(&m._s), _m(&m._m), _m_iter(m_iter), _s_iter(s_iter)
        {}

    private:
        friend class boost::iterator_core_access;
        friend class scoped_hash_map;

        void skip_m()
        {
            while (_m_iter != _m->end() && _s->find(_m_iter->first) != _s->end())
            {
                ++_m_iter;
            }
        }

        void increment()
        {
            if (_s_iter != _s->end())
            {
                ++_s_iter;
                if (_s_iter == _s->end())
                    skip_m();
            }
            else
            {
                ++_m_iter;
                skip_m();
            }
        }

        bool equal(scoped_iterator const& o) const
        {
            return this->_m_iter == o._m_iter && this->_s_iter == o._s_iter;
        }

        std::conditional_t<is_const,
                           const value_type&,
                           value_type&>
        dereference() const
        {
            if constexpr (is_const)
            {
                if (_s_iter != _s->end())
                    return *_s_iter;
                return *_m_iter;
            }
            else
            {
                return *_s_iter;
            }
        }

        const SMap* _s;
        const Map* _m;
        m_iter_t _m_iter;
        std::conditional_t<is_const,
                           s_citer_t,
                           s_iter_t> _s_iter;
    };

    scoped_iterator<true> begin() const
    {
        return scoped_iterator<true>(*this, _m.begin(), _s.begin());
    }

    scoped_iterator<true> end() const
    {
        return scoped_iterator<true>(*this, _m.end(), _s.end());
    }

    scoped_iterator<true> find(const key_type& k) const
    {
        auto s_iter = _s.find(k);
        if (s_iter != _s.end())
            return scoped_iterator<true>(*this, _m.begin(), s_iter);
        auto m_iter = _m.find(k);
        return scoped_iterator<true>(*this, m_iter, s_iter);
    }

    template< class... Args >
    std::pair<scoped_iterator<true>, bool>
    try_emplace(const key_type& k, Args&&... args)
    {
        auto s_iter = _s.find(k);
        if (s_iter != _s.end())
            return {scoped_iterator<true>(*this, _m.begin(), s_iter), false};

        auto m_iter = _m.find(k); // point to underlying value
        if (m_iter != _m.end())
            return {scoped_iterator<true>(*this, m_iter, _s.end()), false};

        s_iter = _s.emplace(k, std::forward<Args>(args)...);
        return {scoped_iterator<true>(*this, _m.begin(), s_iter), true};
    }

    template< class... Args >
    std::pair<scoped_iterator<false>, bool>
    emplace(const key_type& k, Args&&... args)
    {
        auto s_iter = _s.find(k);
        if (s_iter != _s.end())
            return {scoped_iterator<false>(*this, _m.begin(), s_iter), false};

        auto m_iter = _m.find(k);
        if (m_iter != _m.end()) // copy underlying value
        {
            s_iter = _s.insert(*m_iter).first;
            return {scoped_iterator<false>(*this, _m.begin(), s_iter), false};
        }

        s_iter = _s.emplace(k, std::forward<Args>(args)...).first;
        return {scoped_iterator<false>(*this, _m.begin(), s_iter), true};
    }


    scoped_iterator<false> insert(const value_type& v)
    {
        auto s_iter = _s.insert(v);
        return scoped_iterator<false>(*this, _m.begin(), s_iter);
    }

    // Does not actually erase! Reset to default value instead
    template <bool is_const>
    scoped_iterator<is_const> erase(scoped_iterator<is_const> iter)
    {
        auto s_iter = iter._s_iter;
        if (s_iter != _s.end())
        {
            const_cast<mapped_type&>(s_iter->second) = mapped_type();
            return iter;
        }
        else
        {
            s_iter = _s.emplace(iter->first, mapped_type()).first;
            return scoped_iterator<is_const>(*this, s_iter, _m.begin());
        }
    }

    auto erase(const key_type& k)
    {
        auto iter = find(k);
        if (iter != end())
            return erase(iter);
        return iter;
    }

    // always copy
    mapped_type& operator[](const key_type& key)
    {
        return emplace(key, mapped_type()).first->second;
    }

    template <class F>
    void apply_if(F&& f)
    {
        auto& m = const_cast<Map&>(_m);
        for (auto& kv : _s)
        {
            if (f(kv))
                m.insert(kv);
        }
    }

    void apply()
    {
        apply_if([](auto&&){ return true; });
    }

private:

    const Map& _m;
    SMap _s;
};

template <class Map>
auto make_scoped_map(const Map& m)
{
    return scoped_hash_map<Map>(m);
}
