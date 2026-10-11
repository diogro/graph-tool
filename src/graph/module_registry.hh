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

#include <functional>
#include <vector>
#include <tuple>
#include <algorithm>
#include <limits>
#include <typeindex>
#include <unordered_map>
#include <any>
#include <boost/python/class.hpp>
#include <boost/python/exception_translator.hpp>

#include "demangle.hh"

#ifndef __MOD__
#error "__MOD__ needs to be defined"
#endif

#define REGISTER_MOD static __MOD__::RegisterMod __reg

class ObjNotFound: public std::exception
{
public:
    ObjNotFound(const std::string& name)
    {
        _msg = std::format("class not found: {}", name);
    }

    const char* what() const noexcept
    {
        return _msg.c_str();
    }

private:
    std::string _msg;
};

namespace __MOD__
{

// Registry of exposed functions

typedef std::vector<std::tuple<int,std::function<void()>>> reg_t;

reg_t& mod_reg()
#ifdef DEF_REGISTRY
{
    static reg_t* reg = new reg_t();
    return *reg;
}
#else
;
#endif

// Registry of exposed classes and other objects

typedef std::unordered_map<std::type_index, std::any> creg_t;

creg_t& class_reg()
#ifdef DEF_REGISTRY
{
    static creg_t* creg = new creg_t();
    return *creg;
}
#else
;
#endif

// Get existing object from registry; or throw if not found

template <class Obj, class... As>
Obj& get_obj(As&&... as)
{
    auto& creg = class_reg();
    auto idx = std::type_index(typeid(Obj));
    auto iter = creg.find(idx);
    if (iter != creg.end())
        return std::any_cast<Obj&>(iter->second);
    if constexpr (sizeof...(as) == 0)
    {
        throw ObjNotFound(get_type_name<Obj>());
    }
    else
    {
        auto& a = creg[idx];
        a.emplace<Obj>(as...);
        return std::any_cast<Obj&>(a);
    }
}

// Especialization for classes

template <class... Args, class... As>
boost::python::class_<Args...>& get_class(As&&... as)
{
    return get_obj<boost::python::class_<Args...>>(std::forward<As>(as)...);
}

// If object is already registerd do nothing; otherwise register with arguments
// given argument. Return a callable objext that takes an operator on the
// registered object (e.g. that can define member functions). The operator is
// only called if the object has just been registered.

template <class Obj, class... As>
auto def_obj(As&&... as)
{
    Obj* obj = nullptr;
    try
    {
        get_obj<Obj>();
    }
    catch (ObjNotFound&)
    {
        obj = &get_obj<Obj>(std::forward<As>(as)...);
    }
    return [obj](auto&& f){ if (obj != nullptr) f(*obj); };
}

class RegisterMod
{
public:
    RegisterMod(std::function<void()> f, int p = 0)
    {
        mod_reg().emplace_back(p, f);
    }
};

class EvokeRegistry
{
public:
    EvokeRegistry()
    {
        boost::python::register_exception_translator<ObjNotFound>
            ([](const auto& e)
             {
                 PyObject* error = PyExc_RuntimeError;
                 PyErr_SetString(error, e.what());
             });
        reg_t& reg = mod_reg();
        std::sort(reg.begin(), reg.end(),
                  [](const auto& a, const auto& b)
                  { return std::get<0>(a) < std::get<0>(b); });
        for (auto& [p, f] : reg)
            f();
        delete &reg;
        delete &class_reg();
    }
};

} // namespace __MOD__
