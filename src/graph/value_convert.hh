// graph-tool -- a general graph modification and manipulation thingy
//
// Copyright (C) 2006-2024 Tiago de Paula Peixoto <tiago@skewed.de>
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

#ifndef VALUE_CONVERT_HH
#define VALUE_CONVERT_HH

#include <string>
#include <vector>
#include <boost/python/object.hpp>
#include <boost/python/extract.hpp>
#include <boost/lexical_cast.hpp>

#include "demangle.hh"
#include "graph_exceptions.hh"

namespace boost
{
template <class T>
std::string lexical_cast(const std::vector<T>& v)
{
    std::string out = "(";
    for (size_t i = 0; i < v.size(); ++i)
    {
        out += lexical_cast<std::string>(v[i]);
        if (i < v.size() - 1)
            out += ", ";
    }
    out += ")";
    return out;
}
}

namespace graph_tool
{

// handle type convertions

template <class T1, class T2, class Enable = void>
struct convert_dispatch;

// generic types
template <class Type1, class Type2>
Type1 convert(const Type2& v)
{
    try
    {
        if constexpr (std::is_same_v<Type1, Type2>)
        {
            return v;
        }
        else if constexpr (std::is_same_v<Type1, boost::python::object>)
        {
            return boost::python::object(v);
        }
        else if constexpr (std::is_same_v<Type2, boost::python::object>)
        {
            boost::python::extract<Type1> x(v);
            if (x.check())
                return x();
            else
                throw boost::bad_lexical_cast();
        }
        else if constexpr (std::is_same_v<Type2, std::string>)
        {
            //uint8_t is not char, it is bool!
            if (std::is_same_v<Type1, uint8_t>)
                return convert<Type1,int>(boost::lexical_cast<int>(v));
            else
                return boost::lexical_cast<Type1>(v);
        }
        else if constexpr (std::is_same_v<Type1, std::string>)
        {
            //uint8_t is not char, it is bool!
            if (std::is_same_v<Type2, uint8_t>)
                return boost::lexical_cast<std::string>(convert<int, Type2>(v));
            else
                return boost::lexical_cast<std::string>(v);
        }
        else if constexpr (std::is_convertible_v<Type2,Type1>)
        {
            return Type1(v);
        }
        else
        {
            return convert_dispatch<Type1, Type2>()(v);
        }
    }
    catch (boost::bad_lexical_cast&)
    {
        std::string name1 = name_demangle(typeid(Type1).name());
        std::string name2 = name_demangle(typeid(Type2).name());
        std::string val_name;
        try
        {
            if constexpr (std::is_same_v<Type2, boost::python::object>)
            {
                val_name = boost::python::extract<std::string>(v.attr("__str__")());
            }
            else if constexpr (std::is_same_v<Type2, std::vector<boost::python::object>>)
            {
                val_name = "[";
                for (size_t i = 0; i < v.size(); ++i)
                {
                    val_name += boost::python::extract<std::string>(v[i].attr("__str__")());
                    if (i < v.size() - 1)
                        val_name += ", ";
                }
                val_name += "]";
            }
            else
            {
                val_name = boost::lexical_cast<std::string>(v);
            }
        }
        catch (boost::bad_lexical_cast&)
        {
            val_name = "<no lexical cast available>";
        }
        throw ValueException("error converting from type '" + name2 +
                             "' to type '" + name1 + "', val: " + val_name);
    }
}

// specific specializations

// vectors
template <class T1, class T2>
struct convert_dispatch<std::vector<T1>, std::vector<T2>>
{
    std::vector<T1> operator()(const std::vector<T2>& v) const
    {
        std::vector<T1> v2(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            v2[i] = convert<T1, T2>(v[i]);
        return v2;
    }
};

// default action
template <class T1, class T2, class Enable>
struct convert_dispatch
{
    T1 operator()(const T2&) const
    {
        throw boost::bad_lexical_cast();
    }
};

} // namespace graph_tool

#endif // VALUE_CONVERT_HH
