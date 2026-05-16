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

template <class T>
struct is_vector
{
    static bool const value = false;
};

template <class T>
struct is_vector<std::vector<T>>
{
    static bool const value = true;
};

template <class T>
constexpr bool is_vector_v = is_vector<T>::value;

// handle type convertions

template <class To, class From, class Enable = void>
struct convert_dispatch;

template <class To, class From>
struct is_convertible;

template <class To, class From, bool check=false>
//std::conditional_t<std::is_same_v<To, From>, const To&, To>
auto convert(const From& v)
{
    try
    {
        if constexpr (std::is_same_v<To, From>)
        {
            return (const To&)v;
        }
        else if constexpr (std::is_same_v<To, boost::python::object>)
        {
            return boost::python::object(v);
        }
        else if constexpr (std::is_same_v<From, boost::python::object>)
        {
            boost::python::extract<To> x(v);
            if (x.check())
            {
                return x();
            }
            else
            {
                if constexpr (is_vector_v<To>)
                {
                    To y;
                    for (int i = 0; i < boost::python::len(v); ++i)
                    {
                        boost::python::extract<typename To::value_type> x(v[i]);
                        if (x.check())
                            y.push_back(x());
                        else
                            throw boost::bad_lexical_cast();
                    }
                    return y;
                }
                else
                {
                    throw boost::bad_lexical_cast();
                }
            }
        }
        else if constexpr (std::is_same_v<From, std::string> && std::is_scalar_v<To>)
        {
            //uint8_t is not char, it is bool!
            if (std::is_same_v<To, uint8_t>)
                return convert<To,int>(boost::lexical_cast<int>(v));
            else
                return boost::lexical_cast<To>(v);
        }
        else if constexpr (std::is_same_v<To, std::string> && std::is_scalar_v<From>)
        {
            //uint8_t is not char, it is bool!
            if (std::is_same_v<From, uint8_t>)
                return boost::lexical_cast<std::string>(convert<int, From>(v));
            else
                return boost::lexical_cast<std::string>(v);
        }
        else if constexpr (std::is_convertible_v<From, To>)
        {
            return To(v);
        }
        else if constexpr (is_vector_v<To> && is_vector_v<From>)
        {
            constexpr bool is_conv =
                graph_tool::is_convertible<typename To::value_type,
                                           typename From::value_type>::value;
            if constexpr (is_conv)
            {
                To v2(v.size());
                for (size_t i = 0; i < v.size(); ++i)
                    v2[i] = convert<typename To::value_type,
                                    typename From::value_type>(v[i]);
                return v2;
            }
            else
            {
                if constexpr (check)
                    return convert_dispatch<To, From>::check();
                else
                    return convert_dispatch<To, From>()(v);
            }
        }
        else
        {
            if constexpr (check)
                return convert_dispatch<To, From>::check();
            else
                return convert_dispatch<To, From>()(v);
        }
    }
    catch (boost::bad_lexical_cast&)
    {
        std::string name1 = get_type_name<To>();
        std::string name2 = get_type_name<From>();
        std::string val_name;
        try
        {
            if constexpr (std::is_same_v<From, boost::python::object>)
            {
                val_name = boost::python::extract<std::string>(v.attr("__str__")());
            }
            else if constexpr (std::is_same_v<From, std::vector<boost::python::object>>)
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

// extensible specializations

// default action
template <class To, class From, class Enable>
struct convert_dispatch
{
    To operator()(const From&) const
    {
        throw boost::bad_lexical_cast();
    }

    static void check() {}
};

struct convert_dispatch_base
{
    static void* check() { return nullptr; }
};

template <class To, class From>
struct is_convertible
{
    constexpr static bool value =
        !std::is_void_v<decltype(convert<To, From, true>(From()))>;
};

template <class To, class From>
constexpr bool is_convertible_v = is_convertible<To, From>::value;

} // namespace graph_tool

#endif // VALUE_CONVERT_HH
