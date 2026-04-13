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

#ifndef DISPATCH_HH
#define DISPATCH_HH

#include "numpy_bind.hh"

#include <boost/hana.hpp>
namespace hana = boost::hana;
using namespace hana::literals;

// Dispatching
// -----------
//
// We want to generate versions of a template algorithm for every possible type
// of graph views. The types of graph views are the following:
//
//    - The original directed multigraph
//
//    - Filtered graphs, based on MaskFilter below
//
//    - A reversed view of each directed graph (original + filtered)
//
//    - An undirected view of each directed (unreversed) graph (original +
//      filtered)
//
// The total number of graph views is then: 1 + 1 + 2 + 2 = 6
//
// The specific specialization can be called at run time (and generated at
// compile time) with the gt_dispatch() function, which takes as arguments the
// function to be called, the type ranges of each argument (as hana::tuple), and
// the actual parameters as std::any values.
//
// Example:
//
// template <class Graph, class ValueType>
// void my_algorithm(Graph& g, ValueType val)
// {
//      ... do something ...
// };
//
// ...
//
// GraphInterface gi;
// auto value_types = hana::tuple_c<int, double, string>;
// double foo = 42.0;
// gt_dispatch<>()([](auto& g, auto val){ my _algorithm(g, val); },
//                 all_graph_views, value_tyoes)
//                 (g.get_graph_views(), std::any(foo));
//
// The above line will run my_algorithm with Graph being the appropriate graph
// view type and ValueType being 'double' and val = 42.0.

namespace graph_tool
{

#pragma GCC visibility push(default)

class DispatchNotFound
    : public GraphException
{
public:
    DispatchNotFound(const std::string& dispatch,
                     size_t pos,
                     const std::vector<std::string>& trs,
                     const std::string& val)
    : GraphException(""), _dispatch(dispatch), _pos(pos), _trs(trs),
      _val(val)
    {
        _error =
            "No static implementation was found for the desired routine. "
            "This is a graph_tool bug. :-( Please submit a bug report at "
            PACKAGE_BUGREPORT ". What follows is debug information.\n\n";

        _error += "Dispatch: " + _dispatch + "\n\n";

        _error += std::format("Cannot extract argument number {} of desired types {} from value: {}",
                              _pos, _trs, _val);
    }

    virtual ~DispatchNotFound() throw () {}

    const auto& get_pos() const { return _pos; }
    const auto& get_trs() const { return _trs; }
    const auto& get_val() const { return _val; }

private:
    std::string _dispatch;
    size_t _pos;
    std::vector<std::string> _trs;
    std::string _val;
};

#pragma GCC visibility pop

struct all_types_func_t {};

struct last_type_func_t {};

template <class T>
struct extract_param;

struct bad_extract
    : public std::exception
{
    bad_extract() = default;
    bad_extract(const std::string& what) : _what(what) {}
    virtual const char* what() const noexcept { return _what.c_str(); }
private:
    std::string _what;
};

std::vector<std::string> print_tr(auto tr)
{
    std::vector<std::string> trs;
    hana::for_each(tr,
                   [&](auto x)
                   {
                       trs.push_back(get_type_name<typename decltype(x)::type>());
                   });
    return trs;
}

template <typename T>
std::string print_val(T&& a)
{
    if constexpr (std::is_same_v<std::decay_t<T>, std::any>)
        return name_demangle(a.type().name());
    else
        return boost::python::extract<std::string>(a.attr("__repr__")());
};

struct gt_dispatch_args
{
    bool uncheck = true;
    bool pass_types = false;
    bool gil_release = true;
};

template <gt_dispatch_args dargs = {}>
struct gt_dispatch
{
    template <class Dispatch, class... TRS>
    auto operator()(Dispatch&& d, TRS... trs)
    {
        return [=](auto&&... as)
               {
                   dispatcher(d,
                              hana::make_tuple(trs...),
                              hana::make_tuple(std::ref(as)...),
                              hana::tuple<>());
               };
    }

    template <class Type, class IndexMap>
    static auto& pmap(boost::checked_vector_property_map<Type,IndexMap>& a)
    {
        if constexpr (dargs.uncheck)
            return a.get_unchecked();
        else
            return a;
    }

    template <class Type>
    static decltype(auto) pmap(Type&& a)
    {
        return std::forward<Type>(a);
    }

    template <class Dispatch, class TRS, class AS, class XTS, class... XS>
    void dispatcher(Dispatch&& d, TRS trs, AS as, XTS xts, XS&&... xs)
    {
        if constexpr (hana::size(trs) == hana::size_c<0>)
        {
            GILRelease gil(dargs.gil_release);
            if constexpr (dargs.pass_types)
                d(xts, pmap(std::forward<XS>(xs))...);
            else
                d(pmap(std::forward<XS>(xs))...);
        }
        else
        {
            auto& a = as[0_c].get();
            using a_t = std::decay_t<decltype(a)>;

            bool found = false;

            constexpr auto tr =
                hana::transform
                (hana::front(trs),
                 [&](auto t)
                 {
                     using tr_t = decltype(+t)::type;
                     if constexpr (std::is_base_of_v<last_type_func_t, tr_t>)
                     {
                         static_assert(hana::size(xts) != hana::size_c<0>);
                         if constexpr (hana::size(xts) != hana::size_c<0>)  // to avoid a clang miscompilation
                             return tr_t::get(hana::back(xts));
                     }
                     else if constexpr (std::is_base_of_v<all_types_func_t, tr_t>)
                     {
                         return tr_t::get(xts);
                     }
                     else
                     {
                         return t;
                     }
                 });

            hana::for_each
                (tr,
                 [&](auto t)
                 {
                     if (found)
                         return;

                     auto dispatch =
                         [&](auto&& val)
                         {
                             dispatcher(std::forward<Dispatch>(d),
                                        hana::drop_front(trs, hana::size_c<1>),
                                        hana::drop_front(as, hana::size_c<1>),
                                        hana::append(xts, t),
                                        std::forward<XS>(xs)...,
                                        std::forward<decltype(val)>(val));

                             found = true;
                         };

                     using T_R = typename decltype(+t)::type;
                     using T = std::decay_t<T_R>;

                     if constexpr (std::is_same_v<a_t, std::any>)
                     {
                         T* val = std::any_cast<T>(&a);
                         if (val != nullptr)
                         {
                             dispatch(*val);
                         }
                         else
                         {
                             auto rval = std::any_cast<std::reference_wrapper<T>>(&a);
                             if (rval != nullptr)
                             {
                                 dispatch(rval->get());
                             }
                             else
                             {
                                 auto sval = std::any_cast<std::shared_ptr<T>>(&a);
                                 if (sval != nullptr)
                                     dispatch(**sval);
                             }
                         }
                     }
                     else // boost::python::object
                     {
                         try
                         {
                             auto&& val = extract_param<T_R>()(a);
                             dispatch(val);
                         }
                         catch (const bad_extract& e) {};
                     }
                 });

            if (!found)
                throw DispatchNotFound(get_type_name<Dispatch>(),
                                       hana::size(xts), print_tr(tr),
                                       print_val(a));
        }
    }
};


struct same_arg_t
    : public last_type_func_t
{
    template <class T>
    constexpr static auto get(T)
    {
        return hana::type_c<typename T::type>;
    }
};
constexpr inline auto same_arg_type = hana::tuple_t<same_arg_t>;

struct vprop_same_t
    : public last_type_func_t
{
    template <class T>
    constexpr static auto get(T)
    {
        return hana::type_c<vprop_map_t<typename T::type::value_type>>;
    }
};
constexpr inline auto vprop_same_type = hana::tuple_t<vprop_same_t>;

struct eprop_same_t
    : public last_type_func_t
{
    template <class T>
    constexpr static auto get(T)
    {
        return hana::type_c<eprop_map_t<typename T::type::value_type>>;
    }
};

constexpr inline auto eprop_same_type = hana::tuple_t<eprop_same_t>;

struct dynamic_prop_t
    : public last_type_func_t
{
    template <class T>
    constexpr static auto get(T)
    {
        return hana::type_c<DynamicPropertyMapWrap<typename T::type::value_type,
                                                   typename T::type::key_type>>;
    }
};

constexpr inline auto dynamic_prop_type = hana::tuple_t<dynamic_prop_t>;

struct velem_dprop_t
    : public last_type_func_t
{
    template <class T>
    constexpr static auto get(T)
    {
        typedef typename T::type::value_type val_t;
        if constexpr (std::is_same_v<val_t, boost::python::object>)
            return hana::type_c<DynamicPropertyMapWrap<typename T::type::value_type,
                                                       typename T::type::key_type>>;
        else
            return hana::type_c<DynamicPropertyMapWrap<typename T::type::value_type::value_type,
                                                       typename T::type::key_type>>;
    }
};

constexpr inline auto velem_dprop_type = hana::tuple_t<velem_dprop_t>;

struct velem_vprop_t
    : public last_type_func_t
{
    template <class T>
    constexpr static auto get(T)
    {
        typedef typename T::type::value_type val_t;
        if constexpr (std::is_same_v<val_t, boost::python::object>)
            return hana::type_c<vprop_map_t<typename T::type::value_type>>;
        else
            return hana::type_c<vprop_map_t<typename T::type::value_type::value_type>>;
    }
};

constexpr inline auto velem_vprop_type = hana::tuple_t<velem_vprop_t>;

struct velem_eprop_t
    : public last_type_func_t
{
    template <class T>
    constexpr static auto get(T)
    {
        typedef typename T::type::value_type val_t;
        if constexpr (std::is_same_v<val_t, boost::python::object>)
            return hana::type_c<eprop_map_t<typename T::type::value_type>>;
        else
            return hana::type_c<eprop_map_t<typename T::type::value_type::value_type>>;
    }
};

constexpr inline auto velem_eprop_type = hana::tuple_t<velem_eprop_t>;

template <class T>
struct extract_param_base
{
    T operator()(boost::python::object obj)
    {
        boost::python::extract<T> extract(obj);
        if (extract.check())
        {
            return extract();
        }
        else
        {
            boost::python::object aobj;
            if (PyObject_HasAttrString(obj.ptr(), "_get_any"))
                aobj = obj.attr("_get_any")();
            else
                aobj = obj;
            boost::python::extract<std::any&> extract(aobj);
            if (!extract.check())
                throw_err(obj);
            std::any& aval = extract();
            return (*this)(aval);
        }
    }

    T operator()(std::any& aval)
    {
        typedef std::remove_reference_t<T> val_t;

        if constexpr (std::is_same_v<val_t, std::any>)
            return aval;

        val_t* val = std::any_cast<val_t>(&aval);
        if (val != nullptr)
        {
            return *val;
        }
        else
        {
            typedef std::reference_wrapper<val_t> ref_t;
            ref_t* rval = std::any_cast<ref_t>(&aval);
            if (rval != nullptr)
            {
                return rval->get();
            }
            else
            {
                typedef std::shared_ptr<val_t> ptr_t;
                ptr_t* pval = std::any_cast<ptr_t>(&aval);
                if (pval != nullptr)
                    return **pval;
                else
                    throw bad_extract(std::format("cannot extract C++ object of type '{}' from std::any, has type: {} ",
                                                  get_type_name<T>(),
                                                  name_demangle(aval.type().name())));
            }
        }
    }

    [[noreturn]] void throw_err(boost::python::object obj, std::string extra = "")
    {
        throw bad_extract(std::format("cannot extract C++ object of type '{}' from python object: {}{}",
                                      get_type_name<T>(), print_val(obj), extra));
    }
};

template <class T>
struct extract_param
    : public extract_param_base<T>
{};

template <class Type, size_t Dim>
struct extract_param<boost::multi_array_ref<Type, Dim>>
    : public extract_param_base<boost::multi_array_ref<Type, Dim>>
{
    auto operator()(boost::python::object obj)
    {
        try
        {
            return get_array<Type, Dim>(obj);
        }
        catch (InvalidNumpyConversion& e)
        {
            this->throw_err(obj, std::format(", reason: {}", e.what()));
        };
    }
};

template <>
struct extract_param<std::true_type>
    : public extract_param_base<std::true_type>
{
    auto operator()(boost::python::object obj)
    {
        if (!extract_param<bool>()(obj))
            throw_err(obj);
        return std::true_type();
    }
};

template <>
struct extract_param<std::false_type>
    : public extract_param_base<std::false_type>
{
    auto operator()(boost::python::object obj)
    {
        if (extract_param<bool>()(obj))
            throw_err(obj);
        return std::false_type();
    }
};

} //graph_tool namespace

#endif // DISPATCH
