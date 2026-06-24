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

#pragma GCC visibility push(default)

namespace graph_tool
{

class DispatchNotFound
    : public GraphException
{
public:
    DispatchNotFound(const std::type_info& dispatch,
                     const std::vector<std::string>& arg_types,
                     const std::vector<bool>& found);
    virtual ~DispatchNotFound() throw () {}

    const auto& get_arg_types() const { return _arg_types; }
    const auto& get_found() const { return _found; }

private:
    const std::type_info& _dispatch;
    std::vector<std::string> _arg_types;
    std::vector<bool> _found;
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

constexpr auto args_sub(auto args)
{
    return hana::fold
        (args, hana::tuple(),
         [](auto xs, auto x)
         {
             typedef typename decltype(+x)::type x_t;
             if constexpr (std::is_base_of_v<last_type_func_t, x_t>)
                 return hana::append(xs, x_t::get(hana::back(xs)));
             else if constexpr (std::is_base_of_v<all_types_func_t, x_t>)
                 return hana::append(xs, x_t::get(xs));
             else
                 return hana::append(xs, x);
         });
}

template <typename T>
struct print_type;

struct gt_dispatch_args
{
    bool uncheck = true;
    bool pass_types = false;
    bool gil_release = true;
};

template <gt_dispatch_args dargs = {}>
struct gt_dispatch
{
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

    template <class Dispatch, class... TRS>
    auto operator()(Dispatch&& a, TRS... trs)
    {
        return [=](auto&&... as)
               {
                   GILRelease gil(dargs.gil_release);

                   constexpr auto cp = hana::cartesian_product(hana::make_tuple(trs...));

                   std::array<bool, sizeof...(as)> pfound;
                   bool found = false;

                   hana::for_each
                       (cp,
                        [&](auto ats)
                        {
                            if (found)
                                return;

                            constexpr auto fats = args_sub(ats);

                            auto args =
                                hana::transform
                                (hana::zip(fats, hana::make_tuple(std::ref(as)...)),
                                 [&](auto&& tv)
                                 {
                                     using TR = typename decltype(+tv[0_c])::type;
                                     using T = std::decay_t<TR>;

                                     auto& x = tv[1_c].get();

                                     if constexpr (std::is_same_v<std::decay_t<decltype(x)>, std::any>)
                                     {
                                         T* val = std::any_cast<T>(&x);
                                         if (val == nullptr)
                                         {
                                             auto rval = std::any_cast<std::reference_wrapper<T>>(&x);
                                             if (rval == nullptr)
                                             {
                                                 auto sval = std::any_cast<std::shared_ptr<T>>(&x);
                                                 if (sval != nullptr)
                                                     val = sval->get();
                                             }
                                             else
                                             {
                                                 val = &rval->get();
                                             }
                                         }
                                         return val;
                                     }
                                     else // boost::python::object
                                     {
                                         try
                                         {
                                             if constexpr (std::is_reference_v<TR>)
                                                 return &extract_param<TR>()(x);
                                             else
                                                 return std::optional<T>(extract_param<TR>()(x));
                                         }
                                         catch (const bad_extract& e)
                                         {
                                             if constexpr (std::is_reference_v<TR>)
                                                 return (T*) nullptr;
                                             else
                                                 return std::optional<T>();
                                         }
                                     }
                                 });

                            found = true;
                            size_t idx = 0;
                            hana::for_each(args,
                                           [&](auto v)
                                           {
                                               bool vfound = true;
                                               if constexpr (std::is_pointer_v<decltype(v)>)
                                               {
                                                   if (v == nullptr)
                                                       vfound = false;
                                               }
                                               else
                                               {
                                                   if (!v)
                                                       vfound = false;
                                               }
                                               if (vfound)
                                                   pfound[idx] = true;
                                               if (!vfound)
                                                   found = false;
                                               ++idx;
                                           });

                            if (!found)
                                return;

                            if constexpr (!dargs.pass_types)
                                hana::unpack(args, [&](auto&... vals) { a(pmap(*vals)...); });
                            else
                                hana::unpack(args, [&](auto&... vals) { a(fats, pmap(*vals)...); });
                        });

                   if (!found)
                   {
                       auto get_type =
                           [](auto&& a)
                           {
                               if constexpr (std::is_same_v<std::decay_t<decltype(a)>, std::any>)
                                   return name_demangle(a.type().name());
                               else
                                   return boost::python::extract<std::string>(a.attr("__repr__")());
                           };
                       std::vector<std::string> arg_types{get_type(as)...};
                       std::vector<bool> afound(pfound.begin(), pfound.end());
                       throw DispatchNotFound(typeid(Dispatch), arg_types, afound);
                   }
               };
    }

    bool _gil_release;
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
struct extract_param
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
                throw bad_extract("cannot extract 'any' object");
            std::any& aval = extract();
            typedef std::remove_reference_t<T> val_t;
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
                        throw bad_extract("has type: " +
                                          name_demangle(aval.type().name()));
                }
            }
        }
    }
};

template <class Type, size_t Dim>
struct extract_param<boost::multi_array_ref<Type, Dim>>
{
    auto operator()(boost::python::object obj)
    {
        try
        {
            return get_array<Type, Dim>(obj);
        }
        catch (InvalidNumpyConversion& e)
        {
            throw bad_extract(e.what());
        };
    }
};

template <>
struct extract_param<std::true_type>
{
    auto operator()(boost::python::object obj)
    {
        if (!extract_param<bool>()(obj))
            throw bad_extract();
        return std::true_type();
    }
};

template <>
struct extract_param<std::false_type>
{
    auto operator()(boost::python::object obj)
    {
        if (extract_param<bool>()(obj))
            throw bad_extract();
        return std::false_type();
    }
};

} //graph_tool namespace

#endif // DISPATCH
