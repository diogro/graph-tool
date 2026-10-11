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

#include <boost/python.hpp>
#include <boost/preprocessor/comparison/equal.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/seq/size.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/seq/transform.hpp>
#include <boost/preprocessor/seq/filter.hpp>
#include <boost/preprocessor/seq/first_n.hpp>
#include <boost/vmd/vmd.hpp>

#include "graph.hh"
#include "selectors.hh"
#include "properties.hh"
#include "graph_filtering.hh"
#include "graph_util.hh"

#include "numpy_bind.hh"

#include "config.h"

#include <memory>
#include <print>

namespace graph_tool
{

constexpr auto get_ref_tr(auto vals)
{
    return hana::transform
        (vals,
         [](auto t)
         {
             typedef typename decltype(+t)::type T;
             return hana::type_c<T&>;
         });
}

inline constexpr auto all_graph_view_refs = get_ref_tr(all_graph_views);

template <size_t N>
struct nth
{
    template <class T, class... Ts>
    decltype(auto) operator()(T&&, Ts&&... as) const
    {
        static_assert(N < sizeof...(as) + 1, "wrong number of arguments");
        return nth<N-1>()(std::forward<Ts>(as)...);
    }
};

template <>
struct nth<0>
{
    template <class T, class... Ts>
    decltype(auto) operator()(T&& a, Ts&&...) const
    {
        return std::forward<T>(a);
    }
};

template <size_t N, class T, class... Ts>
struct nth_t
{
    typedef std::conditional_t<N == 0,
                               T,
                               typename nth_t<N-1, Ts...>::type> type;
};

template <size_t N, class T>
struct nth_t<N, T>
{
    typedef T type;
};

template <template <class...> class State>
struct StateFactory
{
    template <class... Ts>
    struct apply
    {
        typedef State<Ts...> type;
    };
};

constexpr auto get_type_map(auto plist, auto names)
{
    constexpr auto pairs =
        hana::transform(hana::zip(plist, names),
                        [](auto x)
                        {
                            return hana::make_pair(x[1_c], x[0_c]);
                        });
    return hana::to<hana::map_tag>(pairs);
}

template <class Factory, class... TRS>
struct StateWrap
{
    template <class... Ts>
    struct FactoryWrap
    {
        typedef typename Factory::template apply<Ts...>::type state_t;

        template <class... ATs>
        auto operator()(ATs&&... args) const
        {
            return std::make_shared<state_t>(std::forward<ATs>(args)...);
        };
    };

    template <class F>
    static void dispatch(F&& f, auto names)
    {
        auto final_dispatch =
            [&](auto... args)
            {
                typedef typename FactoryWrap<typename decltype(args)::type...>::state_t
                    state_t;
                state_t* ptr(nullptr);
                f(ptr);
            };

        constexpr auto cp = hana::cartesian_product(hana::make_tuple(TRS()...));

        hana::for_each
            (cp,
             [&](auto ats)
             {
                 auto sargs =
                     hana::fold
                     (ats, hana::tuple(),
                      [&](auto xs, auto x)
                      {
                          typedef typename decltype(+x)::type x_t;
                          if constexpr (std::is_base_of_v<last_type_func_t, x_t>)
                              return hana::append(xs, x_t::get(hana::back(xs)));
                          else if constexpr (std::is_base_of_v<all_types_func_t, x_t>)
                              return hana::append(xs, x_t::get(get_type_map(xs, names)));
                          else
                              return hana::append(xs, x);
                      });

                 auto args =
                     hana::transform
                     (sargs,
                      [](auto t)
                      {
                          typedef typename decltype(+t)::type T;
                          return hana::type_c<typename boost::uncheck<T>::type>;
                      });
                 hana::unpack(args, final_dispatch);
             });
    }

    template <class F>
    static void dispatch(boost::python::object& ostate,
                         F&& f,
                         auto names,
                         bool throw_not_found = true)
    {
        bool found = false;
        dispatch
            ([&](auto* s)
             {
                 typedef std::decay_t<decltype(*s)> state_t;
                 boost::python::extract<state_t&> get_state(ostate);
                 if (get_state.check())
                 {
                     state_t& state = get_state();
                     f(state);
                     found = true;
                 }
             }, names);

        if (!found && throw_not_found)
            throw GraphException("dispatch not found for: " +
                                 get_type_name<StateWrap>());
    }

    template <class T, class Names>
    struct all_types_func_wrap
        : public T
    {
        constexpr static decltype(auto) get(auto ps)
        {
            return T::get(get_type_map(ps, Names()));
        }
    };

    template <class... Extra, class F>
    static void make_dispatch(boost::python::object& ostate,
                              F&& f, auto names,
                              Extra&&... extra)
    {
        auto convert_tr =
            [&](auto tr)
            {
                return hana::transform
                    (tr,
                     [&](auto x)
                     {
                         using T = typename decltype(x)::type;
                         if constexpr (std::is_base_of_v<all_types_func_t, T>)
                             return hana::type_c<all_types_func_wrap<T, decltype(names)>>;
                         else
                             return x;
                     });
            };

        auto dispatch =
            gt_dispatch<{.pass_types=true, .gil_release=false}>()
            ([&](auto ats, auto&&... args)
             {
                 hana::unpack
                     (ats,
                      [&](auto... ts)
                      {
                          auto state =
                              FactoryWrap<typename boost::uncheck<typename decltype(ts)::type>::type...>()
                              (std::forward<Extra>(extra)...,
                               std::forward<decltype(args)>(args)...);
                          f(state);
                      });
             }, convert_tr(TRS())...);

        try
        {
            hana::unpack(hana::transform(names,
                                         [&](auto name)
                                         {
                                             return ostate.attr(name.c_str());
                                         }),
                         dispatch);
        }
        catch (const DispatchNotFound& e)
        {
            std::string msg;
            size_t idx = 0;
            hana::for_each
                (names,
                 [&](auto name)
                 {
                     if (e.get_pos() == idx++)
                         msg = std::format("Cannot extract parameter \"{}\" of desired types {} from value: {}\n",
                                           name.c_str(), e.get_trs(), e.get_val());
                 });
            throw ValueException(msg);
        }
    }
};

// workaround for empty sequences
#define BOOST_PP_SEQ_ENUM_0

#define _PARAM_TYPEDEF(r, pack, pos, name)                                     \
    typedef std::decay_t<typename nth_t<pos, pack...>::type>                   \
    BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(3, 0, name), _t);
#define GET_PARAMS_TYPEDEF(pack, names)                                        \
    BOOST_PP_SEQ_FOR_EACH_I(_PARAM_TYPEDEF, pack, names)

#define _PARAM_USING(r, base, pos, name)                                       \
    using base :: BOOST_PP_CAT(_,BOOST_PP_TUPLE_ELEM(3, 0, name));
#define GET_PARAMS_USING(base, names)                                          \
    BOOST_PP_SEQ_FOR_EACH_I(_PARAM_USING, base, names)

#define _PARAM_DEFINE(r, pack, pos, name)                                      \
    typename nth_t<pos, pack...>::type                                         \
    BOOST_PP_CAT(_, BOOST_PP_TUPLE_ELEM(3, 0, name));
#define GET_PARAMS_DEFINE(pack, names)                                         \
    BOOST_PP_SEQ_FOR_EACH_I(_PARAM_DEFINE, pack, names)

#define _PARAM_INIT(r, pack, pos, name)                                        \
    BOOST_PP_CAT(_, BOOST_PP_TUPLE_ELEM(3, 0, name))(nth<pos>()(pack...)),
#define GET_PARAMS_INIT(pack, names)                                           \
    BOOST_PP_SEQ_FOR_EACH_I                                                    \
       (_PARAM_INIT, pack,                                                     \
        BOOST_PP_SEQ_FIRST_N(BOOST_PP_DEC(BOOST_VMD_SEQ_SIZE(names)),          \
                             names))                                           \
   BOOST_PP_CAT(_, BOOST_PP_TUPLE_ELEM                                         \
                (3, 0,                                                         \
                 BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_VMD_SEQ_SIZE(names)),    \
                                   names)))                                    \
        (nth<BOOST_PP_DEC(BOOST_VMD_SEQ_SIZE(names))>()(pack...))

#define _PARAM_STRING(r, data, pos, name)                                      \
    BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(3, 0, name)),
#define GET_PARAMS_NAMES(names)                                                \
    BOOST_PP_SEQ_FOR_EACH_I                                                    \
       (_PARAM_STRING,,                                                        \
        BOOST_PP_SEQ_FIRST_N(BOOST_PP_DEC(BOOST_VMD_SEQ_SIZE(names)),          \
                             names))                                           \
    BOOST_PP_STRINGIZE                                                         \
       (BOOST_PP_TUPLE_ELEM                                                    \
        (3, 0, BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_VMD_SEQ_SIZE(names)),      \
                                 names)))

// as hana strings
#define _PARAM_HSTRING(r, data, pos, name)                                     \
    BOOST_HANA_STRING(BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(3, 0, name))),
#define GET_PARAMS_HNAMES(names)                                               \
    BOOST_PP_SEQ_FOR_EACH_I                                                    \
       (_PARAM_HSTRING,,                                                       \
        BOOST_PP_SEQ_FIRST_N(BOOST_PP_DEC(BOOST_VMD_SEQ_SIZE(names)),          \
                             names))                                           \
    BOOST_HANA_STRING(BOOST_PP_STRINGIZE                                       \
       (BOOST_PP_TUPLE_ELEM                                                    \
        (3, 0, BOOST_PP_SEQ_ELEM(BOOST_PP_DEC(BOOST_VMD_SEQ_SIZE(names)),      \
                                 names))))

#define _PARAM_SPEC_TYPEDEF(r, pack, pos, name)                                \
    typedef State::BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(3, 0, name), _t)           \
        BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(3, 0, name), _t);
#define GET_PARAMS_SPEC_TYPEDEF(names)                                         \
    BOOST_PP_SEQ_FOR_EACH_I(_PARAM_SPEC_TYPEDEF, r, names)

#define _PARAM_SPEC_DEFINE(r, pack, pos, name)                                 \
    BOOST_PP_CAT(BOOST_PP_TUPLE_ELEM(3, 0, name), _t)&                         \
        BOOST_PP_CAT(_, BOOST_PP_TUPLE_ELEM(3, 0, name));
#define GET_PARAMS_SPEC_DEFINE(names)                                          \
    BOOST_PP_SEQ_FOR_EACH_I(_PARAM_SPEC_DEFINE, r, names)

#define _PARAM_SPEC_INIT(r, pack, pos, name)                                   \
    BOOST_PP_CAT(_, BOOST_PP_TUPLE_ELEM(3, 0, name))                           \
        (state.BOOST_PP_CAT(_, BOOST_PP_TUPLE_ELEM(3, 0, name))),
#define GET_PARAMS_SPEC_INIT(names)                                            \
    BOOST_PP_SEQ_FOR_EACH_I(_PARAM_SPEC_INIT, r, names)

#define GEN_STATE_BASE(base_name, state_params)                                \
    template <class... Ts>                                                     \
    class base_name                                                            \
    {                                                                          \
    public:                                                                    \
        GET_PARAMS_TYPEDEF(Ts, state_params)                                   \
        template <class... ATs>                                                \
        base_name(ATs&&... args)                                               \
            : GET_PARAMS_INIT(args, state_params) {}                           \
        base_name(const base_name&) = delete;                                  \
        static inline auto __param_names =                                     \
            hana::make_tuple(GET_PARAMS_HNAMES(state_params));                 \
    protected:                                                                 \
        GET_PARAMS_DEFINE(Ts, state_params)                                    \
    };

#define _EMPTY_PROTECT(pred, seq, dummy)                                       \
    BOOST_VMD_SEQ_POP_BACK(pred(BOOST_VMD_SEQ_PUSH_BACK(seq, dummy)))

#define _IS_TR(r, data, elem) BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(3, 2, elem),1)
#define _GET_T(r, data, elem)                                                  \
    BOOST_PP_IIF(_IS_TR(r, data, elem),                                        \
                 BOOST_PP_TUPLE_ELEM(3, 1, elem),                              \
                 decltype(hana::tuple_t<BOOST_PP_TUPLE_ELEM(3, 1, elem)>))
#define GET_PARAMS_TRS(params)                                                 \
    BOOST_PP_SEQ_TRANSFORM(_GET_T,,params)

#define _IS_T(r, data, elem) BOOST_PP_EQUAL(BOOST_PP_TUPLE_ELEM(3, 2, elem),0)
#define _GET_TYPE(r, data, elem)                                               \
    BOOST_PP_TUPLE_ELEM(3, 1, elem)
#define GET_PARAMS_TYPES(params)                                               \
    BOOST_PP_SEQ_TRANSFORM(_GET_TYPE,,BOOST_PP_SEQ_FILTER(_IS_T, 1, params))

#define GEN_DISPATCH(gen_name, state_name, params)                             \
    struct gen_name                                                            \
    {                                                                          \
        typedef StateWrap<BOOST_PP_SEQ_ENUM((StateFactory<state_name>)         \
                                            GET_PARAMS_TRS(params))>           \
            state_wrap_t;                                                      \
        template <class F, class... Extra>                                     \
        static void make_dispatch(boost::python::object ostate, F&& f,         \
                                  Extra&&... args)                             \
        {                                                                      \
            constexpr auto names =                                             \
                boost::hana::make_tuple(GET_PARAMS_HNAMES(params));            \
            state_wrap_t::make_dispatch                                        \
                (ostate, std::forward<F>(f), names,                            \
                 std::forward<Extra>(args)...);                                \
        }                                                                      \
        template <class F>                                                     \
        static void dispatch(F&& f)                                            \
        {                                                                      \
            constexpr auto names =                                             \
                boost::hana::make_tuple(GET_PARAMS_HNAMES(params));            \
            state_wrap_t::dispatch                                             \
                (std::forward<F>(f), names);                                   \
        }                                                                      \
        template <class F>                                                     \
        static void dispatch(boost::python::object state, F&& f,               \
                             bool throw_not_found = true)                      \
        {                                                                      \
            constexpr auto names =                                             \
                boost::hana::make_tuple(GET_PARAMS_HNAMES(params));            \
            state_wrap_t::dispatch                                             \
                (state, std::forward<F>(f), names, throw_not_found);           \
        }                                                                      \
    };


#define _PARAM_DEF(r, data, pos, elem)                                         \
    BOOST_PP_TUPLE_ELEM(1, elem) BOOST_PP_TUPLE_ELEM(0, elem);
#define DEFINE_STRUCT(name, params)                                            \
    struct name                                                                \
    {                                                                          \
        BOOST_PP_SEQ_FOR_EACH_I(_PARAM_DEF,, params)                           \
    };
#define DEFINE_STRUCT_BASED(name, based, params)                               \
    struct name                                                                \
        : public based                                                         \
    {                                                                          \
        BOOST_PP_SEQ_FOR_EACH_I(_PARAM_DEF,, params)                           \
    };

#define _PARAM_RW(r, sname, pos, elem)                                         \
    .def_readwrite(BOOST_PP_STRINGIZE(BOOST_PP_TUPLE_ELEM(0, elem)),           \
                   &sname::BOOST_PP_TUPLE_ELEM(0, elem))

#define EXPORT_STRUCT(name, params)                                            \
    try                                                                        \
    {                                                                          \
        __MOD__::get_class<name>();                                            \
    }                                                                          \
    catch (ObjNotFound&)                                                       \
    {                                                                          \
        __MOD__::get_class<name>(BOOST_PP_STRINGIZE(name))                     \
            BOOST_PP_SEQ_FOR_EACH_I(_PARAM_RW, name, params);                  \
    }

#define EXPORT_STRUCT_BASED(name, base, params)                                \
    try                                                                        \
    {                                                                          \
        using namespace boost::python;                                         \
        __MOD__::get_class<name, bases<base>>();                               \
    }                                                                          \
    catch (ObjNotFound&)                                                       \
    {                                                                          \
        using namespace boost::python;                                         \
        __MOD__::get_class<name, bases<base>>(BOOST_PP_STRINGIZE(name))        \
            BOOST_PP_SEQ_FOR_EACH_I(_PARAM_RW, name, params);                  \
    }


} // namespace graph_tool
