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

#ifndef GRAPH_BLOCKMODEL_SPEC_HH
#define GRAPH_BLOCKMODEL_SPEC_HH

#include "config.h"

#include <any>
#include <cstdint>

#include "graph_tool.hh"

#include "idx_map.hh"

#include "../support/graph_state.hh"

#include "entries.hh"
#include "partition.hh"
#include "deg_partition.hh"

#define __MOD__ inference
#include "module_registry.hh"

#define BLOCKMODEL_EARGS_params                                                \
    ((dense, bool))                                                            \
    ((multigraph, bool))                                                       \
    ((adjacency, bool))                                                        \
    ((deg_entropy, bool))                                                      \
    ((parallel_entropy, bool))                                                 \
    ((partition_dl, bool))                                                     \
    ((degree_dl, bool))                                                        \
    ((degree_dl_kind, deg_dl_kind))                                            \
    ((edges_dl, bool))                                                         \
    ((beta_dl, double))                                                        \
    ((Bfield, bool))                                                           \
    ((propagate, bool))                                                        \
    ((constants, bool))

namespace graph_tool
{

template <bool directed, class Deg, class Group, bool>
class DegStats;

struct BlockStateSpec
{
    typedef decltype(all_graph_view_refs) g_tr;

    struct get_bg_t
        : public last_type_func_t
    {
        template <class T>
        constexpr static auto get(T)
        {
            typedef std::decay_t<typename T::type> g_t;
            typedef std::decay_t<decltype(base_graph(std::declval<g_t>()))> bg_t;
            if constexpr (is_directed_v<g_t>)
                return hana::type_c<bg_t&>;
            else
                return hana::type_c<boost::undirected_adaptor<bg_t>&>;
        }
    };

    constexpr inline static auto get_bg_type = hana::tuple_t<get_bg_t>;
    typedef decltype(get_bg_type) bg_tr;

    typedef decltype(hana::tuple_t<std::true_type, std::false_type>) flat_map_tr;

    typedef int64_t count_t;
    typedef int64_t group_t;
    static constexpr group_t null_group = std::numeric_limits<group_t>::max();
    typedef int64_t vcount_t;

    template <class Val>
    using vprop_t = vprop_map_t<Val>;

    typedef vprop_t<count_t> cmap_t;
    typedef vprop_t<group_t> bmap_t;
    typedef decltype(hana::tuple_t<eprop_map_t<count_t>>) wmap_tr;
    typedef decltype(same_arg_type) bwmap_tr;
    typedef decltype(hana::tuple_t<vprop_t<count_t>>) mrp_tr;
    typedef decltype(hana::tuple_t<vprop_t<count_t>>) mrm_tr;
    typedef decltype(hana::tuple_t<vprop_t<count_t>>) vweight_tr;
    typedef decltype(hana::tuple_t<vprop_t<count_t>>) wr_tr;

    typedef vprop_t<std::vector<double>> bfield_t;

    template <class State>
    struct m_entries_t
    {
        typedef EntrySet<is_directed_v<typename State::g_t>,
                         group_t,
                         count_t,
                         GraphInterface::edge_t,
                         typename State::weight_t> type;
    };

    template <class State>
    struct partition_stats_t
    {
        typedef PartitionStats<group_t> type;
    };

    template <class State>
    struct deg_t
    {
        typedef int64_t type;
    };

    template <class State>
    struct deg_stats_t
    {

        typedef DegStats<is_directed_v<typename State::g_t>,
                         typename deg_t<State>::type,
                         typename State::group_t,
                         true> type;
    };

    template <class State>
    struct deg_map_t
    {
        typedef typename deg_t<State>::type d_t;
        typedef vprop_map_t<std::tuple<d_t, d_t>> type;
    };

    typedef idx_map<group_t, idx_set<size_t, {.shared_pos=true}>,
                    {.shared_tpos=true}> groups_t;

    constexpr static bool has_merge = true;
    constexpr static bool has_parallel_move = true;
    constexpr static bool has_relabel = false;
    constexpr static bool has_parallel_relabel = false;
    constexpr static bool edges_dl = true;

    DEFINE_STRUCT(entropy_args_t, BLOCKMODEL_EARGS_params);

    template <class State>
    struct coupled_state_t
    {
        static inline auto anames = State::__param_names;
        static inline auto args_t =
            hana::transform(hana::zip(State::args_t, anames),
                            [](auto x) constexpr
                            {
                                if constexpr (x[1_c] == BOOST_HANA_STRING("g"))
                                    return hana::type_c<typename State::bg_t&>;
                                else
                                    return x[0_c];
                            });

        constexpr static inline auto state_t =
            hana::unpack(args_t,
                         [](auto... as)
                         {
                             return hana::type_c<typename State::template state_t<typename decltype(as)::type...>>;
                         });

        typedef decltype(state_t)::type type;
    };

    static void python_export()
    {
        using namespace boost::python;
        EXPORT_STRUCT(entropy_args_t, BLOCKMODEL_EARGS_params)

        try
        {
            __MOD__::get_obj<enum_<deg_dl_kind>>();
        }
        catch (ObjNotFound&)
        {
            __MOD__::get_obj<enum_<deg_dl_kind>>("deg_dl_kind")
                .value("uniform", deg_dl_kind::UNIFORM)
                .value("distributed", deg_dl_kind::DIST);
        }
    }

    template <class State>
    static constexpr void python_export_state(auto&) {}

    template <class State>
    struct Imp
    {
        Imp(State&) {}
        Imp() {}

        constexpr void post_init() const {}

        constexpr double entropy(const entropy_args_t&) const { return 0; }

        constexpr double entries_dS(const entropy_args_t&, auto&) const { return 0; }
        constexpr double virtual_move_dS(auto&, auto&, const entropy_args_t&, auto&) const { return 0; }
        constexpr double virtual_merge_dS(auto, auto, auto&, auto&) const { return 0; }

        template <bool concurrent=false>
        constexpr void move_vertex(auto, auto, auto&, auto&&) {}
        constexpr void move_vertex(auto, auto) {}
        template <bool concurrent=false, class Deltas = std::monostate>
        constexpr bool apply_delta(auto&, Deltas&& = std::monostate()) { return false; }
        constexpr void merge_groups(auto&, auto&, auto&) const {}
        constexpr void add_group(size_t) const {}

        template <bool Add, bool Deplete=true>
        constexpr void modify_edge(auto, auto, auto&, auto&) const {}
        constexpr double modify_edge_dS(auto, auto, auto&, auto&, auto&) const { return 0; }

        decltype(auto) random_neighbors_range(auto v, auto&& g) const
        {
            return graph_tool::all_neighbors_range(v, g);
        }

        constexpr void propagate_delta(auto&, group_t, group_t, auto&) const {}

        template <bool lock>
        constexpr void update_rs(auto&, auto&, auto&&...) const {}

        constexpr void couple_state(auto&) const {}

        constexpr void update_group_weight(auto&, auto&&) const {}
        constexpr void set_vertex_weight(auto&, auto&&) const {}
        constexpr void coupled_resize_vertices(auto) const {}

        constexpr void set_concurrent(bool, size_t) const {}
        constexpr void set_concurrent_delta(auto&&, bool) const {}

        constexpr bool check_edge_counts() const { return true; }

        constexpr int group_label(auto) const { return 0; }
        constexpr int sample_group_label(auto, auto&) const { return 0; }
        constexpr double get_group_label_prob(auto, auto) const { return 0; }
        constexpr double relabel_group_dS(auto, auto) const { return 0; }
        constexpr void relabel_group(auto, auto) const {}
   };
};

template <class T1, class T2>
struct thresh_diff_imp
{
    auto operator()(T1 before, T2 after, int t = 0)
    {
        return (quiet_less_equal(before, t) && quiet_greater(after, t)) ? 1 : 0;
    }
};

template <class T1, class T2>
auto thresh_diff(T1&& before, T2&& after, int t = 0)
{
    return thresh_diff_imp<T1, T2>()(std::forward<T1>(before),
                                     std::forward<T2>(after), t);
}

template <>
struct extract_param<eprop_map_t<int64_t>>
{
    auto operator()(boost::python::object oemap)
    {
        try
        {
            auto a = python::extract<std::any>(oemap.attr("_get_any")())();
            return std::any_cast<eprop_map_t<int64_t>>(a);
        }
        catch (std::bad_any_cast&)
        {
            oemap = oemap.attr("copy")("int64_t");
            auto a = python::extract<std::any>(oemap.attr("_get_any")())();
            return std::any_cast<eprop_map_t<int64_t>>(a);
        }
    }
};

}

#include "entropy.hh"
#include "partition.hh"
#include "entries.hh"
#include "emat.hh"
#include "elist.hh"

#endif //GRAPH_BLOCKMODEL_SPEC_HH
