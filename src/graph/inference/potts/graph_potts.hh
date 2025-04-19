#ifndef GRAPH_POTTS_HH
#define GRAPH_POTTS_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "random.hh"
#include "../support/util.hh"
#include "../support/graph_state.hh"

#include "idx_map.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

// parameters to the description length
struct eargs_t
{
};

typedef int64_t group_t;
typedef vprop_map_t<group_t> vmap_t;
typedef vprop_map_t<std::vector<double>> tmap_t;
typedef eprop_map_t<double> wmap_t;
typedef boost::multi_array_ref<double, 2> f_t;

#define POTTS_STATE_params                                                     \
   ((g, &, decltype(all_graph_views), 1))                                      \
   ((b,, vmap_t, 0))                                                           \
   ((theta,, tmap_t, 0))                                                       \
   ((w,, wmap_t, 0))                                                           \
   ((f,, f_t, 0))

GEN_STATE_BASE(PottsStateBase, POTTS_STATE_params)

template <class... Ts>
class PottsState
    : public PottsStateBase<Ts...>
{
public:
    GET_PARAMS_USING(PottsStateBase<Ts...>, POTTS_STATE_params)
    GET_PARAMS_TYPEDEF(Ts, POTTS_STATE_params)
    using typename PottsStateBase<Ts...>::args_t;
    using PottsStateBase<Ts...>::dispatch_args;

    template <class... ATs,
              typename std::enable_if_t<sizeof...(ATs) == sizeof...(Ts)>* = nullptr>
    PottsState(ATs&&... args)
        : PottsStateBase<Ts...>(std::forward<ATs>(args)...),
          _q(_f.shape()[0]), _nr(_q)
    {
        for (auto v : vertices_range(_g))
        {
            _nr[_b[v]]++;
            _theta[v].resize(_q);
        }

        for (size_t r = 0; r < _q; ++r)
        {
            if (_nr[r] == 0)
                _empty_groups.insert(r);
        }
    }

    size_t _q;
    std::vector<size_t> _nr;
    idx_set<group_t> _empty_groups;

    typedef eargs_t _entropy_args_t;

    // Computes the DL difference of moving vertex from group r to nr
    double virtual_move(size_t v, group_t r, group_t nr, eargs_t&)
    {
        double dS = 0;
        for (auto e : out_edges_range(v, _g))
        {
            if (target(e, _g) == v)
            {
                dS += _f[r][r] * _w[e];
                dS -= _f[nr][nr] * _w[e];
            }
            else
            {
                size_t s = _b[target(e, _g)];
                dS += _f[r][s] * _w[e];
                dS -= _f[nr][s] * _w[e];
            }
        }

        for (auto e : in_edges_range(v, _g))
        {
            if (source(e, _g) == v)
            {
                dS += _f[r][r] * _w[e];
                dS -= _f[nr][nr] * _w[e];
            }
            else
            {
                size_t s = _b[source(e, _g)];
                dS += _f[s][r] * _w[e];
                dS -= _f[s][nr] * _w[e];
            }
        }

        dS += _theta[v][r];
        dS -= _theta[v][nr];

        return dS;
    }

    // Moves vertex v to group s
    void move_vertex(size_t v, group_t s)
    {
        auto r = _b[v];
        _b[v] = s;
        _nr[r]--;
        _nr[s]++;
        if (r != s)
        {
            if (_nr[r] == 0)
                _empty_groups.insert(r);
            if (_nr[s] == 1)
                _empty_groups.erase(s);
        }
    }

    // Sample a new group membership for vertex v
    group_t sample_group(size_t, rng_t& rng)
    {
        std::uniform_int_distribution<group_t> sample(0, _q-1);
        return sample(rng);
    }

    // Calculate the log-probability of moving vertex v from group r to s (or
    // the reverse probability if reverse == true)
    double get_move_lprob(size_t, group_t, group_t, bool)
    {
        return 0;
    }

    double entropy(const eargs_t&)
    {
        double S = 0;
        for (auto e : edges_range(_g))
        {
            size_t r = _b[source(e, _g)];
            size_t s = _b[target(e, _g)];
            S -= _f[r][s] * _w[e];
        }

        for (auto v : vertices_range(_g))
            S -= _theta[v][_b[v]];

        return S;
    }
};
} // graph_tool namespace

#endif //GRAPH_POTTS_HH
