#ifndef POTTS_MULTIFLIP_MCMC_HH
#define POTTS_MULTIFLIP_MCMC_HH

#include "config.h"

#include <vector>
#include <algorithm>

#include "graph_tool.hh"
#include "inference/support/graph_state.hh"
#include "inference/support/util.hh"

#include "graph_potts.hh"

#include "idx_map.hh"
#include "inference/loops/merge_split.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_POTTS_STATE_params(State)                                         \
    ((__class__,&, decltype(hana::tuple_t<python::object>), 1))                \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((psingle,, double, 0))                                                    \
    ((psplit,, double, 0))                                                     \
    ((pmerge,, double, 0))                                                     \
    ((pmergesplit,, double, 0))                                                \
    ((pmovelabel,, double, 0))                                                 \
    ((nproposal, &, vector<size_t>&, 0))                                       \
    ((nacceptance, &, vector<size_t>&, 0))                                     \
    ((gibbs_sweeps,, size_t, 0))                                               \
    ((entropy_args,, eargs_t, 0))                                              \
    ((verbose,, int, 0))                                                       \
    ((force_move,, bool, 0))                                                   \
    ((niter,, double, 0))

template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCPottsStateBase, MCMC_POTTS_STATE_params(State))

    template <class... Ts>
    class MCMCPottsStateImp
        : public MCMCPottsStateBase<Ts...>,
          public MergeSplitStateBase
    {
    public:
        GET_PARAMS_USING(MCMCPottsStateBase<Ts...>,
                         MCMC_POTTS_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_POTTS_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCPottsStateImp(ATs&&... as)
           : MCMCPottsStateBase<Ts...>(as...)
        {
        }

        constexpr static group_t _null_group = std::numeric_limits<group_t>::max();

        constexpr static double _psrandom = 1;
        constexpr static double _psscatter = 0;
        constexpr static double _pscoalesce = 0;

        template <class F>
        void iter_nodes(F&& f)
        {
            for (auto v : vertices_range(_state._g))
                f(v);
        }

        template <class F>
        void iter_groups(F&& f)
        {
            for (size_t r = 0; r < _state._q; ++r)
            {
                if (_state._nr[r] > 0)
                    f(group_t(r));
            }
        }

        auto get_group(size_t v)
        {
            return _state._b[v];
        }

        template <bool sample_branch=true, class RNG, class VS = std::array<group_t,0>>
        group_t sample_new_group(size_t, RNG& rng, VS&& = VS())
        {
            return uniform_sample(_state._empty_groups.begin(),
                                  _state._empty_groups.end(), rng);
        }

        void move_node(size_t v, group_t r, bool)
        {
            _state.move_vertex(v, r);
        }

        double virtual_move(size_t v, group_t r, group_t s)
        {
            return _state.virtual_move(v, r, s, _entropy_args);
        }

        template <class RNG>
        auto sample_group(size_t v, bool, RNG& rng)
        {
            return _state.sample_group(v, rng);
        }

        double get_move_prob(size_t v, group_t r, group_t s, bool, bool reverse)
        {
            return _state.get_move_lprob(v, r, s, reverse);
        }

        size_t get_max_B()
        {
            return _state._q;
        }
    };

    using gmap_t = idx_map<size_t, idx_set<size_t, true>, false, true, true>;

    template <class T>
    using iset = idx_set<T>;

    template <class T, class V>
    using imap = idx_map<T, V>;

    template <class... Ts>
    class MCMCPottsState:
        public MergeSplit<MCMCPottsStateImp<Ts...>,
                          size_t,
                          group_t,
                          iset,
                          imap,
                          iset,
                          gmap_t, true, true>
    {
    public:
        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCPottsState(ATs&&... as)
           : MergeSplit<MCMCPottsStateImp<Ts...>,
                        size_t,
                        group_t,
                        iset,
                        imap,
                        iset,
                        gmap_t, true, true>(as...)
        {}
    };
};

} // graph_tool namespace

#endif //POTTS_MULTIFLIP_MCMC_HH
