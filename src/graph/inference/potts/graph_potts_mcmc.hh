#ifndef POTTS_MCMC_HH
#define POTTS_MCMC_HH

#include "config.h"

#include <vector>

#include "graph_tool.hh"
#include "random.hh"
#include "inference/support/graph_state.hh"
#include "inference/loops/mcmc_loop.hh"

namespace graph_tool
{
using namespace boost;
using namespace std;

#define MCMC_POTTS_STATE_params(State)                                       \
    ((__class__,&, decltype(hana::tuple_t<python::object>), 1))                \
    ((state, &, State&, 0))                                                    \
    ((beta,, double, 0))                                                       \
    ((entropy_args,, eargs_t, 0))                                              \
    ((sequential,, bool, 0))                                                   \
    ((deterministic,, bool, 0))                                                \
    ((verbose,, int, 0))                                                       \
    ((niter,, size_t, 0))


template <class State>
struct MCMC
{
    GEN_STATE_BASE(MCMCPottsStateBase, MCMC_POTTS_STATE_params(State))

    template <class... Ts>
    class MCMCPottsState
        : public MCMCPottsStateBase<Ts...>,
          public MetropolisStateBase
    {
    public:
        GET_PARAMS_USING(MCMCPottsStateBase<Ts...>,
                         MCMC_POTTS_STATE_params(State))
        GET_PARAMS_TYPEDEF(Ts, MCMC_POTTS_STATE_params(State))

        template <class... ATs,
                  typename std::enable_if_t<sizeof...(ATs) ==
                                            sizeof...(Ts)>* = nullptr>
        MCMCPottsState(ATs&&... as)
           : MCMCPottsStateBase<Ts...>(as...),
            _vlist(num_vertices(_state._g))
        {
            std::iota(_vlist.begin(), _vlist.end(), 0);
        }

        std::vector<size_t> _vlist;
        constexpr static group_t _null_move = std::numeric_limits<group_t>::max();

        auto node_state(size_t v)
        {
            return _state._b[v];
        }

        template <class RNG>
        auto move_proposal(size_t v, RNG& rng)
        {
            auto s = _state.sample_group(v, rng);
            if (s == node_state(v))
                return _null_move;
            return s;
        }

        std::tuple<double, double>
        virtual_move_dS(size_t v, size_t s)
        {
            size_t r =  node_state(v);
            if (r == s)
                return {0., 0.};

            double dS = _state.virtual_move(v, r, s, _entropy_args);
            double a = 0;
            // if (!std::isinf(_beta))
            // {
            //     double pf = _state.get_move_lprob(v, r, s, false);
            //     double pb = _state.get_move_lprob(v, s, r, true);
            //     a = pb - pf;
            // }
            return {dS, a};
        }

        void perform_move(size_t v, group_t s)
        {
            _state.move_vertex(v, s);
        }

        bool is_deterministic()
        {
            return _deterministic;
        }

        bool is_sequential()
        {
            return _sequential;
        }

        auto& get_vlist()
        {
            return _vlist;
        }

        double get_beta()
        {
            return _beta;
        }

        size_t get_niter()
        {
            return _niter;
        }
    };
};


} // graph_tool namespace

#endif //POTTS_MCMC_HH
