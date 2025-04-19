#include "graph_tool.hh"
#include "random.hh"

#include <boost/python.hpp>

#include "graph_potts.hh"
#include "graph_potts_mcmc.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(potts_state, PottsState, POTTS_STATE_params)

template <class State>
GEN_DISPATCH(mcmc_potts_state, MCMC<State>::template MCMCPottsState,
             MCMC_POTTS_STATE_params(State))

python::object potts_mcmc_sweep(python::object omcmc_state,
                                  python::object opotts_state,
                                  rng_t& rng)
{
    python::object ret;
    potts_state::dispatch
        (opotts_state,
         [&](auto& potts_state)
         {
             typedef typename std::remove_reference<decltype(potts_state)>::type
                 state_t;

             mcmc_potts_state<state_t>::make_dispatch
                 (omcmc_state,
                  [&](auto& s)
                  {
                      auto ret_ = mcmc_sweep(*s, rng);
                      ret = tuple_apply([&](auto&... args){ return python::make_tuple(args...); }, ret_);
                  });
         });
    return ret;
}

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    def("potts_mcmc_sweep", &potts_mcmc_sweep);
});
