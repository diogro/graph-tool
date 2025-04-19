#include <boost/python.hpp>

#include "graph_potts.hh"

using namespace boost;
using namespace graph_tool;

GEN_DISPATCH(potts_state, PottsState, POTTS_STATE_params)

python::object make_potts_state(boost::python::object ostate)
{
    python::object state;
    potts_state::make_dispatch(ostate,
                                  [&](auto& s){state = python::object(s);});
    return state;
}

#define __MOD__ inference
#include "module_registry.hh"
REGISTER_MOD
([]
{
    using namespace boost::python;
    def("make_potts_state", &make_potts_state);

    potts_state::dispatch
        ([&](auto* s)
         {
             typedef typename std::remove_reference<decltype(*s)>::type state_t;
             class_<state_t, bases<>, std::shared_ptr<state_t>>
                 c(name_demangle(typeid(state_t).name()).c_str(),
                   no_init);
             c.def("virtual_move", &state_t::virtual_move)
                 .def("move_vertex", &state_t::move_vertex)
                 .def("entropy", &state_t::entropy);
         });

    class_<eargs_t>("potts_eargs");
});
