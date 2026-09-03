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

#include "graph_filtering.hh"
#include "graph.hh"
#include "graph_python_interface.hh"
#include "demangle.hh"

#include <boost/python.hpp>

using namespace std;
using namespace boost;
using namespace graph_tool;

// this will register the property maps types and all its possible access
// functions to python
struct export_vertex_property_map
{
    template <class PropertyMap>
    void operator()(PropertyMap) const
    {
        typedef PythonPropertyMap<PropertyMap> pmap_t;

        string type_name;
        if constexpr (hana::find(value_types, hana::type<typename pmap_t::value_type>()) !=
                      hana::just(hana::type<typename pmap_t::value_type>()))
            type_name =
                get_type_name<typename pmap_t::value_type>();
        else
            type_name =
                type_names[hana::index_if(value_types,
                                          hana::equal.to(hana::type<typename pmap_t::value_type>())).value().value];

        string class_name = "VertexPropertyMap<" + type_name + ">";

        typedef typename boost::mpl::if_<
            typename return_reference<typename pmap_t::value_type>::type,
            boost::python::return_internal_reference<>,
            boost::python::return_value_policy<boost::python::return_by_value> >
            ::type return_policy;

        boost::python::class_<pmap_t> pclass(class_name.c_str(),
                                             boost::python::no_init);
        pclass.def("__hash__", &pmap_t::get_hash)
            .def("value_type", &pmap_t::get_type)
            .def("get_map", &pmap_t::get_map)
            .def("get_dynamic_map", &pmap_t::get_dynamic_map)
            .def("get_array", &pmap_t::get_array)
            .def("is_writable", &pmap_t::is_writable)
            .def("reserve", &pmap_t::reserve)
            .def("resize", &pmap_t::resize)
            .def("shrink_to_fit", &pmap_t::shrink_to_fit)
            .def("swap", &pmap_t::swap)
            .def("data_ptr", &pmap_t::data_ptr);

        auto const_graph_views
            = hana::transform(graph_tool::all_graph_views,
                              [](auto t){ return hana::type<std::add_const_t<typename decltype(t)::type>>(); });
        auto all_gviews
            = hana::transform(const_graph_views,
                              [](auto t){ return hana::type<std::add_pointer_t<typename decltype(t)::type>>(); });
        auto all_const_graph_views
            = hana::transform(const_graph_views,
                              [](auto t){ return hana::type<std::add_pointer_t<typename decltype(t)::type>>(); });
        auto graph_views
            = hana::concat(all_gviews, all_const_graph_views);

        hana::for_each(graph_views,
                       [&](auto graph)
                       {
                           return dispatch_access<PropertyMap>()
                               (typename decltype(+graph)::type(), pclass,
                                return_policy());
                       });
    }

    template <class PropertyMap>
    struct dispatch_access
    {
        typedef PythonPropertyMap<PropertyMap> pmap_t;

        template <class Graph, class PClass, class ReturnPolicy>
        void operator()(Graph*, PClass& pclass, ReturnPolicy return_policy) const
        {
            pclass
                .def("__getitem__", &pmap_t::get_value_int,
                     return_policy)
                .def("__setitem__", &pmap_t::set_value_int);
        }
    };
};

struct export_edge_property_map
{
    template <class PropertyMap>
    struct dispatch_access
    {
        typedef PythonPropertyMap<PropertyMap> pmap_t;
        template <class Graph, class PClass, class ReturnPolicy>
        void operator()(Graph*, PClass& pclass, ReturnPolicy return_policy) const
        {
            pclass
                .def("__getitem__",
                     &pmap_t::template get_value<PythonEdge<Graph> >,
                     return_policy)
                .def("__setitem__",
                     &pmap_t::template set_value<PythonEdge<Graph> >);
        }
    };

    template <class PropertyMap>
    void operator()(PropertyMap) const
    {
        typedef PythonPropertyMap<PropertyMap> pmap_t;

        string type_name;
        if constexpr (hana::find(value_types, hana::type<typename pmap_t::value_type>()) !=
                      hana::just(hana::type<typename pmap_t::value_type>()))
            type_name =
                get_type_name<typename pmap_t::value_type>();
        else
            type_name =
                type_names[hana::index_if(value_types,
                                          hana::equal.to(hana::type<typename pmap_t::value_type>())).value().value];

        string class_name = "EdgePropertyMap<" + type_name + ">";

        typedef typename boost::mpl::if_<
            typename return_reference<typename pmap_t::value_type>::type,
            boost::python::return_internal_reference<>,
            boost::python::return_value_policy<boost::python::return_by_value> >
            ::type return_policy;

        boost::python::class_<pmap_t> pclass(class_name.c_str(),
                                             boost::python::no_init);
        pclass.def("__hash__", &pmap_t::get_hash)
            .def("value_type", &pmap_t::get_type)
            .def("get_map", &pmap_t::get_map)
            .def("get_dynamic_map", &pmap_t::get_dynamic_map)
            .def("get_array", &pmap_t::get_array)
            .def("is_writable", &pmap_t::is_writable)
            .def("reserve", &pmap_t::reserve)
            .def("resize", &pmap_t::resize)
            .def("shrink_to_fit", &pmap_t::shrink_to_fit)
            .def("swap", &pmap_t::swap)
            .def("data_ptr", &pmap_t::data_ptr);

        auto const_graph_views
            = hana::transform(graph_tool::all_graph_views,
                              [](auto t){ return hana::type<std::add_const_t<typename decltype(t)::type>>(); });
        auto all_gviews
            = hana::transform(graph_tool::all_graph_views,
                              [](auto t){ return hana::type<std::add_pointer_t<typename decltype(t)::type>>(); });
        auto all_const_graph_views
            = hana::transform(const_graph_views,
                              [](auto t){ return hana::type<std::add_pointer_t<typename decltype(t)::type>>(); });
        auto graph_views
            = hana::concat(all_gviews, all_const_graph_views);

        hana::for_each(graph_views,
                       [&](auto graph)
                       {
                           return dispatch_access<PropertyMap>()
                               (typename decltype(+graph)::type(), pclass,
                                return_policy());
                       });
    }
};

struct export_graph_property_map
{
    template <class PropertyMap>
    void operator()(PropertyMap) const
    {
        typedef PythonPropertyMap<PropertyMap> pmap_t;

        string type_name =
            type_names[hana::index_if(value_types,
                                      hana::equal.to(hana::type<typename pmap_t::value_type>())).value().value];

        string class_name = "GraphPropertyMap<" + type_name + ">";

        typedef typename std::conditional<
            return_reference<typename pmap_t::value_type>::type::value,
            boost::python::return_internal_reference<>,
            boost::python::return_value_policy<boost::python::return_by_value> >
            ::type return_policy;

        boost::python::class_<pmap_t> pclass(class_name.c_str(),
                                             boost::python::no_init);
        pclass.def("__hash__", &pmap_t::get_hash)
            .def("value_type", &pmap_t::get_type)
            .def("__getitem__", &pmap_t::template get_value<GraphInterface>,
                 return_policy())
            .def("__setitem__", &pmap_t::template set_value<GraphInterface>)
            .def("get_map", &pmap_t::get_map)
            .def("get_dynamic_map", &pmap_t::get_dynamic_map)
            .def("get_array", &pmap_t::get_array)
            .def("is_writable", &pmap_t::is_writable)
            .def("reserve", &pmap_t::reserve)
            .def("resize", &pmap_t::resize)
            .def("shrink_to_fit", &pmap_t::shrink_to_fit);
    }
};

void export_python_properties()
{
    auto graph_properties =
        hana::transform(value_types, [](auto t){return hana::type<property_map_t<typename decltype(t)::type,
                                                                                 ConstantPropertyMap<size_t,graph_property_tag>>>(); });
    hana::for_each(vertex_properties, [](auto t){export_vertex_property_map()(typename decltype(t)::type());});
    hana::for_each(edge_properties, [](auto t){export_edge_property_map()(typename decltype(t)::type());});
    hana::for_each(graph_properties, [](auto t){export_graph_property_map()(typename decltype(t)::type());});
}
