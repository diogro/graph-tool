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

#include "graph.hh"
#include "graph_util.hh"

#define NUMPY_EXPORT
#include "numpy_bind.hh"

#include "graph_python_interface.hh"

#include "random.hh"

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include "bytesobject.h"

using namespace std;
using namespace graph_tool;
using namespace boost;
using namespace boost::python;

struct LibInfo
{
    string GetName()      const {return PACKAGE_NAME;}
    string GetAuthor()    const {return AUTHOR;}
    string GetCopyright() const {return COPYRIGHT;}
    string GetVersion()   const {return VERSION " (commit " GIT_COMMIT
                                        ", " GIT_COMMIT_DATE ")";}
    string GetLicense()   const {return "GPL version 3 or above";}
    string GetCXXFLAGS()  const {return CPPFLAGS " " MOD_CXXFLAGS " " CXXFLAGS " " LDFLAGS;}
    string GetInstallPrefix() const {return INSTALL_PREFIX;}
    string GetPythonDir() const {return PYTHON_DIR;}
    string GetGCCVersion() const
    {
        stringstream s;
        s << __GNUC__ << "." << __GNUC_MINOR__ << "." <<  __GNUC_PATCHLEVEL__;
        return s.str();
    }
};

namespace std
{
bool operator==(const std::any&, const std::any&)
{
    throw ValueException("std::any comparison not implemented.");
}
}

// exception translation
template <class Exception>
void graph_exception_translator(const Exception& e)
{
    PyObject* error;
    if constexpr (std::is_same_v<Exception, GraphException>)
        error = PyExc_RuntimeError;
    else if constexpr (std::is_same_v<Exception, IOException>)
        error = PyExc_IOError;
    else if constexpr (std::is_same_v<Exception, ValueException>)
        error = PyExc_ValueError;
    // else
    //     static_assert(false);
    PyErr_SetString(error, e.what());
}

void raise_error(const string& msg)
{
    throw GraphException(msg);
}

template <class T1, class T2>
struct pair_to_tuple
{
    static PyObject* convert(const pair<T1,T2>& p)
    {
        boost::python::tuple t = boost::python::make_tuple(p.first,p.second);
        return incref(t.ptr());
    }
};

template <class T1, class T2>
struct pair_from_tuple
{
    pair_from_tuple()
    {
        converter::registry::push_back(&convertible, &construct,
                                       boost::python::type_id<pair<T1,T2> >());
    }

    static void* convertible(PyObject* obj_ptr)
    {
        handle<> x(borrowed(obj_ptr));
        object o(x);
        if (boost::python::len(o) < 2)
            return 0;
        extract<T1> first(o[0]);
        extract<T2> second(o[1]);
        if (!first.check() || !second.check())
            return 0;
        return obj_ptr;
    }

    static void construct(PyObject* obj_ptr,
                          converter::rvalue_from_python_stage1_data* data)
    {
        handle<> x(borrowed(obj_ptr));
        object o(x);
        pair<T1,T2> value;
        value.first = extract<T1>(o[0])();
        value.second = extract<T2>(o[1])();
        void* storage =
            ( (boost::python::converter::rvalue_from_python_storage
               <pair<T1,T2> >*) data)->storage.bytes;
        new (storage) pair<T1,T2>(value);
        data->convertible = storage;
    }
};

template <class ValueType>
struct variant_from_python
{
    variant_from_python()
    {
        converter::registry::push_back
            (&convertible, &construct,
             boost::python::type_id<GraphInterface::deg_t>());
    }

    static void* convertible(PyObject* obj_ptr)
    {
        handle<> x(borrowed(obj_ptr));
        object o(x);
        extract<ValueType> str(o);
        if (!str.check())
            return 0;
        return obj_ptr;
    }

    static void construct(PyObject* obj_ptr,
                          converter::rvalue_from_python_stage1_data* data)
    {
        handle<> x(borrowed(obj_ptr));
        object o(x);
        extract<ValueType> str(o);
        ValueType value = str();
        GraphInterface::deg_t deg = value;
        void* storage =
            ((boost::python::converter::rvalue_from_python_storage
              <GraphInterface::deg_t>*) data)->storage.bytes;
        new (storage) GraphInterface::deg_t(deg);
        data->convertible = storage;
    }
};

template <class T>
struct integer_from_convertible
{
    integer_from_convertible()
    {
        converter::registry::push_back(&convertible, &construct,
                                       boost::python::type_id<T>());
    }

    static void* convertible(PyObject* obj_ptr)
    {
        if (PyObject_HasAttrString(obj_ptr, "__int__"))
            return obj_ptr;
        return 0;
    }

    static void construct(PyObject* obj_ptr,
                          converter::rvalue_from_python_stage1_data* data)
    {
        handle<> x(borrowed(obj_ptr));
        object o(x);
        T value = extract<T>(o.attr("__int__")())();
        void* storage =
            ( (boost::python::converter::rvalue_from_python_storage<T>*) data)->storage.bytes;
        new (storage) T(value);
        data->convertible = storage;
    }
};

template <class T>
struct float_from_convertible
{
    float_from_convertible()
    {
        converter::registry::push_back(&convertible, &construct,
                                       boost::python::type_id<T>());
    }

    static void* convertible(PyObject* obj_ptr)
    {
        if (PyObject_HasAttrString(obj_ptr, "__float__"))
            return obj_ptr;
        return 0;
    }

    static void construct(PyObject* obj_ptr,
                          converter::rvalue_from_python_stage1_data* data)
    {
        handle<> x(borrowed(obj_ptr));
        object o(x);
        T value = extract<T>(o.attr("__float__")())();
        void* storage =
            ( (boost::python::converter::rvalue_from_python_storage<T>*) data)->storage.bytes;
        new (storage) T(value);
        data->convertible = storage;
    }
};


// persistent python object IO
namespace graph_tool
{
extern python::object object_pickler;
extern python::object object_unpickler;
}

void set_pickler(boost::python::object o)
{
    graph_tool::object_pickler = o;
}

void set_unpickler(boost::python::object o)
{
    graph_tool::object_unpickler = o;
}

boost::python::list get_property_types()
{
    boost::python::list plist;
    for (size_t i = 0; i < hana::length(value_types).value; ++i)
        plist.append(string(type_names[i]));
    return plist;
}

struct graph_type_name
{
    typedef void result_type;
    template <class Graph>
    void operator()(const Graph&, string& name) const
    {
        name = name_demangle(typeid(Graph).name());
    }
};

string get_graph_type(GraphInterface& g)
{
    string name;
    run_action<>()
        (g,
         [&](auto&& graph)
         {
             return graph_type_name()
                 (std::forward<decltype(graph)>(graph), name);
         })();
    return name;
}

size_t get_ptr(std::shared_ptr<GraphInterface::multigraph_t>& p)
{
    return size_t(p.get());
};

void ungroup_vector_property(GraphInterface& g, std::any vector_prop,
                             std::any prop, size_t pos, bool edge);
void group_vector_property(GraphInterface& g, std::any vector_prop,
                           std::any prop, size_t pos, bool edge);
void property_map_values(GraphInterface& g, std::any src_prop,
                         std::any tgt_prop, boost::python::object mapper,
                         bool edge);
void infect_vertex_property(GraphInterface& gi, std::any prop,
                            boost::python::object val);
void edge_endpoint(GraphInterface& gi, std::any prop,
                   std::any eprop, std::string endpoint);
void out_edges_op(GraphInterface& gi, std::any eprop, std::any vprop,
                  std::string op);
void mark_edges(GraphInterface& gi, std::any prop);

void perfect_ehash(GraphInterface& gi, std::any prop, std::any hprop,
                   std::any& dict);
void perfect_vhash(GraphInterface& gi, std::any prop, std::any hprop,
                   std::any& dict);
void set_vertex_property(GraphInterface& gi, std::any prop,
                         boost::python::object val);
void set_edge_property(GraphInterface& gi, std::any prop,
                       boost::python::object val);

void copy_external_edge_property(const GraphInterface& src,
                                 const GraphInterface& tgt,
                                 std::any prop_src,
                                 std::any prop_tgt);

bool compare_vertex_properties(const GraphInterface& gi,
                               std::any prop1,
                               std::any prop2);
bool compare_edge_properties(const GraphInterface& gi,
                             std::any prop1,
                             std::any prop2);

void export_python_interface();

void export_openmp();

BOOST_PYTHON_MODULE(libgraph_tool_core)
{
    python::docstring_options dopt(true, false);

    // numpy
    PyArray_ImportNumPyAPI();
    export_python_interface();

    // random numbers
    class_<rng_t>("rng_t");
    def("get_rng", get_rng, return_value_policy<reference_existing_object>());
    def("seed_rng", seed_rng);

    register_exception_translator<GraphException>
        (graph_exception_translator<GraphException>);
    register_exception_translator<IOException>
        (graph_exception_translator<IOException>);
    register_exception_translator<ValueException>
        (graph_exception_translator<ValueException>);

    def("raise_error", &raise_error);
    def("get_property_types", &get_property_types);
    class_<std::any>("any")
        .def("has_value", &std::any::has_value)
        .def("type",  &std::any::type,
             return_value_policy<reference_existing_object>());
    class_<std::type_info, boost::noncopyable>("type_info", no_init)
        .def("name", &std::type_info::name)
        .def("hash_code", &std::type_info::hash_code);
    def("name_demangle", &name_demangle);

    export_openmp();

    boost::hana::for_each(boost::hana::append(scalar_types,boost::hana::type<string>()),
                          [](auto t){export_vector_types<>()(typename decltype(+t)::type());});
    export_vector_types<>()(size_t(), "size_t");
    export_vector_types<>()(std::vector<double>(), "Vector_double");
    export_vector_types<false,false>()(std::any(), "any");
    export_vector_types<>()(std::complex<double>(), "cdouble");
    export_vector_types<>()(bool(), "bool");

    class_<GraphInterface>("GraphInterface", init<>())
        .def(init<GraphInterface,bool,boost::python::object,
                  boost::python::object, boost::python::object>())
        .def("get_num_vertices", &GraphInterface::get_num_vertices)
        .def("get_num_edges", &GraphInterface::get_num_edges)
        .def("set_directed", &GraphInterface::set_directed)
        .def("get_directed", &GraphInterface::get_directed)
        .def("set_reversed", &GraphInterface::set_reversed)
        .def("get_reversed", &GraphInterface::get_reversed)
        .def("set_keep_epos", &GraphInterface::set_keep_epos)
        .def("get_keep_epos", &GraphInterface::get_keep_epos)
        .def("set_keep_ehash", &GraphInterface::set_keep_ehash)
        .def("get_keep_ehash", &GraphInterface::get_keep_ehash)
        .def("set_vertex_filter_property",
             &GraphInterface::set_vertex_filter_property)
        .def("is_vertex_filter_active", &GraphInterface::is_vertex_filter_active)
        .def("set_edge_filter_property",
             &GraphInterface::set_edge_filter_property)
        .def("is_edge_filter_active", &GraphInterface::is_edge_filter_active)
        .def("purge_vertices",  &GraphInterface::purge_vertices)
        .def("purge_edges",  &GraphInterface::purge_edges)
        .def("shift_vertex_property",  &GraphInterface::shift_vertex_property)
        .def("move_vertex_property",  &GraphInterface::move_vertex_property)
        .def("re_index_vertex_property",  &GraphInterface::re_index_vertex_property)
        .def("write_to_file", &GraphInterface::write_to_file)
        .def("read_from_file",&GraphInterface::read_from_file)
        .def("degree_map", &GraphInterface::degree_map)
        .def("clear", &GraphInterface::clear)
        .def("clear_edges", &GraphInterface::clear_edges)
        .def("get_vertex_index", &GraphInterface::get_vertex_index)
        .def("get_edge_index", &GraphInterface::get_edge_index)
        .def("get_edge_index_range", &GraphInterface::get_edge_index_range)
        .def("re_index_edges", &GraphInterface::re_index_edges)
        .def("shrink_to_fit", &GraphInterface::shrink_to_fit)
        .def("get_graph_index", &GraphInterface::get_graph_index)
        .def("copy_vertex_property", &GraphInterface::copy_vertex_property)
        .def("copy_edge_property", &GraphInterface::copy_edge_property)
        .def("get_graph_ptr", &GraphInterface::get_graph_ptr)
        .def("get_graph_view", &GraphInterface::get_graph_view);

    class_<GraphInterface::vertex_index_map_t>("vertex_index_map", no_init);
    class_<GraphInterface::edge_index_map_t>("edge_index_map", no_init);
    class_<GraphInterface::graph_index_map_t>("graph_index_map", no_init);

    enum_<GraphInterface::degree_t>("Degree")
        .value("In", GraphInterface::IN_DEGREE)
        .value("Out", GraphInterface::OUT_DEGREE)
        .value("Total", GraphInterface::TOTAL_DEGREE);

    variant_from_python<std::any>();
    variant_from_python<GraphInterface::degree_t>();
    to_python_converter<pair<string,bool>, pair_to_tuple<string,bool> >();
    to_python_converter<pair<size_t,size_t>, pair_to_tuple<size_t,size_t> >();
    to_python_converter<pair<double,double>, pair_to_tuple<double,double> >();
    pair_from_tuple<double,double>();
    pair_from_tuple<size_t,size_t>();
    integer_from_convertible<uint8_t>();
    integer_from_convertible<int32_t>();
    integer_from_convertible<int64_t>();
    integer_from_convertible<uint32_t>();
    integer_from_convertible<uint64_t>();
    integer_from_convertible<size_t>();
    integer_from_convertible<bool>();
    float_from_convertible<float>();
    float_from_convertible<double>();
    float_from_convertible<long double>();

    class_<std::shared_ptr<GraphInterface::multigraph_t>>
        ("shared_ptr<multigraph_t>", no_init)
        .def("get", &get_ptr);

    class_<IStream>("IStream", no_init).def("read", &IStream::read);
    class_<OStream>("OStream", no_init).def("write", &OStream::write).
        def("flush", &OStream::flush);
    def("set_pickler", &set_pickler);
    def("set_unpickler", &set_unpickler);

    def("group_vector_property", &group_vector_property);
    def("ungroup_vector_property", &ungroup_vector_property);
    def("property_map_values", &property_map_values);
    def("infect_vertex_property", &infect_vertex_property);
    def("edge_endpoint", &edge_endpoint);
    def("out_edges_op", &out_edges_op);
    def("mark_edges", &mark_edges);
    def("perfect_ehash", &perfect_ehash);
    def("perfect_vhash", &perfect_vhash);
    def("set_vertex_property", &set_vertex_property);
    def("set_edge_property", &set_edge_property);
    def("copy_external_edge_property", &copy_external_edge_property);
    def("compare_vertex_properties", &compare_vertex_properties);
    def("compare_edge_properties", &compare_edge_properties);

    class_<LibInfo>("mod_info")
        .add_property("name", &LibInfo::GetName)
        .add_property("author", &LibInfo::GetAuthor)
        .add_property("copyright", &LibInfo::GetCopyright)
        .add_property("version", &LibInfo::GetVersion)
        .add_property("license", &LibInfo::GetLicense)
        .add_property("cxxflags", &LibInfo::GetCXXFLAGS)
        .add_property("install_prefix", &LibInfo::GetInstallPrefix)
        .add_property("python_dir", &LibInfo::GetPythonDir)
        .add_property("gcc_version", &LibInfo::GetGCCVersion);

    def("get_graph_type", &get_graph_type);

    def("get_null_vertex",
        +[](){ return graph_traits<GraphInterface::multigraph_t>::null_vertex();});
}
