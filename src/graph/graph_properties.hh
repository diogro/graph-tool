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

#ifndef GRAPH_PROPERTIES_HH
#define GRAPH_PROPERTIES_HH

#include <string>
#include <vector>
#include <unordered_map>
#include <any>

#include <boost/functional/hash.hpp>
#include <boost/python/object.hpp>
#include <boost/python/extract.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_map/property_map.hpp>

#include "fast_vector_property_map.hh"
#include "graph_exceptions.hh"
#include "graph_adjacency.hh"
#include "value_convert.hh"
#include "demangle.hh"

#include <boost/hana.hpp>
namespace hana = boost::hana;
using namespace hana::literals;

// this file provides general functions for manipulating graph properties

namespace graph_tool
{

// Metaprogramming
// ===============
//
// Metafunctions and data structures to deal with property maps

// Global Property Types
// ---------------------
// global property types. only these types are allowed in property maps
// Note: we must avoid a vector<bool> (and bools in general) since it is quite
//       broken, and use a vector<uint8_t> instead!
//       see: http://www.gotw.ca/publications/N1211.pdf

inline constexpr auto value_types =
    hana::tuple_t<uint8_t, int16_t, int32_t, int64_t, double,
                  long double, std::string, std::vector<uint8_t>,
                  std::vector<int16_t>, std::vector<int32_t>,
                  std::vector<int64_t>, std::vector<double>,
                  std::vector<long double>,
                  std::vector<std::string>,
                  boost::python::object>;


// respective type names
inline const char* type_names[] =
    {"bool", "int16_t", "int32_t", "int64_t", "double", "long double",
     "string", "vector<bool>", "vector<int16_t>", "vector<int32_t>",
     "vector<int64_t>", "vector<double>", "vector<long double>",
     "vector<string>", "python::object"};

// scalar types: types contained in value_types which are scalar
inline constexpr auto scalar_types =
    hana::filter(value_types, [](auto t){return std::is_scalar<typename decltype(t)::type>(); });

// integer_types: scalar types which are integer
inline constexpr auto integer_types =
    hana::filter(value_types, [](auto t){return std::is_integral<typename decltype(t)::type>(); });

// floating_types: scalar types which are floating point
inline constexpr auto floating_types =
    hana::filter(value_types, [](auto t){return std::is_floating_point<typename decltype(t)::type>(); });

// scalar_vector_types: vector types with floating point values
inline constexpr auto vector_types =
    hana::transform(hana::append(scalar_types, hana::type<std::string>()),
                    [](auto t){return hana::type<std::vector<typename decltype(t)::type>>(); });

// scalar_vector_types: vector types with floating point values
inline constexpr auto scalar_vector_types =
    hana::transform(scalar_types, [](auto t){return hana::type<std::vector<typename decltype(t)::type>>(); });

// integer_vector_types: vector types with floating point values
inline constexpr auto integer_vector_types =
    hana::transform(integer_types, [](auto t){return hana::type<std::vector<typename decltype(t)::type>>(); });

// floating_vector_types: vector types with floating point values
inline constexpr auto floating_vector_types =
    hana::transform(floating_types, [](auto t){return hana::type<std::vector<typename decltype(t)::type>>(); });

//
// Property Map Types
// ------------------

// metafunction to generate the correct property map type given a value type and
// an index map
template <class ValueType, class IndexMap>
struct property_map_type
{
    typedef boost::checked_vector_property_map<ValueType,IndexMap> type;
};

template <class ValType, class IndexMap>
using property_map_t = typename property_map_type<ValType, IndexMap>::type;

template <class IndexMap, class Vals>
constexpr auto get_prop_types(Vals vals)
{
    return hana::transform(vals, [](auto t){return hana::type<property_map_t<typename decltype(t)::type, IndexMap>>(); });
}

typedef boost::property_map<boost::adj_list<size_t>,
                            boost::edge_index_t>::type edge_index_map_t;

typedef boost::property_map<boost::adj_list<size_t>,
                            boost::vertex_index_t>::type vertex_index_map_t;

template <class ValType>
struct eprop_map_type
{
    typedef property_map_t<ValType, edge_index_map_t> type;
};

template <class ValType>
using eprop_map_t = typename eprop_map_type<ValType>::type;

template <class ValType>
struct vprop_map_type
{
    typedef property_map_t<ValType, vertex_index_map_t> type;
};

template <class ValType>
using vprop_map_t = typename vprop_map_type<ValType>::type;

template <class Vals>
constexpr auto get_eprop_types(Vals vals)
{
    return get_prop_types<boost::property_map<boost::adj_list<size_t>,
                                              boost::edge_index_t>::type>(vals);
}

template <class Vals>
constexpr auto get_vprop_types(Vals vals)
{
    return get_prop_types<boost::property_map<boost::adj_list<size_t>,
                                              boost::vertex_index_t>::type>(vals);
}

inline constexpr auto vertex_properties =
    hana::append(get_vprop_types(value_types), hana::type<vertex_index_map_t>());

inline constexpr auto writable_vertex_properties =
    get_vprop_types(value_types);

inline constexpr auto vertex_scalar_properties =
    hana::append(get_vprop_types(scalar_types), hana::type<vertex_index_map_t>());

inline constexpr auto writable_vertex_scalar_properties =
    get_vprop_types(scalar_types);

inline constexpr auto vertex_integer_properties =
    hana::append(get_vprop_types(integer_types), hana::type<vertex_index_map_t>());

inline constexpr auto writable_vertex_integer_properties =
    get_vprop_types(integer_types);

inline constexpr auto vertex_floating_properties =
    get_vprop_types(floating_types);

inline constexpr auto vertex_vector_properties =
    get_vprop_types(vector_types);

inline constexpr auto vertex_scalar_vector_properties =
    get_vprop_types(scalar_vector_types);

inline constexpr auto vertex_integer_vector_properties =
    get_vprop_types(integer_vector_types);

inline constexpr auto vertex_floating_vector_properties =
    get_vprop_types(floating_vector_types);

inline constexpr auto edge_properties =
    hana::append(get_eprop_types(value_types), hana::type<edge_index_map_t>());

inline constexpr auto writable_edge_properties =
    get_eprop_types(value_types);

inline constexpr auto edge_scalar_properties =
    hana::append(get_eprop_types(scalar_types), hana::type<edge_index_map_t>());

inline constexpr auto writable_edge_scalar_properties =
    get_eprop_types(scalar_types);

inline constexpr auto edge_integer_properties =
    hana::append(get_eprop_types(integer_types), hana::type<edge_index_map_t>());

inline constexpr auto writable_edge_integer_properties =
    get_eprop_types(integer_types);

inline constexpr auto edge_floating_properties =
    get_eprop_types(floating_types);

inline constexpr auto edge_vector_properties =
    get_eprop_types(vector_types);

inline constexpr auto edge_scalar_vector_properties =
    get_eprop_types(scalar_vector_types);

inline constexpr auto edge_integer_vector_properties =
    get_eprop_types(integer_vector_types);

inline constexpr auto edge_floating_vector_properties =
    get_eprop_types(floating_vector_types);

// Property map manipulation
// =========================
//
// Functions which deal with several aspects of property map manipulation

// this will return the name of a given type
template <class Type>
std::string get_type_name()
{
    if constexpr (hana::find(value_types, hana::type<Type>()) ==
                  hana::just(hana::type<Type>()))
    {
        constexpr size_t index =
            hana::index_if(value_types,
                           hana::equal.to(hana::type<Type>())).value().value;
        return type_names[index];
    }
    else
    {
        return name_demangle(typeid(Type).name());
    }
}

// this functor tests whether or not a given std::any object holds a type
// contained in a given type Sequence
template <class Sequence>
bool belongs(Sequence seq, const std::any& prop)
{
    bool found = false;
    hana::for_each(seq,
                   [&](auto t)
                   {
                       typedef typename decltype(t)::type T;
                       const T* ptr = std::any_cast<T>(&prop);
                       if (ptr != 0)
                           found = true;
                   });
    return found;
}

std::tuple<std::string, std::string, bool> get_pmap_type(const std::any& prop);

template <bool pmap, class Sequence>
std::tuple<std::vector<std::string>, std::vector<std::string>, bool> get_seq_type_names(Sequence seq)
{
    bool has_unwritable = false;
    std::vector<std::string> type_names, key_names;
    hana::for_each(seq,
                   [&](auto t)
                   {
                       typedef typename decltype(+t)::type T;
                       if constexpr (pmap)
                       {
                           type_names.push_back(get_type_name<typename boost::property_traits<T>::value_type>());
                           if constexpr (std::is_same_v<typename boost::property_traits<T>::key_type,
                                                        typename boost::property_traits<edge_index_map_t>::key_type>)
                               key_names.push_back("edge");
                           else
                               key_names.push_back("vertex");
                       }
                       else
                       {
                           type_names.push_back(get_type_name<T>());
                       }
                       if constexpr (std::is_same_v<T, vertex_index_map_t> ||
                                     std::is_same_v<T, edge_index_map_t>)
                           has_unwritable = true;
                   });
    return {key_names, type_names, has_unwritable};
}

template <class Sequence>
void check_belongs(Sequence seq, const std::any& prop, std::string name = "")
{
    if (!belongs(seq, prop))
    {
        auto [key_type, type_name, writable] = get_pmap_type(prop);
        if (!writable)
            key_type = "unwritable " + key_type;
        auto [key_types, type_names, has_unwritable] = get_seq_type_names<true>(seq);

        std::string tnames;
        for (size_t i = 0; i < type_names.size(); ++i)
            tnames += (i > 0) ? (", " + type_names[i]) : type_names[i];

        throw ValueException(key_type + " property map " +
                             (name != "" ? ("'" + name + "' ") : std::string("")) +
                             "of type '" + type_name +
                             "' should be instead of key type '" +
                             key_types.front() + "' and one of the following " +
                             std::string(has_unwritable ? "" : "writable ") +
                             "value types: " + tnames);
    }
}

template <class ToSeq>
bool pmap_is_convertible(ToSeq to_seq, const std::any& prop)
{
    bool conv = false;
    hana::for_each(hana::concat(vertex_properties, edge_properties),
                   [&](auto t)
                   {
                       typedef typename decltype(t)::type From;
                       typedef typename boost::property_traits<From>::value_type
                           from_val_t;
                       const From* ptr = std::any_cast<From>(&prop);
                       if (ptr != 0)
                       {
                           hana::for_each(to_seq,
                                          [&](auto x)
                                          {
                                              typedef typename decltype(+x)::type To;
                                              typedef typename boost::property_traits<To>::value_type
                                                  to_val_t;
                                              if constexpr (graph_tool::is_convertible_v<to_val_t, from_val_t>)
                                                  conv = true;
                                          });
                       }
                   });
    return conv;
}

template <class ToSeq>
void check_pmap_is_convertible(ToSeq to_seq, const std::any& prop, std::string name = "")
{
    if (!pmap_is_convertible(to_seq, prop))
    {
        auto [key_type, type_name, writable] = get_pmap_type(prop);
        auto [key_names, type_names, has_unwritable] = get_seq_type_names<false>(to_seq);
        std::string tnames;
        for (size_t i = 0; i < type_names.size(); ++i)
            tnames += (i > 0) ? (", " + type_names[i]) : type_names[i];
        throw ValueException(key_type + " property map " +
                             (name != "" ? ("'" + name + "' ") : std::string("")) +
                             "of type '" + type_name +
                             "' canot be converted to any of the types: " + tnames);
    }
}

//
// Convert property maps
// =====================

std::any prop_map_as(std::any amap, std::any atgt, bool copy=false);
std::any vprop_map_as_dynamic(std::any amap, std::any atgt, bool identity=true);
std::any eprop_map_as_dynamic(std::any amap, std::any atgt, bool identity=true);
std::any vprop_map_as_dvelem(std::any amap, std::any atgt);
std::any eprop_map_as_dvelem(std::any amap, std::any atgt);

//
// Extra Property Map Types
// ========================

// the following class wraps a generic property map, so it can be used as a
// property with a given Key and Value type. The keys and values are converted
// to the desired Key and Value type, which may cause a performance impact,
// since virtual functions are used. Should be used only when property map
// access time is not crucial
template <class Value, class Key>
class DynamicPropertyMapWrap
{
public:
    typedef Value value_type;
    typedef Value reference;
    typedef Key key_type;
    typedef boost::read_write_property_map_tag category;

    template <class PropertyTypes>
    DynamicPropertyMapWrap(std::any pmap, PropertyTypes pts)
    {
        hana::for_each(pts,
                       [&](auto t)
                       {
                           typedef typename decltype(t)::type map_t;
                           choose_converter<map_t>(pmap, _converter, _value_type);
                       });

        if (_converter.get() == nullptr)
            throw boost::bad_lexical_cast();
    }

    DynamicPropertyMapWrap() {}

    Value get(const Key& k) const
    {
        return (*_converter).get(k);
    }

    void put(const Key& k, const Value& val)
    {
        (*_converter).put(k, val);
    }

    const std::type_info& get_underlying_value_type() const
    {
        return *_value_type;
    }

private:
    class ValueConverter
    {
    public:
        virtual Value get(const Key& k) = 0;
        virtual void put(const Key& k, const Value& val) = 0;
        virtual ~ValueConverter() {}
    };

    template <class PropertyMap>
    class ValueConverterImp: public ValueConverter
    {
    public:
        ValueConverterImp(PropertyMap pmap): _pmap(pmap) {}
        virtual ~ValueConverterImp() {}
        typedef typename boost::property_traits<PropertyMap>::value_type val_t;
        typedef typename boost::property_traits<PropertyMap>::key_type key_t;

        virtual Value get(const Key& k)
        {
            return get_dispatch(_pmap, k,
                                std::is_convertible<typename boost::property_traits<PropertyMap>::category,
                                                    boost::readable_property_map_tag>());
        }

        virtual void put(const Key& k, const Value& val)
        {
            put_dispatch(_pmap, k, convert<val_t>(val),
                         std::is_convertible<typename boost::property_traits<PropertyMap>::category,
                                             boost::writable_property_map_tag>());
        }

        template <class PMap>
        Value get_dispatch(PMap&& pmap, const typename boost::property_traits<std::remove_reference_t<PMap>>::key_type& k,
                           std::true_type)
        {
            return convert<Value>(boost::get(pmap, k));
        }

        template <class PMap>
        Value get_dispatch(PMap&&, const typename boost::property_traits<std::remove_reference_t<PMap>>::key_type&,
                           std::false_type)
        {
            throw graph_tool::ValueException("Property map is not readable.");
        }

        template <class PMap>
        void put_dispatch(PMap&& pmap, const typename boost::property_traits<std::remove_reference_t<PMap>>::key_type& k,
                          typename boost::property_traits<std::remove_reference_t<PMap>>::value_type val,
                          std::true_type)
        {
            boost::put(pmap, k, val);
        }

        template <class PMap>
        void put_dispatch(PMap&&, const typename boost::property_traits<std::remove_reference_t<PMap>>::key_type&,
                          typename boost::property_traits<std::remove_reference_t<PMap>>::value_type,
                          std::false_type)
        {
            throw ValueException("Property map is not writable.");
        }

    private:
        PropertyMap _pmap;
    };

    template <class PropertyMap>
    void choose_converter(std::any& dmap,
                          std::shared_ptr<ValueConverter>& converter,
                          std::type_info const*& value_type)
    {
        PropertyMap* pmap = std::any_cast<PropertyMap>(&dmap);
        if (pmap != nullptr)
        {
            converter = std::make_shared<ValueConverterImp<PropertyMap>>(*pmap);
            value_type = &typeid(typename boost::property_traits<PropertyMap>::value_type);
        }
    };

    std::shared_ptr<ValueConverter> _converter;
    const std::type_info* _value_type;
};

template <class Value>
using vdprop_map_t =
    DynamicPropertyMapWrap<Value,
                           typename boost::property_traits<vertex_index_map_t>::key_type>;

template <class Value>
using edprop_map_t =
    DynamicPropertyMapWrap<Value,
                           typename boost::property_traits<edge_index_map_t>::key_type>;

template <class Value, class Key, class ConvKey>
Value get(const graph_tool::DynamicPropertyMapWrap<Value,Key>& pmap,
          ConvKey k)
{
    Key key = k;
    return pmap.get(key);
}

template <class Value, class Key, class T>
void put(graph_tool::DynamicPropertyMapWrap<Value,Key>& pmap,
         Key k, const T& val)
{
    pmap.put(k, Value(val));
}

template <class Value, class Key>
const std::type_info& get_underlying_value_type(const DynamicPropertyMapWrap<Value, Key>& m)
{
    return m.get_underlying_value_type();
}

template <class T>
const std::type_info& get_underlying_value_type(const T&)
{
    return typeid(typename boost::property_traits<T>::value_type);
}

template <class T>
struct is_dynamic_map
{
    constexpr static bool value = false;
};

template <class Value, class Key>
struct is_dynamic_map<DynamicPropertyMapWrap<Value, Key>>
{
    constexpr static bool value = true;
};

template <class T>
constexpr inline bool is_dynamic_map_v = is_dynamic_map<T>::value;

// the following is hash functor which, will hash a vertex or edge descriptor
// based on its index
template <class IndexMap>
class DescriptorHash
{
public:
    DescriptorHash() {}
    DescriptorHash(IndexMap index_map): _index_map(index_map) {}
    size_t operator()(typename IndexMap::key_type const& d) const
    {
        std::hash<typename IndexMap::value_type> hash;
        return hash(_index_map[d]);
    }
private:
    IndexMap _index_map;
};

// the following is a property map based on a hashed container, which uses the
// above hash function for vertex or edge descriptors
template <class IndexMap, class Value>
class HashedDescriptorMap
    : public boost::put_get_helper<Value&, HashedDescriptorMap<IndexMap,Value>>
{
public:
    typedef DescriptorHash<IndexMap> hashfc_t;
    typedef std::unordered_map<typename IndexMap::key_type,Value,hashfc_t>
        map_t;
    typedef boost::associative_property_map<map_t> prop_map_t;

    typedef typename boost::property_traits<prop_map_t>::value_type value_type;
    typedef typename boost::property_traits<prop_map_t>::reference reference;
    typedef typename boost::property_traits<prop_map_t>::key_type key_type;
    typedef typename boost::property_traits<prop_map_t>::category category;

    HashedDescriptorMap(IndexMap index_map)
        : _base_map(std::make_shared<map_t>(0, hashfc_t(index_map))),
        _prop_map(*_base_map) {}
    HashedDescriptorMap(){}

    reference operator[](const key_type& k) { return _prop_map[k]; }
    const reference operator[](const key_type& k) const { return _prop_map[k]; }

private:
    std::shared_ptr<map_t> _base_map;
    prop_map_t _prop_map;
};


// this wraps a container as a property map which is automatically initialized
// with a given default value
template <class Container>
class InitializedPropertyMap
    : public boost::put_get_helper<typename Container::value_type::second_type&,
                                   InitializedPropertyMap<Container>>
{
public:
    typedef typename Container::value_type::second_type value_type;
    typedef value_type& reference;
    typedef typename Container::key_type key_type;
    typedef boost::read_write_property_map_tag category;

    InitializedPropertyMap(Container& base_map, value_type def)
        : _base_map(&base_map), _default(def) {}
    InitializedPropertyMap(){}

    reference operator[](const key_type& k)
    {
        return get(k);
    }

    reference operator[](const key_type& k) const
    {
        return get(k);
    }

    reference get(const key_type& k) const
    {
        typename Container::iterator val;
        val = _base_map->find(k);
        if (val == _base_map->end())
            val = _base_map->insert({k, _default}).first;
        return val->second;
    }

private:
    Container* _base_map;
    value_type _default;
};

// the following is a property map which always returns the same value
template <class Value, class Key>
class ConstantPropertyMap
    : public boost::put_get_helper<Value, ConstantPropertyMap<Value,Key>>
{
public:
    typedef Value value_type;
    typedef value_type& reference;
    typedef Key key_type;
    typedef boost::readable_property_map_tag category;

    ConstantPropertyMap(const value_type& c): _c(c) {}
    ConstantPropertyMap(): _c() {}
    ConstantPropertyMap(const ConstantPropertyMap& o): _c(o._c) {}

    const value_type& operator[](const key_type&) const { return _c; }

    ConstantPropertyMap& operator=(const ConstantPropertyMap& other)
    {
        _c = other._c;
        return *this;
    }

private:
    value_type _c;
};

template <class Value, class Key>
void put(const ConstantPropertyMap<Value, Key>&, const Key&, const Value&) {}

template <class T>
struct is_constant_map
{
    constexpr static bool value = false;
};

template <class Value, class Key>
struct is_constant_map<ConstantPropertyMap<Value, Key>>
{
    constexpr static bool value = true;
};

template <class T>
constexpr inline bool is_constant_map_v = is_constant_map<T>::value;

// the following is a property map which always returns one
template <class Value, class Key>
class UnityPropertyMap
    : public boost::put_get_helper<Value, UnityPropertyMap<Value, Key>>
{
public:
    typedef Value value_type;
    typedef value_type reference;
    typedef Key key_type;
    typedef boost::readable_property_map_tag category;

    template <class K>
    constexpr value_type operator[](const K&) const { return value_type(1); }
};

template <class Value, class Key>
void put(const UnityPropertyMap<Value, Key>&, const Key&, const Value&) {}

template <class T>
struct is_unity_map
{
    constexpr static bool value = false;
};

template <class Value, class Key>
struct is_unity_map<UnityPropertyMap<Value, Key>>
{
    constexpr static bool value = true;
};

template <class T>
constexpr inline bool is_unity_map_v = is_unity_map<T>::value;

// this wraps an existing property map, but always converts its values to a
// given type
template <class PropertyMap, class Type>
class ConvertedPropertyMap
{
public:
    typedef Type value_type;
    typedef typename boost::property_traits<PropertyMap>::value_type orig_type;
    typedef value_type reference;
    typedef typename boost::property_traits<PropertyMap>::key_type key_type;
    typedef boost::read_write_property_map_tag category;

    ConvertedPropertyMap(PropertyMap base_map)
        : _base_map(base_map) {}
    ConvertedPropertyMap(){}

    value_type do_get(const key_type& k) const
    {
        return convert<value_type>(get(_base_map, k));
    }

    void do_put(const key_type& k, const value_type& v)
    {
        put(_base_map, k, convert<orig_type>(v));
    }
private:
    PropertyMap _base_map;
};

template <class PropertyMap, class Type>
Type get(ConvertedPropertyMap<PropertyMap,Type> pmap,
         typename ConvertedPropertyMap<PropertyMap,Type>::key_type k)
{
    return pmap.do_get(k);
}

template <class PropertyMap, class Type>
void put(ConvertedPropertyMap<PropertyMap,Type> pmap,
         typename boost::property_traits<PropertyMap>::key_type k,
         const typename ConvertedPropertyMap<PropertyMap,Type>::value_type& val)
{
    pmap.do_put(k, val);
}

template <class T>
struct is_converted_map
{
    constexpr static bool value = false;
};

template <class Map, class Type>
struct is_converted_map<ConvertedPropertyMap<Map, Type>>
{
    constexpr static bool value = true;
};

template <class T>
constexpr inline bool is_converted_map_v = is_converted_map<T>::value;


} // graph_tool namespace

#endif
