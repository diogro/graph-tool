//=======================================================================
// Copyright (c) 2018 Yi Ji
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
//=======================================================================

#ifndef BOOST_GRAPH_MAXIMUM_WEIGHTED_MATCHING_BRUTE_HPP
#define BOOST_GRAPH_MAXIMUM_WEIGHTED_MATCHING_BRUTE_HPP

#include <algorithm> // for std::iter_swap
#include <boost/graph/max_cardinality_matching.hpp>

namespace boost
{
namespace alt
{
    template <typename Graph, typename WeightMap, typename MateMap, typename VertexIndexMap>
    typename property_traits<WeightMap>::value_type
    matching_weight_sum(const Graph& g, WeightMap weight, MateMap mate, VertexIndexMap vm)
    {
        typedef typename graph_traits<Graph>::vertex_iterator vertex_iterator_t;
        typedef typename graph_traits<Graph>::vertex_descriptor vertex_descriptor_t;
        typedef typename property_traits<WeightMap>::value_type edge_property_t;

        edge_property_t weight_sum = 0;
        vertex_iterator_t vi, vi_end;

        for (boost::tie(vi,vi_end) = vertices(g); vi != vi_end; ++vi)
        {
            vertex_descriptor_t v = *vi;
            if (get(mate, v) != graph_traits<Graph>::null_vertex() && get(vm, v) < get(vm, get(mate,v)))
                weight_sum += get(weight, edge(v,mate[v],g).first);
        }
        return weight_sum;
    }

    template <typename Graph, typename WeightMap, typename MateMap>
    inline typename property_traits<WeightMap>::value_type
    matching_weight_sum(const Graph& g, WeightMap weight, MateMap mate)
    {
        return matching_weight_sum(g, weight, mate, get(vertex_index,g));
    }

    // brute-force matcher searches all possible combinations of matched edges to get the maximum weighted matching
    // which can be used for testing on small graphs (within dozens vertices)
    template <typename Graph, typename WeightMap, typename MateMap, typename VertexIndexMap>
    class brute_force_matching
    {
    public:

        typedef typename graph_traits<Graph>::vertex_descriptor vertex_descriptor_t;
        typedef typename graph_traits<Graph>::vertex_iterator vertex_iterator_t;
        typedef typename std::vector<vertex_descriptor_t>::iterator vertex_vec_iter_t;
        typedef typename graph_traits<Graph>::edge_iterator edge_iterator_t;
        typedef boost::iterator_property_map<vertex_vec_iter_t, VertexIndexMap> vertex_to_vertex_map_t;

        brute_force_matching(const Graph& arg_g, WeightMap arg_weight, MateMap arg_mate, VertexIndexMap arg_vm) :
        g(arg_g),
        weight(arg_weight),
        vm(arg_vm),
        mate_vector(num_vertices(g)),
        best_mate_vector(num_vertices(g)),
        mate(mate_vector.begin(), vm),
        best_mate(best_mate_vector.begin(), vm)
        {
            vertex_iterator_t vi,vi_end;
            for (boost::tie(vi,vi_end) = vertices(g); vi != vi_end; ++vi)
                best_mate[*vi] = mate[*vi] = get(arg_mate, *vi);
        }

        template <typename PropertyMap>
        void find_matching(PropertyMap pm)
        {
            edge_iterator_t ei;
            boost::tie(ei, ei_end) = edges(g);
            select_edge(ei);

            vertex_iterator_t vi,vi_end;
            for (boost::tie(vi,vi_end) = vertices(g); vi != vi_end; ++vi)
                put(pm, *vi, best_mate[*vi]);
        }

    private:

        const Graph& g;
        WeightMap weight;
        VertexIndexMap vm;
        std::vector<vertex_descriptor_t> mate_vector, best_mate_vector;
        vertex_to_vertex_map_t mate, best_mate;
        edge_iterator_t ei_end;

        void select_edge(edge_iterator_t ei)
        {
            if (ei == ei_end)
            {
                if (alt::matching_weight_sum(g, weight, mate) > alt::matching_weight_sum(g, weight, best_mate))
                {
                    vertex_iterator_t vi, vi_end;
                    for (boost::tie(vi,vi_end) = vertices(g); vi != vi_end; ++vi)
                        best_mate[*vi] = mate[*vi];
                }
                return;
            }

            vertex_descriptor_t v, w;
            v = source(*ei, g);
            w = target(*ei, g);

            select_edge(++ei);

            if (mate[v] == graph_traits<Graph>::null_vertex() &&
                mate[w] == graph_traits<Graph>::null_vertex())
            {
                mate[v] = w;
                mate[w] = v;
                select_edge(ei);
                mate[v] = mate[w] = graph_traits<Graph>::null_vertex();
            }
        }

    };

    template <typename Graph, typename WeightMap, typename MateMap, typename VertexIndexMap>
    void brute_force_maximum_weighted_matching(const Graph& g, WeightMap weight, MateMap mate, VertexIndexMap vm)
    {
        empty_matching<Graph, MateMap>::find_matching(g, mate);
        brute_force_matching<Graph, WeightMap, MateMap, VertexIndexMap> brute_force_matcher(g, weight, mate, vm);
        brute_force_matcher.find_matching(mate);
    }

    template <typename Graph, typename WeightMap, typename MateMap>
    inline void brute_force_maximum_weighted_matching(const Graph& g, WeightMap weight, MateMap mate)
    {
        brute_force_maximum_weighted_matching(g, weight, mate, get(vertex_index, g));
    }
}
}

#endif
