#include <vector>
#include <array>
#include <map>
#include <cassert>
#include <tuple>
#include <queue>
#include <iostream>
#include <set>
#include <unordered_set>
#include <tuple>

#include <partition.hxx>


/**
 * This class implements an undirected edge weighted graph data structure.
 * The number of nodes n is fixed. Edges can be inserted and removed in time log(n).
 */
template<typename VALUE_TYPE>
class DynamicGraph
{
public:
    /**
     * Initialize the data structure by specifying the number of nodes n
     */
    DynamicGraph(size_t n) :
        // create an empty adjacency vector of length n
        adjacency_(n)
    {}

    /**
     * check if the edge {a, b} exists
     */
    bool edgeExists(size_t a, size_t b) const
    {
        return adjacency_[a].count(b);
    }

    /**
     * Get a constant reference to the adjacent vertices of vertex v
     */
    std::map<size_t, VALUE_TYPE> const& getAdjacentVertices(size_t v) const
    {
        return adjacency_[v];
    }

    /**
     * Get the weight of the edge {a, b}
     */
    VALUE_TYPE getEdgeWeight(size_t a, size_t b) const
    {
        return adjacency_[a].at(b);
    }

    /**
     * Remove all edges incident to the vertex v
     */
    void removeVertex(size_t v)
    {
        // for each vertex p that is incident to v, remove v from the adjacency of p
        for (auto& p : adjacency_[v])
            adjacency_[p.first].erase(v);
        // clear the adjacency of v
        adjacency_[v].clear();
    }

    /**
     * Increase the weight of the edge {a, b} by w.
     * In particular this function can be used to insert a new edge.
     */
    void updateEdgeWeight(size_t a, size_t b, VALUE_TYPE w)
    {
        adjacency_[a][b] += w;
        adjacency_[b][a] += w;
    }

    void print() const {
        std::cout << "DynamicGraph mit " << adjacency_.size() << " Knoten:\n";
        for (size_t u = 0; u < adjacency_.size(); ++u) {
            std::cout << "  " << u << ":";
            for (auto const& [v, w] : adjacency_[u]) {
                std::cout << " (" << v << ", " << w << ")";
            }
            std::cout << "\n";
        }
        std::cout << std::flush;
    }


private:
    // data structure that contains one map for each vertex v and maps from
    // the neighbors w of v to the weight of the edge {v, w}. This data
    // structure is kept symmetric as it models undirected graph, i.e.,
    // at all times it holds that adjacency[v][w] = adjacency_[w][v]
    std::vector<std::map<size_t, VALUE_TYPE>> adjacency_;
};


/**
 * This struct implements an edge of a graph consisting of its two end vertices
 * a and b (where by convention a is the vertex with the smaller index) and a weight w.
 * This struct also implements a comparison operator for comparing two edges based on their weight.
 */
template<typename VALUE_TYPE>
struct Edge
{
    /**
     * Initialize and edge {a, b} with weight w
     */
    Edge(size_t _a, size_t _b, VALUE_TYPE _w)
    {
        if (_a > _b)
            std::swap(_a, _b);

        a = _a;
        b = _b;
        w = _w;
    }

    size_t a;
    size_t b;
    VALUE_TYPE w;

    /**
     * Compare this edge to another edge based on their weights
     */
    bool operator <(Edge const& other) const
    {
        return w < other.w;
    }
};



template<typename VALUE_TYPE>
std::pair<VALUE_TYPE, std::vector<size_t>>
greedy_joining(size_t n, std::vector<std::array<size_t, 2>> edges, std::vector<VALUE_TYPE> edge_values)
{
    assert (edges.size() == edge_values.size());

    DynamicGraph<VALUE_TYPE> graph(n);
    std::priority_queue<Edge<VALUE_TYPE>> queue;
    VALUE_TYPE value_of_cut = 0;

    // initialize the graph by inserting all edges with their respective values
    for (size_t i = 0; i < edges.size(); ++i)
    {
        size_t a = edges[i][0];
        size_t b = edges[i][1];
        VALUE_TYPE w = edge_values[i];
        graph.updateEdgeWeight(a, b, w);
        value_of_cut += w;  // initially all edges are cut
        // If the edge has positive weight, add it to the queue for possibly joining this edge
        if (w > 0)
        {
            queue.push({a, b, w});
        }
    }

    // initialize the partition where originally all vertices are in their own cluster
    andres::Partition<size_t> partition(n);

    // main loop of the greedy joining algorithm
    while (!queue.empty())
    {
        // get the top edge from the queue, i.e. the edge with the most negative value
        auto edge = queue.top();
        queue.pop();

        // check if this edge is outdated (i.e. it is no longer part of the graph or has different costs)
        if (!graph.edgeExists(edge.a, edge.b) || edge.w != graph.getEdgeWeight(edge.a, edge.b))
            continue;

        // print the current number of clusters
        //std::cout << "\rNumber of clusters: " << partition.numberOfSets() << "    " << std::flush;

        // select one of the two vertices a and b that should be kept and the other vertex is
        // merged into the vertex that is kept. Do this such that the vertex with the larger
        // adjacency is the vertex that is kept.
        auto stable_vertex = edge.a;
        auto merge_vertex = edge.b;
        if (graph.getAdjacentVertices(stable_vertex).size() < graph.getAdjacentVertices(merge_vertex).size())
            std::swap(stable_vertex, merge_vertex);

        // merge the clusters associated with the two vertices in the partition
        partition.merge(stable_vertex, merge_vertex);

        // update the graph by adding and edge from the stable_vertex to all neighbors of
        // the merge_vertex and updating the value of such an edge if it already exists
        for (auto& neighbor : graph.getAdjacentVertices(merge_vertex))
        {
            if (neighbor.first == stable_vertex)
                continue;

            graph.updateEdgeWeight(stable_vertex, neighbor.first, neighbor.second);

            // if the value of the new weight is positive, add it to the queue as a
            // candidate for joining in a future iteration
            VALUE_TYPE new_weight = graph.getEdgeWeight(stable_vertex, neighbor.first);
            if (new_weight > 0)
                queue.push({stable_vertex, neighbor.first, new_weight});
        }
        // remove all edges incident to the merge vertex from the graph
        graph.removeVertex(merge_vertex);
        value_of_cut -= edge.w;
    }
    //std::cout << "\r                                      \r" << std::flush;


    // return a node labeling and the value of the computed solution
    std::vector<size_t> labeling(n);
    partition.elementLabeling(labeling.begin());
    return {value_of_cut, labeling};
}


template<typename VALUE_TYPE>
std::pair<VALUE_TYPE, std::vector<size_t>>
greedy_joining_spanning_forest(size_t n, std::vector<std::array<size_t, 2>> edges, std::vector<VALUE_TYPE> edge_values, double tau){
    
    assert (edges.size() == edge_values.size());

    DynamicGraph<VALUE_TYPE> graph(n);
    andres::Partition<size_t> partition(n);
    VALUE_TYPE value_of_cut = 0;
    int iteration = 0;
    size_t frac = static_cast<size_t>(n * tau);
    if (frac == 0) {
        frac = 1;
    }

    
    for (size_t i = 0; i < edges.size(); ++i)
    {
        size_t a = edges[i][0];
        size_t b = edges[i][1];
        VALUE_TYPE w = edge_values[i];
        graph.updateEdgeWeight(a, b, w);
        value_of_cut += w;  // initially all edges are cut
        
    }
        

    while(1){
        //std::priority_queue<Edge<VALUE_TYPE>> queue;
        std::vector<std::unordered_set<size_t>> mutexes(n);
        std::vector<std::tuple<VALUE_TYPE, size_t, size_t>> positive_edges;
        for (size_t u = 0; u < n; ++u) {
            
            for (auto const& [v, w] : graph.getAdjacentVertices(u)) {
                if (u >= v) 
                    continue;       
        
                if (w < 0) {
                    mutexes[u].insert(v);
                    mutexes[v].insert(u);
                }
                else if (w > 0) {
                    positive_edges.emplace_back(w, u, v);
                }
            }
        }
    /*std::cout << "Iteration " << iteration++ 
          << ": positive_edges = " << positive_edges.size() 
          << ", value_of_cut = " << value_of_cut << "\n";
    print_edges(positive_edges);*/
        
        
        if(positive_edges.empty()){
            std::vector<size_t> labeling(n);
            partition.elementLabeling(labeling.begin());
            return {value_of_cut, labeling};
        }

        std::stable_sort(positive_edges.begin(), positive_edges.end(),
        [](const auto& a, const auto& b) {
            if (std::get<0>(a) != std::get<0>(b))
                return std::get<0>(a) > std::get<0>(b); // Gewicht absteigend
            else if (std::get<1>(a) != std::get<1>(b))
                return std::get<1>(a) < std::get<1>(b); // u aufsteigend
            else
                return std::get<2>(a) < std::get<2>(b); // v aufsteigend
        });
        //print_edges(positive_edges);

        std::vector<std::tuple<size_t, size_t>> mst = kruskal(positive_edges, mutexes, frac);

        /*if(frac < mst.size()){
            mst.resize(frac);
        }*/

        //std::cout << "mst size: " << mst.size() << "\n";
        //printVector(mst);
        //graph.print();
        
        for (auto [u,v] : mst) {
            
            size_t ru = partition.find(u), rv = partition.find(v);
            if (ru == rv){
              continue;     
            }else if(!graph.edgeExists(ru,rv)){
                std::cout << "break edge doesnt exist: " << ru << " - " << rv << "\n"; 
                continue;
            }
              
            auto w = graph.getEdgeWeight(ru,rv);
          
          
            
            size_t kept = (graph.getAdjacentVertices(ru).size() >=
                           graph.getAdjacentVertices(rv).size() ? ru : rv);
            size_t removed = (kept == ru ? rv : ru);
            
           
            partition.merge(kept, removed);
            
            size_t new_rep = partition.find(kept);
            size_t merge_vertex = (removed != new_rep ? removed : kept);
            //std::cout << "for edge: (" << u << ", " << v << ") merged: ( " << kept << ", " << removed << ")\n";
          
            for (auto const& [nbr, w2] : graph.getAdjacentVertices(merge_vertex)) {
                size_t nbr_rep = partition.find(nbr); 
              if (nbr_rep == new_rep) continue;
              graph.updateEdgeWeight(new_rep, nbr_rep, w2);
            }
            
            graph.removeVertex(merge_vertex);
          
           
            value_of_cut -= w;
            //graph.print();
          }

    

    }

}




template <typename VALUE_TYPE>
std::vector<std::tuple<size_t, size_t>> kruskal(std::vector<std::tuple<VALUE_TYPE, size_t, size_t>>& edges, std::vector<std::unordered_set<size_t>>& mutexes, size_t frac){
    andres::Partition<size_t> components(mutexes.size());
    andres::Partition<size_t> conflicts(mutexes.size());
    std::vector<std::tuple<size_t, size_t>> mst;
    /*
    for (const auto& edge : edges) {
        int u, v, w;
        std::tie(u, v, w) = edge;  // Entpacken des Tupels
        std::cout << "Edge: (" << u << ", " << v << ", " << w << ")" << std::endl;
    }

    for (size_t i = 0; i < mutexes.size(); ++i) {
        std::cout << "Mutexes[" << i << "]: {";
        for (const auto& item : mutexes[i]) {
            std::cout << item;
            if (&item != &(*mutexes[i].rbegin())) {  // Prüfen, ob es das letzte Element ist
                std::cout << ", ";
            }
        }
        std::cout << "}" << std::endl;
    }*/
    size_t counter = 0;


    for(size_t i = 0; i < edges.size(); i++){
        size_t u = std::get<1>(edges[i]);
        size_t v = std::get<2>(edges[i]);

        size_t root_u = components.find(u);
        size_t root_v = components.find(v);

        if(root_u != root_v){
            //std::cout << "condition true" << std::endl;
            components.merge(root_u, root_v);
            size_t ru = conflicts.find(u);
            size_t rv = conflicts.find(v);
            if(mutexes[ru].count(rv) == 0){
                
                mst.push_back(std::make_tuple(u, v));
                counter++;
                if(counter >= frac) break;
                conflicts.merge(ru, rv);
                size_t root_uv = conflicts.find(ru);

                for(size_t r : {ru, rv}){
                    
                    if(r == root_uv){
                        continue;
                    }
                    mutexes[root_uv].insert(mutexes[r].begin(), mutexes[r].end());
                    
                    for (const size_t & m : mutexes[r])
                    {
                        mutexes[m].erase(r);
                        mutexes[m].insert(root_uv);
                    }  
                    mutexes[r].clear();
                }
            }
        }
    }

    return mst;
}

void printVector(const std::vector<std::tuple<size_t, size_t>>& vec) {
    std::cout << "mst (size: "<< vec.size() <<"): [";
    for (const auto& [a, b] : vec) {
        std::cout << "(" << a << ", " << b << "), ";
    }
    std::cout << "] \n";
}
template <typename VALUE_TYPE>
void print_edges(const std::vector<std::tuple<VALUE_TYPE, size_t, size_t>>& vec) {
    std::cout << "positive edges (size: "<< vec.size() <<"): [";
    for (const auto& [a, b, c] : vec) {
        std::cout << "(" << a << ", " << b << ", " << c <<")";
    }
    std::cout << "]\n";
}