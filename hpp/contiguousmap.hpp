#include <iostream>
#include <vector>
#include "utils.h"

template<typename K, typename V>
class cmap {

	size_t get_left_most(size_t start) const
	{
		size_t i = get_left(start);
		while (i < _nodes.size() && _nodes[i] != std::pair<K,V>())
			i = get_left(i);

		return get_parent(i);
	}

	static size_t get_left(size_t start)
	{
		return 2 * start + 1;
	}

	static size_t get_right(size_t start)
	{
		return 2 * start + 2;
	}

	static size_t get_parent(size_t start)
	{
		if (start == 0)
			throw std::invalid_argument("Node zero has no parent");
		return (start - 1) / 2;
	}

	int get_next(size_t i) const
	{
		// When right branch search there.
		// Get the left most in the right
		if (size_t right = get_right(i);
			right < _nodes.size()
				&& _nodes[right] != std::pair<K,V>()) {
			return get_left_most(right);
		}

		// If no right, then the next is my parent
		// If I am in the right of my parent, then I need to iterate up
		while (true) {
			if (i == 0) // Root node has no parent, so here return end
				return -1;

			size_t parent = get_parent(i);

			if (_nodes[parent].first > _nodes[i].first)
				return parent;

			i = parent;
		}
	}

public:
    cmap(int size)
	: _nodes(size)
	{
	}

	V& operator[](K key)
	{
		if (_nodes[0] == std::pair<K,V>()) {
			_nodes[0].first = key;
			return _nodes[0].second;
		}

		size_t i;
		for (i = 0; _nodes[i] != std::pair<K,V>();) {
			if (_nodes[i].first == key)
				return _nodes[i].second;

			// 2*i + 1 left, 2*i + 2 right
			i = (2*i + 1 + (_nodes[i].first < key));
			if (i >= _nodes.size())
				_nodes.resize(_nodes.size());
		}

		_nodes[i].first = key;
		return _nodes[i].second;
	}

	template <typename F>
	void traverse_dfs(size_t start, F fun) const
	{
		if (start >= _nodes.size()
			|| _nodes[start] == std::pair<K,V>())
			return;

		size_t i = get_left_most(start);

		while (true) {
			fun(i, _nodes[i]);

			traverse_dfs(get_right(i), fun);

			if (i == start)
				break;
			i = get_parent(i);
		}
	}

	template <typename F>
	void traverse_dfs_it(size_t start, F fun) const
	{
		for (int i = get_left_most(start); i >= 0; i = get_next(i)) {
			fun(i, _nodes[i]);
		}
	}

	void printvec() const
	{
		for (size_t i = 0; i < _nodes.size(); ++i)
			std::cout << i << " : "<< _nodes[i] << std::endl;
	}

private:
    std::vector<std::pair<K,V>> _nodes;
};

