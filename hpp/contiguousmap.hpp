#include <vector>

#include "utils.h"
#include "iterator_t.hpp"

template<typename K,
	typename V,
	class Compare = std::less<K>>
class cmap {

public:
	using key_type = K;
	using mapped_type = V;
	using value_type = std::pair<K, V>;
	using size_type = size_t;
	using key_compare = Compare;
	using reference = std::pair<K, V>&;
	using const_reference = const std::pair<K, V>&;
	using iterator = iterator_t<cmap>;
	using const_iterator = iterator_t<const cmap>;
	using sub_type = reference;


	size_t get_left_most(size_t start) const
	{
		size_t i = get_left(start);
		while (i < _nodes.size() && _nodes[i] != default_node)
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
			std::numeric_limits<size_t>::max();
		return (start - 1) / 2;
	}

	int get_next(size_t i) const
	{
		// When right branch search there.
		// Get the left most in the right
		if (size_t right = get_right(i);
			right < _nodes.size()
				&& _nodes[right] != default_node) {
			return get_left_most(right);
		}

		// If no right, then the next is my parent
		// If I am in the right of my parent, then I need to iterate up
		while (true) {
			size_t parent = get_parent(i);

			if (parent == std::numeric_limits<size_t>::max()
				|| _nodes[parent].first > _nodes[i].first)
				return parent;

			i = parent;
		}
	}

	/** Access index. */
	size_type get_idx(K key) const
	{
		size_t i = 0;
		while (i < _nodes.size()
			&& _nodes[i] != default_node
			&& _nodes[i].first != key) {

			// left = 2*i + 1 , right = 2*i + 2
			i = (2 * i + 1 + (_nodes[i].first < key));
		}

		return i;
	}

	K advance(K start, int step) const
	{
		size_t it = start;
		for (int i = 0; i < step; ++i) {
			it = get_next(it);

			if (it == std::numeric_limits<K>::max())
				break;
		}

		return it;
	}

	/** Get the element with idx

        This function always returns a pair, either initialized or not.
		Returned pair needs to be initialized by the function consumer
		in case of non-const version
	 **/
	const sub_type get(size_t idx) const
	{
		if (idx > _nodes.size()
			|| _nodes[idx].first == std::numeric_limits<K>::max())
			return _end;
		return _nodes[idx];
	}

	sub_type get(size_t idx)
	{
		if (idx > _nodes.size()
			|| _nodes[idx].first == std::numeric_limits<K>::max())
			return _end;

		return _nodes[idx];
	}

	static constexpr std::pair<K,V> default_node
		= std::pair<K,V>(
			std::numeric_limits<K>::max(),
			std::numeric_limits<V>::max()
		);

public:

    cmap(int size)
	: _nodes(size, default_node)
	{
	}

	V& operator[](K key)
	{
		if (_nodes[0] == default_node) {
			_nodes[0].first = key;
			return _nodes[0].second;
		}

		size_t i = 0;
		while (_nodes[i] != default_node) {
			if (_nodes[i].first == key)
				return _nodes[i].second;

			// left = 2*i + 1 , right = 2*i + 2
			i = (2*i + 1 + (_nodes[i].first < key));
			if (i >= _nodes.size())
				_nodes.resize(2 * get_pow_2(i), default_node);
		}

		_nodes[i].first = key;
		return _nodes[i].second;
	}

	template <typename F>
	void traverse_dfs(size_t start, F fun) const
	{
		if (start >= _nodes.size()
			|| _nodes[start] == default_node) {
			return;
		}

		for (size_t i = get_left_most(start);
			 i != std::numeric_limits<size_t>::max();
			 i = get_parent(i)
		) {
			fun(i, _nodes[i]);
			traverse_dfs(get_right(i), fun);
			// Remember start is not necesarily zero.
			if (i == start)
				break;

		}
	}

	template <typename F>
	void traverse_dfs_it(size_t start, F fun) const
	{
		for (int i = get_left_most(start); i >= 0; i = get_next(i)) {
			fun(i, _nodes[i]);
		}
	}

	const std::vector<std::pair<K,V>> &get_internal_vector() const
	{
		return _nodes;
	}

	const_iterator begin() const
	{
		return const_iterator(*this, get_left_most(0));
	}

	iterator begin()
	{
		return iterator(*this, get_left_most(0));
	}

	const_iterator end() const
	{
		return const_iterator(*this, std::numeric_limits<size_t>::max());
	}

	iterator end()
	{
		return iterator(*this, std::numeric_limits<size_t>::max());
	}

private:
    std::vector<std::pair<K,V>> _nodes;
	std::pair<K,V> _end{default_node};
};
