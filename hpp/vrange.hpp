#include <iostream>

template <typename T>
class vrange {

public:
	class it {
		const vrange &_varray;
		size_t _vpos;

		void check() const
		{
			if (_varray._step == 0)
				throw std::invalid_argument("Invalid iterator.");

			if (_vpos > _varray._count)
				throw std::out_of_range("Iterator out of range.");
		}

	public:

		it(const vrange &arr, T pos)
		: _varray(arr), _vpos(pos)
		{
			check();
		}

		it& operator+=(int n)
		{
			_vpos = std::min(_vpos + n, _varray._count);
			return *this;
		}

		it& operator++()
		{
			return this->operator+=(1);
		}

		bool operator!=(const it& other) const
		{
			if (&_varray != &other._varray)
				throw std::invalid_argument("Comparing iterators in different range objects.");

			check();
			other.check();

			return _vpos != other._vpos;
		}

		T operator *() const
		{
			return _varray[_vpos];
		}
	};

private:
    T _start;
	size_t _count;
	int _step;

public:
    vrange(T start, size_t count, int step=1):
        _start(start),
		_count(count),
		_step(step)
	{}

    it begin() const { return it(*this, 0); }
    it end() const { return it(*this, _count); }

    T operator[](size_t pos) const
	{
		return _start + pos * _step;
	}

	size_t size() const
	{
		return _count;
	}
};



