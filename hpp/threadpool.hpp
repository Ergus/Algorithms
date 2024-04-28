/*
 * Copyright (C) 2024  Jimmy Aguilar Mena
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <memory>
#include <thread>
#include <atomic>
#include <utility>
#include <vector>
#include <functional>
#include <mutex>
#include <deque>
#include <utility>

template<typename T>
class readyQueue_t {
	std::mutex _mutex;
	std::deque<T> _queue;

public:
	T get()
	{
		std::lock_guard<std::mutex> lk(_mutex);
		if (_queue.empty())
			return T();

		T ret = std::move(_queue.front());
		_queue.pop_front();

		return std::move(ret);
	}

	void push(T &&in)
	{
		std::lock_guard<std::mutex> lk(_mutex);
		_queue.push_back(std::forward<T>(in));
	}
};

class threadpool_t {

public:

	class task_t {

		std::function<void(void)> _function;

	public:

		task_t(task_t&) = delete;
		task_t() = default;

		task_t(std::function<void(void)> func)
		{
			_function = func;
		}

		template<typename ...Params>
		task_t(std::function<Params...> func, Params& ...params)
			: task_t(std::bind(func, std::forward<Params>(params)...))
		{
		}

		void evaluate() const
		{
			_function();
		}

		operator bool() const
		{
			return static_cast<bool>(_function);
		}
	};

private:

	class worker_t {
	public:
		enum class status_t {
			finished = 0,
			running,
			sleep
		};

		worker_t(const worker_t &other) = delete;  // because of the tread
		worker_t(const worker_t &&other) = delete;  // because of the atomic

		worker_t(size_t id, threadpool_t *pool)
			: _id(id), _pool(pool),
			  _thread(&worker_t::workerFunction, this)
		{}

		void setStatus(status_t status)
		{
			_status.store(status);
			_status.notify_one();
		}

		void join()
		{
			if (_thread.joinable())
				_thread.join();
		}

	private:
		const size_t _id;
		std::atomic<status_t> _status {status_t::sleep};
		std::thread _thread;
		threadpool_t * const _pool;

		void workerFunction()
		{
			size_t voidloops = 0;
			while (true) {

				if (_status.load() == status_t::sleep)
					_status.wait(status_t::sleep);

				size_t executed = 0;
				for (std::unique_ptr<task_t> task = _pool->getTask(this);
					 task;
					 task = _pool->getTask(this)
				) {
					task->evaluate();
					++executed;
					// When the counter is zero notify in case some thread is
					// waiting.
					if (_pool->_taskCounter.fetch_sub(1) == 1)
						_pool->_taskCounter.notify_one();
				}

				if (_status.load() == status_t::finished)
					break;

				if (executed == 0)
				{
					if (++voidloops < 10)
						std::this_thread::yield();
					else {
						_status.store(status_t::sleep);
						voidloops = 0;
					}
				}
			}
		}
	};

	using funWorker_t = std::function<void(worker_t &worker)>;

	void forSome(size_t start, size_t end, funWorker_t fun)
	{
		for (size_t i = start; i < end; ++i)
			fun(*_pool[i]);
	}

	void forSome(size_t start, size_t end, funWorker_t fun) const
	{
		for (size_t i = start; i < end; ++i)
			fun(*_pool[i]);
	}

	void killThreads(size_t start, size_t end)
	{
		forSome(start, end,
		        [](worker_t &worker) {
					worker.setStatus(worker_t::status_t::finished);
				}
		);

		forSome(start, end,
		        [](worker_t &worker) {
					worker.join();
				}
		);
	}

	std::vector<std::unique_ptr<worker_t>> _pool;
	readyQueue_t<std::unique_ptr<task_t>> _readyQueue;
	std::atomic<size_t> _taskCounter;

	friend class worker_t;

public:

	threadpool_t(size_t ncores = std::thread::hardware_concurrency())
	{
		this->resize(ncores);
	}

	~threadpool_t()
	{
		this->taskWait();

		killThreads(0, _pool.size());
	}

	size_t size() const
	{
		return _pool.size();
	}

	void resize(size_t newSize)
	{
		const size_t oldSize = _pool.size();

		_pool.resize(newSize);

		for (size_t i = oldSize; i < newSize; ++i)
			_pool[i] = std::make_unique<worker_t>(i, this);
	}

	std::unique_ptr<task_t> getTask(const worker_t *worker)
	{
		return _readyQueue.get();
	}

	template<typename F, typename ...Params>
	void pushTask(F func, Params ...params)
	{
		auto taskPtr = std::make_unique<task_t>(func, std::forward<Params>(params)...);
		_readyQueue.push(std::move(taskPtr));

		// The queue was empty, so wake up every one.
		if (_taskCounter++ == 0)
			forSome(
				0, _pool.size(),
				[](worker_t &worker) {
					worker.setStatus(worker_t::status_t::running);
				}
			);
	}

	void taskWait()
	{
		_taskCounter.wait(0);
	}
};
