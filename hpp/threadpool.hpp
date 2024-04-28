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

#include <cassert>

template<typename T>
class scheduler_t {
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

	void push(std::vector<T> &input)
	{
		std::lock_guard<std::mutex> lk(_mutex);
		_queue.insert(
			_queue.end(),
			std::make_move_iterator(input.begin()),
			std::make_move_iterator(input.end())
		);
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

		worker_t(size_t id, threadpool_t &pool)
			: _id(id), _parentPool(pool),
			  _thread(&worker_t::workerFunction, this)
		{}

		~worker_t()
		{
			assert(_status.load() == status_t::finished);

			if (_thread.joinable())
				_thread.join();
		}

		void setStatus(status_t status)
		{
			if (_status.load() == status)
				return;

			_status.store(status);
			_status.notify_one();
		}

		const size_t get_id() const { return _id; };

	private:
		std::atomic<status_t> _status {status_t::sleep};
		const size_t _id;
		std::thread _thread;
		threadpool_t &_parentPool;

		void workerFunction()
		{
			_thisThreadWorker = this;

			size_t voidloops = 0;
			while (true) {

				if (_status.load() == status_t::sleep)
					_status.wait(status_t::sleep);

				size_t executed = 0;
				for (std::unique_ptr<task_t> task = _parentPool.getTask(this);
					 task;
					 task = _parentPool.getTask(this)
				) {
					task->evaluate();
					++executed;
					// When the counter is zero notify in case some thread is
					// waiting.
					if (_parentPool._taskCounter.fetch_sub(1) == 1)
						_parentPool._taskCounter.notify_one();
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

	void forSome(
		size_t start, size_t end, std::function<void(worker_t &worker)> fun
	) {
		for (size_t i = start; i < end; ++i)
			fun(*_pool[i]);
	}

	void forSome(
		size_t start, size_t end, std::function<void(const worker_t &worker)> fun
	) const {
		for (size_t i = start; i < end; ++i)
			fun(*_pool[i]);
	}

	/** Get a task to execute; intended to be called from the workers only. */
	std::unique_ptr<task_t> getTask(const worker_t *worker)
	{
		return _scheduler.get();
	}


	std::vector<std::unique_ptr<worker_t>> _pool;
	scheduler_t<std::unique_ptr<task_t>> _scheduler;
	std::atomic<size_t> _taskCounter {0};
	static thread_local worker_t *_thisThreadWorker;

public:

	/** Thread pool constructor.
	 @param size Initial size of the pool.
	*/
	threadpool_t(size_t size = std::thread::hardware_concurrency())
	{
		this->resize(size);
	}

	~threadpool_t()
	{
		this->resize(0);
	}

	/** Return the number of threads in the pool. */
	size_t size() const
	{
		return _pool.size();
	}

	static const worker_t &get_worker() 
	{
		return *_thisThreadWorker;
	}

	/** Resize the thread pool.
		Construct or destruct threads to resize the pool.
		@param newSize New desired size for the pool.
	 */
	void resize(size_t newSize)
	{
		const size_t oldSize = _pool.size();

		this->taskWait();

		// When newSize < oldSize
		if (newSize < oldSize)
			forSome(newSize, oldSize,
			        [](worker_t &worker) {
						worker.setStatus(worker_t::status_t::finished);
					});

		_pool.resize(newSize);

		// newSize > oldSize
		for (size_t i = oldSize; i < newSize; ++i)
			_pool[i] = std::make_unique<worker_t>(i, *this);
	}

	/** Push/create a new task.
		If the task que was empty, then this function wakes up the other workers.
		@param func Function to execute
		@param params Function arguments
	*/
	template<typename F, typename ...Params>
	void pushTask(F func, Params ...params)
	{
		auto taskPtr = std::make_unique<task_t>(func, std::forward<Params>(params)...);
		_scheduler.push(std::move(taskPtr));
		++_taskCounter;

		// The queue was empty, so wake up every one.
		forSome(
			0, _pool.size(),
			[](worker_t &worker) {
				worker.setStatus(worker_t::status_t::running);
			}
		);
	}

	/** Push/create a new task.
		If the task que was empty, then this function wakes up the other workers.
	*/
	void pushTasks(std::vector<std::unique_ptr<task_t>> &&tasks)
	{
		_scheduler.push(tasks);
		_taskCounter += tasks.size();

		// The queue was empty, so wake up every one.
		forSome(
			0, _pool.size(),
			[](worker_t &worker) {
				worker.setStatus(worker_t::status_t::running);
			}
		);
	}

	/** Block this thread until all the submitted tasks are executed. */
	void taskWait()
	{
		if (_taskCounter.load() > 0)
			_taskCounter.wait(0);
	}
};

namespace my {

	template<class ForwardIt1, class ForwardIt2, class UnaryOp >
	ForwardIt2 transform(
		threadpool_t& pool,
		ForwardIt1 first1, ForwardIt1 last1,
		ForwardIt2 d_first, UnaryOp unary_op
	) {
		const size_t nThreads = pool.size();

		std::vector<size_t> ranges
			= computeRanges(std::distance(first1, last1), nThreads);

		std::vector<std::unique_ptr<threadpool_t::task_t>> tasks(nThreads);

		ForwardIt1 it1 = first1;
		ForwardIt2 it2 = d_first;

		for (size_t i = 0; i < nThreads; ++i) {

			size_t step = ranges[i + 1] - ranges[i];

			tasks[i] = std::make_unique<threadpool_t::task_t>(
				std::transform<ForwardIt1, ForwardIt2, UnaryOp>,
				it1, it1 += step,
				it2,
				unary_op
			);

			it2 += step;
		}

		pool.pushTasks(std::move(tasks));
	}
}

// Initialize static member of class Box
thread_local threadpool_t::worker_t *threadpool_t::_thisThreadWorker = nullptr;
