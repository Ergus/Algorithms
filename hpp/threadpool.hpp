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
#include <future>
#include <functional>

#include <deque>
#include <utility>

#include <iostream>

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

	size_t push(T &&in)
	{
		std::lock_guard<std::mutex> lk(_mutex);
		_queue.push_back(std::forward<T>(in));
		return _queue.size();
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

		worker_t(threadpool_t &pool)
			: _pool(pool),
			  _status(status_t::sleep),
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
		threadpool_t &_pool;
		std::atomic<status_t> _status;
		std::thread _thread;


		void workerFunction()
		{
			std::thread::id this_id = std::this_thread::get_id();

			while (true) {

				if (_status.load() == status_t::sleep) {
					_status.wait(status_t::sleep);
				}

				for (std::unique_ptr<task_t> task = _pool.getTask();
					 task;
					 task = _pool.getTask()
				) {
					task->evaluate();
				}

				if (_status.load() == status_t::finished) {
					break;
				}
			}
			std::cout << "Exiting" << std::endl;
		}
	};

	void forAll(std::function<void(worker_t &worker)> fun)
	{
		for (std::unique_ptr<worker_t> &worker : _pool)
			fun(*worker);
	}

	void forAll(std::function<void(const worker_t &worker)> fun) const
	{
		for (const std::unique_ptr<worker_t> &worker : _pool)
			fun(*worker);
	}

	const size_t _ncores;
	std::vector<std::unique_ptr<worker_t>> _pool;
	readyQueue_t<std::unique_ptr<task_t>> _readyQueue;

public:

	std::unique_ptr<task_t> getTask()
	{
		return std::move(_readyQueue.get());
	}

	template<typename ...Params>
	void pushTask(std::function<void(Params...)> func, Params ...params)
	{
		auto taskPtr = std::make_unique<task_t>(func, std::forward<Params>(params)...);
		const size_t size = _readyQueue.push(std::move(taskPtr));

		// The queue was empty, so wake up every one.
		if (size == 1)
			forAll(
				[](worker_t &worker) {
					worker.setStatus(worker_t::status_t::running);
				}
			);
	}

	void pushTask(std::function<void()> func)
	{
		pushTask<>(func);
	}

	threadpool_t(size_t ncores = std::thread::hardware_concurrency())
		: _ncores(ncores)
	{
		for (size_t i = 0; i < _ncores; ++i)
			_pool.emplace_back(std::make_unique<worker_t>(*this));
	}

	~threadpool_t()
	{
		forAll(
			[](worker_t &worker) {
				worker.setStatus(worker_t::status_t::finished);
			}
		);

		forAll(
			[](worker_t &worker) {
				worker.join();
			}
		);
	}

};
