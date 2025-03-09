/*
 * Copyright (C) 2025  Jimmy Aguilar Mena
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	 See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.	 If not, see <http://www.gnu.org/licenses/>.
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
#include <algorithm>
#include <condition_variable>

#include <cassert>

namespace my {

	/**
	   The scheduler class which is actually a mutex protected queue.

	   This threadpool works with work-stealing, so every task is
	   enqueued and workers request new work when needed.
	 **/
	template<typename T>
	class scheduler_t {
		std::mutex _mutex;
		std::deque<T> _queue;
		std::condition_variable _cv;

		bool shutdown = {false};

	public:
		T get()
		{
			std::unique_lock lk(_mutex);
			while(_queue.empty() && !shutdown) {
				_cv.wait(lk);
			}

			// The notification was due to shutdown.
			if (_queue.empty() && shutdown) {
				return T();
			}

			T ret = std::move(_queue.front());
			_queue.pop_front();

			return std::move(ret);
		}

		void notify_shutdown()
		{
			{
				std::lock_guard lk(_mutex);
				shutdown = true;
			}
			_cv.notify_all();
		}

		void push(T &&in)
		{
			bool was_empty = false;
			{
				std::lock_guard lk(_mutex);
				was_empty = _queue.empty();
				_queue.push_back(std::forward<T>(in));
			}
			if (was_empty)
				_cv.notify_one();
		}

		void push(std::vector<T> &input)
		{
			bool was_empty = false;
			{
				std::lock_guard<std::mutex> lk(_mutex);
				was_empty = _queue.empty();
				_queue.insert(
					_queue.end(),
					std::make_move_iterator(input.begin()),
					std::make_move_iterator(input.end())
				);
			}
			if (was_empty)
				_cv.notify_all();
		}
	};

	/**
	   Thread pool class.

	   This is a basic thread pool class implemented using only basic
	   C++-20 features.	 While the current compilers actually support
	   parallel execution policies, some of them have issues
	   supporting parallel execution ones. This is a sort of poor guy
	   parallel execution implementation just for demonstration
	   purposes.
	**/
	class threadpool_t {

	private:

		class task_t {

			std::function<void(void)> _function;

		public:

			task_t(task_t&) = delete;
			task_t() = default;

			template<typename F, typename ...Params>
			task_t(F&& func, Params&& ...params)
				: _function(std::bind(std::forward<F>(func), std::forward<Params>(params)...))
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

		class worker_t {
		public:
			enum class status_t {
				finished = 0,
				running,
				sleep
			};

			worker_t() = delete;
			worker_t(const worker_t &other) = delete;  // because of the tread
			worker_t(const worker_t &&other) = delete;	// because of the atomic

			worker_t(size_t id, threadpool_t &pool)
				: _id(id),
					_parentPool(pool),
					_thread(&worker_t::workerFunction, this)
			{
			}

			~worker_t()
			{
				assert(_status.load() == status_t::finished);

				if (_thread.joinable())
					_thread.join();
			}

			/** Get the worker unique id */
			size_t get_id() const { return _id; };

		private:
			std::atomic<status_t> _status {status_t::sleep};
			const size_t _id;
			threadpool_t &_parentPool;
			std::thread _thread;
			friend class threadpool_t;

			/**
			   Spin function executed by the workers.

			   This function will run contiguously until the thread is notified
			   to die. The function also performs some sleep and wake_up when there
			   is not work to do.
			*/
			void workerFunction()
			{
				_thisThreadWorker = this;

				for (std::unique_ptr<task_t> task = _parentPool.getTask(this);
					 task;
					 task = _parentPool.getTask(this)
				) {
					task->evaluate();
					// When the counter is zero notify in case
					// some thread is waiting.
					if (_parentPool._taskCounter.fetch_sub(1) == 1)
						_parentPool._twCondVar.notify_all();
				}
			}
		};

		/** Get a task to execute; intended to be called from the workers only. */
		std::unique_ptr<task_t> getTask([[maybe_unused]] const worker_t * worker)
		{
			return _scheduler.get();
		}

		std::vector<std::unique_ptr<worker_t>> _pool;
		scheduler_t<std::unique_ptr<task_t>> _scheduler;
		std::atomic<size_t> _taskCounter {0};
		static thread_local worker_t *_thisThreadWorker;

		// For taskwait
		std::mutex _twMutex;
		std::condition_variable _twCondVar;

		template<class ForwardIt1, class ForwardIt2, class UnaryOp >
		friend ForwardIt2 transform(
			threadpool_t& pool,
			ForwardIt1 first1, ForwardIt1 last1,
			ForwardIt2 d_first, UnaryOp unary_op
		);

	public:

		/** Thread pool constructor.
			@param size Initial size of the pool.
		*/
		threadpool_t(size_t size = std::thread::hardware_concurrency())
			: _pool(size)
		{
			for (size_t i = 0; i < size; ++i)
				_pool[i] = std::make_unique<worker_t>(i, *this);
		}

		~threadpool_t()
		{
			_scheduler.notify_shutdown();
			this->taskWait();
		}

		/** Return the number of threads in the pool. */
		size_t size() const
		{
			return _pool.size();
		}

		static size_t getWorkerId()
		{
			if (_thisThreadWorker)
				return _thisThreadWorker->_id;
			return -1;
		}


		/** Push/create a new task.
			If the task que was empty, then this function wakes up the other workers.
			@param func Function to execute
			@param params Function arguments
		*/
		template<typename F, typename ...Params>
		void pushTask(F&& func, Params&& ...params)
		{
			_scheduler.push(
				std::make_unique<task_t>(
					std::forward<F>(func),
					std::forward<Params>(params)...
				)
			);
			++_taskCounter;
		}

		/** Push/create a new task.
			If the task que was empty, then this function wakes up the other workers.
		*/
		void pushTasks(std::vector<std::unique_ptr<task_t>> &&tasks)
		{
			_scheduler.push(tasks);
			_taskCounter += tasks.size();
		}

		/** Block this thread until all the submitted tasks are executed. */
		void taskWait()
		{
			if (_taskCounter.load() == 0)
				return;

			// this implementation is save cause the _taskCounter is modified
			// outside the lock, but it is atomic AND what it notifies is precisely
			// that this is the last thread that attempts to modify this var
			// (because there are not more tasks in the queue.)
			std::unique_lock lk(_twMutex);
			while (_taskCounter.load() > 0)
				_twCondVar.wait(lk);
		}
	};

	template<class ForwardIt1, class ForwardIt2, class UnaryOp >
	ForwardIt2 transform(
		threadpool_t& pool,
		ForwardIt1 first1, ForwardIt1 last1,
		ForwardIt2 d_first, UnaryOp unary_op
	) {
		const size_t nThreads = pool.size();

		// When policy is 0 use the static policy (1 chunk/worker)
		std::vector<size_t> ranges;
		ranges = computeRanges(std::distance(first1, last1), nThreads);

		const size_t nEntries = ranges.size() - 1;
		std::vector<std::unique_ptr<threadpool_t::task_t>> tasks(nEntries);

		ForwardIt1 it1 = first1;
		ForwardIt2 it2 = d_first;

		for (size_t i = 0; i < nEntries; ++i) {

			size_t step = ranges[i + 1] - ranges[i];
			ForwardIt1 end = it1 + step;

			tasks[i] = std::make_unique<threadpool_t::task_t>(
				std::transform<ForwardIt1, ForwardIt2, UnaryOp>,
				it1, end,
				it2,
				unary_op
			);
			it1 = end;
			it2 += step;
		}

		pool.pushTasks(std::move(tasks));
		return it2;
	}

	// Initialize static member of class Box
	thread_local threadpool_t::worker_t *threadpool_t::_thisThreadWorker = nullptr;
} // my namespace


