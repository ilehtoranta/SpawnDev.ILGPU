// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU
//                    Reusable Web Worker Pool for Blazor WebAssembly
//
// File: WorkerPool.cs
//
// Creates a fixed pool of Web Workers at initialization. Each worker runs a
// universal bootstrap script that accepts work via messages, avoiding the
// overhead of creating/destroying workers per kernel dispatch.
// ---------------------------------------------------------------------------------------

using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.BlazorJS.Toolbox;

namespace SpawnDev.ILGPU
{
    /// <summary>
    /// A reusable pool of Web Workers. Workers are created once with a universal
    /// bootstrap script and accept work via PostMessage. After completing work,
    /// workers return to the pool for reuse.
    /// </summary>
    public class WorkerPool : IDisposable
    {
        private readonly List<Worker> _allWorkers = new();
        private readonly Queue<Worker> _available = new();
        private readonly object _lock = new();
        private bool _disposed;

        /// <summary>
        /// Gets the total number of workers in this pool.
        /// </summary>
        public int Size => _allWorkers.Count;

        /// <summary>
        /// Universal bootstrap script for JS-based workers (Workers backend).
        /// The worker listens for messages containing a { script, data } payload.
        /// It evaluates the script as a function body with 'data' parameter,
        /// then calls it with the message data.
        /// </summary>
        private static readonly string JSBootstrapScript = @"
self.onmessage = function(e) {
  var d = e.data;
  try {
    var fn = new Function('d', d.script);
    fn(d);
  } catch(ex) {
    self.postMessage({ done: false, error: (ex && ex.message) ? ex.message : String(ex) });
  }
};
";

        /// <summary>
        /// Universal bootstrap script for Wasm-based workers (Wasm backend).
        /// The worker listens for messages containing { script, ... } payload.
        /// The script is an async function body that receives the full message data.
        /// </summary>
        private static readonly string WasmBootstrapScript = @"
self.onmessage = async function(e) {
  var d = e.data;
  try {
    var fn = new AsyncFunction('d', d.script);
    await fn(d);
  } catch(ex) {
    self.postMessage({ done: false, error: (ex && ex.message) ? ex.message : String(ex) });
  }
};
const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
";

        /// <summary>
        /// Creates a new worker pool with the specified number of workers.
        /// </summary>
        /// <param name="size">Number of workers to create.</param>
        /// <param name="useAsync">If true, uses the async bootstrap for Wasm workers.</param>
        public WorkerPool(int size, bool useAsync = false)
        {
            var script = useAsync ? WasmBootstrapScript : JSBootstrapScript;
            var workers = QuickWorker.CreateWorkersFromJS(script, size);
            foreach (var worker in workers)
            {
                _allWorkers.Add(worker);
                _available.Enqueue(worker);
            }
        }

        /// <summary>
        /// Acquires all currently available workers (up to the pool size).
        /// The caller is responsible for returning workers via <see cref="Return"/>.
        /// </summary>
        /// <param name="count">Number of workers requested.</param>
        /// <returns>List of available workers. May be fewer than requested if pool is busy.</returns>
        public List<Worker> Acquire(int count)
        {
            var result = new List<Worker>();
            lock (_lock)
            {
                while (result.Count < count && _available.Count > 0)
                {
                    result.Add(_available.Dequeue());
                }
            }
            return result;
        }

        /// <summary>
        /// Returns a worker to the pool for reuse.
        /// </summary>
        public void Return(Worker worker)
        {
            if (_disposed) return;
            lock (_lock)
            {
                if (!_disposed)
                {
                    _available.Enqueue(worker);
                }
            }
        }

        /// <summary>
        /// Returns multiple workers to the pool.
        /// </summary>
        public void Return(IEnumerable<Worker> workers)
        {
            if (_disposed) return;
            lock (_lock)
            {
                if (!_disposed)
                {
                    foreach (var worker in workers)
                    {
                        _available.Enqueue(worker);
                    }
                }
            }
        }

        /// <summary>
        /// Ensures the pool has at least the specified number of workers.
        /// Creates additional workers if needed.
        /// </summary>
        public void EnsureSize(int requiredSize, bool useAsync = false)
        {
            lock (_lock)
            {
                if (_allWorkers.Count >= requiredSize) return;

                int toCreate = requiredSize - _allWorkers.Count;
                var script = useAsync ? WasmBootstrapScript : JSBootstrapScript;
                var newWorkers = QuickWorker.CreateWorkersFromJS(script, toCreate);
                foreach (var worker in newWorkers)
                {
                    _allWorkers.Add(worker);
                    _available.Enqueue(worker);
                }
            }
        }

        /// <summary>
        /// Disposes all workers in the pool.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            lock (_lock)
            {
                _available.Clear();
                foreach (var worker in _allWorkers)
                {
                    try
                    {
                        worker.Terminate();
                        worker.Dispose();
                    }
                    catch { }
                }
                _allWorkers.Clear();
            }
        }
    }
}
