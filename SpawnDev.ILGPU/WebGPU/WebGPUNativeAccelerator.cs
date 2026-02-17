// ---------------------------------------------------------------------------------------
//                               SpawnDev.ILGPU.WebGPU
//                 WebGPU Compute Library for Blazor WebAssembly
//
// File: WebGPUNativeAccelerator.cs
// ---------------------------------------------------------------------------------------

using SpawnDev.BlazorJS.JSObjects;
using SpawnDev.ILGPU.WebGPU.Backend;

namespace SpawnDev.ILGPU.WebGPU
{
    /// <summary>
    /// Represents a WebGPU accelerator for GPU compute in the browser.
    /// </summary>
    public sealed class WebGPUNativeAccelerator : IDisposable
    {
        #region Static

        /// <summary>
        /// Creates a WebGPU accelerator asynchronously.
        /// </summary>
        public static async Task<WebGPUNativeAccelerator> CreateAsync(WebGPUDevice device)
        {
            var accelerator = new WebGPUNativeAccelerator(device);
            await accelerator.InitializeAsync();
            return accelerator;
        }

        /// <summary>
        /// Creates a WebGPU accelerator from an externally-provided GPUDevice.
        /// This is used when sharing a device with another library (e.g., ONNX Runtime Web)
        /// that has already created its own GPUDevice. Skips adapter probing and device
        /// creation — uses the provided device directly.
        /// </summary>
        /// <param name="externalDevice">An existing GPUDevice (e.g., from ORT's env.webgpu.device).</param>
        public static WebGPUNativeAccelerator CreateFromExternalDevice(GPUDevice externalDevice)
        {
            if (externalDevice == null)
                throw new ArgumentNullException(nameof(externalDevice));

            var accelerator = new WebGPUNativeAccelerator(externalDevice);
            return accelerator;
        }

        #endregion

        #region Instance

        private GPUDevice? _gpuDevice;
        private GPUQueue? _queue;
        private bool _isInitialized;
        private bool _disposed;

        // Shader cache: key is WGSL source, value is compiled shader
        private readonly Dictionary<string, WebGPUComputeShader> _shaderCache = new();

        /// <summary>
        /// Constructs a new WebGPU accelerator from a WebGPUDevice (adapter-based, needs InitializeAsync).
        /// </summary>
        private WebGPUNativeAccelerator(WebGPUDevice device)
        {
            Device = device ?? throw new ArgumentNullException(nameof(device));
        }

        /// <summary>
        /// Constructs a WebGPU accelerator from an existing GPUDevice (external device injection).
        /// The device is already initialized — no adapter probing or device creation needed.
        /// </summary>
        private WebGPUNativeAccelerator(GPUDevice externalDevice)
        {
            _gpuDevice = externalDevice ?? throw new ArgumentNullException(nameof(externalDevice));
            _queue = externalDevice.Queue;
            _isInitialized = true;

            // Read device features
            EnabledFeatures = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            try
            {
                using var features = externalDevice.Features;
                foreach (var featureName in new[] { "shader-f16", "subgroups", "timestamp-query" })
                {
                    if (features.Has(featureName))
                        EnabledFeatures.Add(featureName);
                }
            }
            catch { }

            // Listen for uncaptured GPU errors
            _gpuDevice.OnUncapturedError += OnGPUUncapturedError;

            Console.WriteLine("[WebGPU] Accelerator created from external GPUDevice (shared with ORT)");
        }

        /// <summary>
        /// Initializes the accelerator by requesting the GPU device with detected features.
        /// </summary>
        private async Task InitializeAsync()
        {
            if (_isInitialized)
                return;

            var adapter = Device.Adapter;
            var requestedFeatures = Device.SupportedFeatures.ToList();

            // Query the adapter's actual limits so we can request the maximum supported.
            // WebGPU defaults to 8 maxStorageBuffersPerShaderStage, but ILGPU kernels
            // use one storage buffer per parameter, so complex kernels need much more.
            int maxStorageBuffers = 10; // safe fallback
            try
            {
                using var adapterLimits = adapter.Limits;
                maxStorageBuffers = adapterLimits.MaxStorageBuffersPerShaderStage ?? 10;
            }
            catch
            {
                // Limits query failed — use fallback
            }
            Console.WriteLine($"[WebGPU] Adapter maxStorageBuffersPerShaderStage: {maxStorageBuffers}");

            // Use Dictionary for RequiredLimits to ensure reliable JS interop serialization.
            // Anonymous objects may not serialize correctly through BlazorJS's interop layer.
            var requiredLimits = new Dictionary<string, object>
            {
                ["maxStorageBuffersPerShaderStage"] = maxStorageBuffers
            };

            // Request device with detected features and the adapter's max storage buffer limit.
            try
            {
                var descriptor = new GPUDeviceDescriptor
                {
                    RequiredFeatures = requestedFeatures.Count > 0 ? requestedFeatures : null,
                    RequiredLimits = requiredLimits
                };
                _gpuDevice = await adapter.RequestDevice(descriptor);
                Console.WriteLine($"[WebGPU] Device created with features + limits (maxStorage={maxStorageBuffers})");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[WebGPU] Device creation with features failed: {ex.Message}");
                // Fall back: try without features but with limits
                try
                {
                    var descriptor = new GPUDeviceDescriptor
                    {
                        RequiredLimits = requiredLimits
                    };
                    _gpuDevice = await adapter.RequestDevice(descriptor);
                    Console.WriteLine($"[WebGPU] Device created with limits only (maxStorage={maxStorageBuffers})");
                    // Clear features since we couldn't request them
                    requestedFeatures.Clear();
                }
                catch (Exception ex2)
                {
                    Console.WriteLine($"[WebGPU] Device creation with limits failed: {ex2.Message}");
                    // Fall back to fully default device
                    _gpuDevice = await adapter.RequestDevice();
                    Console.WriteLine("[WebGPU] Device created with defaults (no limits override)");
                    requestedFeatures.Clear();
                }
            }
            if (_gpuDevice == null)
                throw new InvalidOperationException("Failed to request WebGPU device");

            // Store actually enabled features
            EnabledFeatures = new HashSet<string>(requestedFeatures, StringComparer.OrdinalIgnoreCase);

            // Listen for uncaptured GPU errors (e.g., Invalid ComputePipeline, validation errors)
            _gpuDevice.OnUncapturedError += OnGPUUncapturedError;

            _queue = _gpuDevice.Queue;
            _isInitialized = true;
        }

        /// <summary>
        /// Handles uncaptured GPU validation errors.
        /// </summary>
        private void OnGPUUncapturedError(GPUUncapturedErrorEvent e)
        {
            var message = e.Error?.Message ?? "Unknown GPU error";
            Console.Error.WriteLine($"[WebGPU ERROR] {message}");
            if (_lastCompiledWGSL != null)
            {
                Console.Error.WriteLine($"[WebGPU ERROR] Last WGSL source ({_lastCompiledWGSL.Length} chars):");
                // Dump first 2000 chars to avoid flooding
                var snippet = _lastCompiledWGSL.Length > 2000 ? _lastCompiledWGSL.Substring(0, 2000) + "\n...TRUNCATED..." : _lastCompiledWGSL;
                Console.Error.WriteLine(snippet);
            }
        }

        /// <summary>
        /// The last WGSL source that was compiled, for error debugging.
        /// </summary>
        internal string? _lastCompiledWGSL;

        #endregion

        #region Properties

        /// <summary>
        /// Returns the parent WebGPU device. Null when using an externally-provided GPUDevice.
        /// </summary>
        public WebGPUDevice? Device { get; }

        /// <summary>
        /// Returns the native GPU device.
        /// </summary>
        public GPUDevice? NativeDevice => _gpuDevice;

        /// <summary>
        /// Returns the GPU command queue.
        /// </summary>
        public GPUQueue? Queue => _queue;

        /// <summary>
        /// Returns whether the accelerator is initialized.
        /// </summary>
        public bool IsInitialized => _isInitialized;

        /// <summary>
        /// Gets the set of WebGPU features that were successfully enabled on this device.
        /// </summary>
        public HashSet<string> EnabledFeatures { get; private set; } = new(StringComparer.OrdinalIgnoreCase);

        /// <summary>Returns true if shader-f16 is enabled on this device.</summary>
        public bool HasShaderF16 => EnabledFeatures.Contains("shader-f16");

        /// <summary>Returns true if subgroups are enabled on this device.</summary>
        public bool HasSubgroups => EnabledFeatures.Contains("subgroups");

        /// <summary>Returns true if timestamp queries are enabled on this device.</summary>
        public bool HasTimestampQuery => EnabledFeatures.Contains("timestamp-query");

        /// <summary>
        /// Optional callback to flush pending batched ILGPU kernel dispatches.
        /// Set by WebGPUAccelerator to allow WebGPUBuffer readback operations
        /// to auto-flush before copying, ensuring kernel results are available.
        /// </summary>
        internal Action? FlushPendingCommands { get; set; }

        #endregion

        #region Buffer Methods

        /// <summary>
        /// Allocates a GPU buffer with the specified size.
        /// </summary>
        public WebGPUBuffer<T> Allocate<T>(long length) where T : unmanaged
        {
            EnsureInitialized();
            return new WebGPUBuffer<T>(this, length);
        }

        /// <summary>
        /// Allocates a GPU buffer and copies data from an array.
        /// </summary>
        public WebGPUBuffer<T> Allocate<T>(T[] data) where T : unmanaged
        {
            EnsureInitialized();
            var buffer = new WebGPUBuffer<T>(this, data.Length);
            buffer.CopyFromHost(data);
            return buffer;
        }

        #endregion

        #region Compute Methods

        /// <summary>
        /// Creates a compute shader from WGSL source code.
        /// </summary>
        public WebGPUComputeShader CreateComputeShader(
            string wgslSource,
            string entryPoint = "main",
            Dictionary<string, object>? overrideConstants = null)
        {
            EnsureInitialized();
            return new WebGPUComputeShader(this, wgslSource, entryPoint, overrideConstants);
        }

        /// <summary>
        /// Gets a cached compute shader or creates a new one.
        /// When caching is enabled, shaders are reused across kernel invocations.
        /// Override constants are part of the cache key since they affect pipeline creation.
        /// </summary>
        public WebGPUComputeShader GetOrCreateComputeShader(
            string wgslSource,
            string entryPoint = "main",
            Dictionary<string, object>? overrideConstants = null)
        {
            if (!WebGPUBackend.EnableShaderCaching)
                return CreateComputeShader(wgslSource, entryPoint, overrideConstants);

            // Build cache key: WGSL source + sorted override constants
            string cacheKey = wgslSource;
            if (overrideConstants != null && overrideConstants.Count > 0)
            {
                var constantsKey = string.Join(",",
                    overrideConstants.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}={kv.Value}"));
                cacheKey = $"{wgslSource}\n__CONSTANTS__:{constantsKey}";
            }

            if (_shaderCache.TryGetValue(cacheKey, out var cached))
            {
                WebGPUBackend.Log("[WebGPU] Using cached shader");
                return cached;
            }

            WebGPUBackend.Log("[WebGPU] Creating and caching new shader");
            var shader = CreateComputeShader(wgslSource, entryPoint, overrideConstants);
            _shaderCache[cacheKey] = shader;
            return shader;
        }

        /// <summary>
        /// Runs a compute shader with the specified dispatch size.
        /// </summary>
        public void Dispatch(WebGPUComputeShader shader, uint workgroupCountX, uint workgroupCountY = 1, uint workgroupCountZ = 1)
        {
            EnsureInitialized();

            var encoder = _gpuDevice!.CreateCommandEncoder();
            var passEncoder = encoder.BeginComputePass();

            passEncoder.SetPipeline(shader.Pipeline!);

            if (shader.BindGroup != null)
            {
                passEncoder.SetBindGroup(0, shader.BindGroup);
            }

            passEncoder.DispatchWorkgroups(workgroupCountX, workgroupCountY, workgroupCountZ);
            passEncoder.End();

            var commandBuffer = encoder.Finish();
            _queue!.Submit(new[] { commandBuffer });

            // Clean up
            commandBuffer.Dispose();
            passEncoder.Dispose();
            encoder.Dispose();
        }

        #endregion

        #region Helpers

        private void EnsureInitialized()
        {
            if (!_isInitialized)
                throw new InvalidOperationException(
                    "Accelerator not initialized. Call InitializeAsync first.");
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            // Dispose cached shaders
            foreach (var shader in _shaderCache.Values)
            {
                shader.Dispose();
            }
            _shaderCache.Clear();

            _gpuDevice?.Destroy();
            _gpuDevice?.Dispose();
        }

        #endregion
    }
}
