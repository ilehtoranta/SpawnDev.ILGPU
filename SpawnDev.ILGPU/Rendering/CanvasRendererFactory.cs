using global::ILGPU.Runtime;
using SpawnDev.ILGPU.WebGL;
using SpawnDev.ILGPU.WebGL.Rendering;
using SpawnDev.ILGPU.WebGPU;
using SpawnDev.ILGPU.WebGPU.Rendering;

namespace SpawnDev.ILGPU.Rendering
{
    /// <summary>
    /// Creates the optimal <see cref="ICanvasRenderer"/> for the given accelerator type.
    /// </summary>
    public static class CanvasRendererFactory
    {
        public static ICanvasRenderer Create(Accelerator accelerator) => accelerator switch
        {
            WebGPUAccelerator wgpu => new WebGPUCanvasRenderer(wgpu),
            WebGLAccelerator wgl   => new WebGLCanvasRenderer(wgl),
            _                      => new CPUCanvasRenderer(accelerator),
        };
    }
}
