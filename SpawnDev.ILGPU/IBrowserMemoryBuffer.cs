using SpawnDev.BlazorJS.JSObjects;

namespace SpawnDev.ILGPU
{
    /// <summary>
    /// Defines a contract for managing a memory buffer in a browser environment and provides asynchronous methods to
    /// copy its contents to a host-side Uint8Array.
    /// </summary>
    /// <remarks>Implementations of this interface enable efficient transfer of memory buffer data from
    /// browser-managed memory to .NET-managed arrays, which is useful for interoperability scenarios such as
    /// WebAssembly or JavaScript interop in web applications. The asynchronous nature of the copy operation allows for
    /// non-blocking data transfers, which can improve application responsiveness.</remarks>
    public interface IBrowserMemoryBuffer
    {
        /// <summary>
        /// Asynchronously copies a specified range of bytes from the buffer to a new Uint8Array on the host.
        /// </summary>
        /// <remarks>Use this method to transfer data from the buffer to a host-accessible Uint8Array for
        /// further processing or interoperability with JavaScript APIs. Ensure that the specified offset and byte count
        /// do not exceed the bounds of the source buffer to avoid errors.</remarks>
        /// <param name="sourceByteOffset">The zero-based byte offset in the source buffer at which to begin copying. Must be greater than or equal to
        /// 0.</param>
        /// <param name="copyBytes">The number of bytes to copy from the source buffer. If null, copies all bytes from the specified offset to
        /// the end of the buffer.</param>
        /// <returns>A task that represents the asynchronous copy operation. The task result contains a Uint8Array with the
        /// copied bytes.</returns>
        Task<Uint8Array> CopyToHostUint8ArrayAsync(long sourceByteOffset = 0, long? copyBytes = null);
    }
}
