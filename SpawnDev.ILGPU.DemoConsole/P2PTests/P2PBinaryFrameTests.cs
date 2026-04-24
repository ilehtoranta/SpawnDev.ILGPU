using SpawnDev.ILGPU.P2P;
using SpawnDev.UnitTesting;

namespace SpawnDev.ILGPU.DemoConsole.P2PTests;

/// <summary>
/// Unit tests for <see cref="P2PBinaryFrame"/>. Run in-process with no accelerator,
/// no network, no tracker - pure serialization coverage. Every real-WebRTC test
/// depends on this frame being right, so these unit tests are the canary for any
/// round-trip correctness regression.
/// </summary>
public class P2PBinaryFrameTests
{
    private static BufferChunk MakeChunk(
        string bufferId = "tensor-abc",
        int chunkIndex = 0,
        int totalChunks = 1,
        int totalBytes = 1024,
        byte[]? data = null)
        => new()
        {
            BufferId = bufferId,
            ChunkIndex = chunkIndex,
            TotalChunks = totalChunks,
            TotalBytes = totalBytes,
            Data = data ?? new byte[] { 1, 2, 3, 4, 5 },
        };

    [TestMethod]
    public Task Encode_BufferSend_FirstByteIs_0x02()
    {
        var frame = P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferSend, MakeChunk());
        if (frame[0] != 0x02) throw new Exception($"Expected 0x02, got 0x{frame[0]:X2}");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task Encode_BufferData_FirstByteIs_0x03()
    {
        var frame = P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferData, MakeChunk());
        if (frame[0] != 0x03) throw new Exception($"Expected 0x03, got 0x{frame[0]:X2}");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task Encode_NonBufferType_Throws()
    {
        try
        {
            P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.KernelDispatch, MakeChunk());
        }
        catch (ArgumentException)
        {
            return Task.CompletedTask;
        }
        throw new Exception("Expected ArgumentException for non-buffer type");
    }

    [TestMethod]
    public Task RoundTrip_TinyChunk_PreservesAllFields()
    {
        var original = MakeChunk(
            bufferId: "scale_in",
            chunkIndex: 3,
            totalChunks: 7,
            totalBytes: 10_240_000,
            data: new byte[] { 0xDE, 0xAD, 0xBE, 0xEF });

        var frame = P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferSend, original);

        if (!P2PBinaryFrame.TryDecodeBufferChunk(frame, out var decoded, out var type))
            throw new Exception("TryDecodeBufferChunk returned false for valid frame");
        if (type != P2PMessageType.BufferSend) throw new Exception($"Expected BufferSend, got {type}");
        if (decoded.BufferId != original.BufferId) throw new Exception($"BufferId mismatch: {decoded.BufferId}");
        if (decoded.ChunkIndex != original.ChunkIndex) throw new Exception($"ChunkIndex mismatch: {decoded.ChunkIndex}");
        if (decoded.TotalChunks != original.TotalChunks) throw new Exception($"TotalChunks mismatch: {decoded.TotalChunks}");
        if (decoded.TotalBytes != original.TotalBytes) throw new Exception($"TotalBytes mismatch: {decoded.TotalBytes}");
        if (!decoded.Data.SequenceEqual(original.Data)) throw new Exception("Data mismatch");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task RoundTrip_RealisticChunk_64KB_PreservesData()
    {
        var rng = new Random(42);
        var payload = new byte[64 * 1024];
        rng.NextBytes(payload);

        var original = MakeChunk(
            bufferId: "tensor_a",
            chunkIndex: 5,
            totalChunks: 160,
            totalBytes: 10 * 1024 * 1024,
            data: payload);

        var frame = P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferData, original);
        var expectedLen = P2PBinaryFrame.FixedHeaderSize + 8 /* "tensor_a" */ + payload.Length;
        if (frame.Length != expectedLen)
            throw new Exception($"Frame length: expected {expectedLen}, got {frame.Length}");

        if (!P2PBinaryFrame.TryDecodeBufferChunk(frame, out var decoded, out var type))
            throw new Exception("TryDecodeBufferChunk returned false");
        if (type != P2PMessageType.BufferData) throw new Exception($"Expected BufferData, got {type}");
        if (!decoded.Data.SequenceEqual(payload)) throw new Exception("Payload mismatch");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task RoundTrip_LargeChunk_NearMaxFrame_Works()
    {
        var payload = new byte[P2PBinaryFrame.MaxFrameSize - P2PBinaryFrame.FixedHeaderSize - 16];
        for (int i = 0; i < payload.Length; i++)
            payload[i] = (byte)(i & 0xFF);

        var original = MakeChunk(
            bufferId: "xl_tensor",       // 9 utf-8 bytes - fits in the -16 budget above
            chunkIndex: 0,
            totalChunks: 1,
            totalBytes: payload.Length,
            data: payload);

        var frame = P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferSend, original);
        if (frame.Length > P2PBinaryFrame.MaxFrameSize)
            throw new Exception($"Frame exceeds MaxFrameSize: {frame.Length}");

        if (!P2PBinaryFrame.TryDecodeBufferChunk(frame, out var decoded, out _))
            throw new Exception("TryDecodeBufferChunk returned false");
        if (decoded.Data.Length != payload.Length)
            throw new Exception($"Data length: expected {payload.Length}, got {decoded.Data.Length}");
        if (decoded.Data[0] != 0) throw new Exception($"Data[0] mismatch: {decoded.Data[0]}");
        if (decoded.Data[payload.Length / 2] != payload[payload.Length / 2])
            throw new Exception("Data midpoint mismatch");
        if (decoded.Data[^1] != payload[^1]) throw new Exception("Data tail mismatch");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task Encode_ExceedsMaxFrame_Throws()
    {
        var oversized = new byte[P2PBinaryFrame.MaxFrameSize + 1];
        var chunk = MakeChunk(bufferId: "id", data: oversized);
        try
        {
            P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferSend, chunk);
        }
        catch (ArgumentException) { return Task.CompletedTask; }
        throw new Exception("Expected ArgumentException for oversized frame");
    }

    [TestMethod]
    public Task Encode_BufferIdExceedsMax_Throws()
    {
        var tooLong = new string('x', P2PBinaryFrame.MaxBufferIdBytes + 1);
        var chunk = MakeChunk(bufferId: tooLong);
        try
        {
            P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferSend, chunk);
        }
        catch (ArgumentException) { return Task.CompletedTask; }
        throw new Exception("Expected ArgumentException for oversized bufferId");
    }

    [TestMethod]
    public Task TryDecode_NullBytes_ReturnsFalse()
    {
        if (P2PBinaryFrame.TryDecodeBufferChunk(null!, out _, out _))
            throw new Exception("Expected false for null input");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task TryDecode_EmptyBytes_ReturnsFalse()
    {
        if (P2PBinaryFrame.TryDecodeBufferChunk(Array.Empty<byte>(), out _, out _))
            throw new Exception("Expected false for empty input");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task TryDecode_NonBufferTypeByte_ReturnsFalse()
    {
        // 0x7B = '{' (JSON); should be rejected as non-binary.
        var bogus = new byte[P2PBinaryFrame.FixedHeaderSize];
        bogus[0] = 0x7B;
        if (P2PBinaryFrame.TryDecodeBufferChunk(bogus, out _, out _))
            throw new Exception("Expected false for JSON-byte input");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task TryDecode_TruncatedHeader_ReturnsFalse()
    {
        // Valid prefix but shorter than FixedHeaderSize - still not enough to even read bufferIdLen.
        var truncated = new byte[] { 0x02, 0x00, 0x05, 0x00 };
        if (P2PBinaryFrame.TryDecodeBufferChunk(truncated, out _, out _))
            throw new Exception("Expected false for truncated header");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task TryDecode_BufferIdLenOverrunsFrame_ReturnsFalse()
    {
        // Claim 9999 bytes of bufferId in a 30-byte frame - should fail the header-end bounds check.
        var malformed = new byte[30];
        malformed[0] = 0x02;
        malformed[1] = (byte)(9999 >> 8);
        malformed[2] = (byte)(9999 & 0xFF);
        if (P2PBinaryFrame.TryDecodeBufferChunk(malformed, out _, out _))
            throw new Exception("Expected false for overflow bufferIdLen");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task TryDecode_NegativeChunkIndex_ReturnsFalse()
    {
        var original = MakeChunk();
        var frame = P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferSend, original);
        // Overwrite chunkIndex with -1 (0xFFFFFFFF big-endian).
        var bufferIdLen = (frame[1] << 8) | frame[2];
        var cursor = 3 + bufferIdLen;
        frame[cursor] = 0xFF;
        frame[cursor + 1] = 0xFF;
        frame[cursor + 2] = 0xFF;
        frame[cursor + 3] = 0xFF;
        if (P2PBinaryFrame.TryDecodeBufferChunk(frame, out _, out _))
            throw new Exception("Expected false for negative chunkIndex");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task TryDecode_ChunkIndexBeyondTotal_ReturnsFalse()
    {
        var original = MakeChunk(chunkIndex: 0, totalChunks: 1);
        var frame = P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferSend, original);
        // Overwrite chunkIndex = totalChunks (boundary miss - must be strictly less).
        var bufferIdLen = (frame[1] << 8) | frame[2];
        var cursor = 3 + bufferIdLen;
        frame[cursor] = 0x00;
        frame[cursor + 1] = 0x00;
        frame[cursor + 2] = 0x00;
        frame[cursor + 3] = 0x01;
        if (P2PBinaryFrame.TryDecodeBufferChunk(frame, out _, out _))
            throw new Exception("Expected false for chunkIndex >= totalChunks");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task IsBinaryFrame_Detects_0x02_and_0x03()
    {
        if (!P2PBinaryFrame.IsBinaryFrame(new byte[] { 0x02 })) throw new Exception("0x02 should be binary");
        if (!P2PBinaryFrame.IsBinaryFrame(new byte[] { 0x03 })) throw new Exception("0x03 should be binary");
        if (P2PBinaryFrame.IsBinaryFrame(new byte[] { 0x7B })) throw new Exception("0x7B (JSON) should NOT be binary");
        if (P2PBinaryFrame.IsBinaryFrame(Array.Empty<byte>())) throw new Exception("empty should NOT be binary");
        if (P2PBinaryFrame.IsBinaryFrame(null!)) throw new Exception("null should NOT be binary");
        return Task.CompletedTask;
    }
}
