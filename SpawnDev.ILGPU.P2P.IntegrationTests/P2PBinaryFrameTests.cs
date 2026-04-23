using NUnit.Framework;

namespace SpawnDev.ILGPU.P2P.IntegrationTests;

/// <summary>
/// Unit tests for <see cref="P2PBinaryFrame"/>. These run in-process with no
/// accelerator, no network, no tracker - pure serialization coverage. Every
/// real-WebRTC test depends on this frame being right, so these unit tests are
/// the canary for any round-trip correctness regression.
/// </summary>
[TestFixture]
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

    [Test]
    public void Encode_BufferSend_FirstByteIs_0x02()
    {
        var frame = P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferSend, MakeChunk());
        Assert.That(frame[0], Is.EqualTo((byte)0x02));
    }

    [Test]
    public void Encode_BufferData_FirstByteIs_0x03()
    {
        var frame = P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferData, MakeChunk());
        Assert.That(frame[0], Is.EqualTo((byte)0x03));
    }

    [Test]
    public void Encode_NonBufferType_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.KernelDispatch, MakeChunk()));
    }

    [Test]
    public void RoundTrip_TinyChunk_PreservesAllFields()
    {
        var original = MakeChunk(
            bufferId: "scale_in",
            chunkIndex: 3,
            totalChunks: 7,
            totalBytes: 10_240_000,
            data: new byte[] { 0xDE, 0xAD, 0xBE, 0xEF });

        var frame = P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferSend, original);

        Assert.That(P2PBinaryFrame.TryDecodeBufferChunk(frame, out var decoded, out var type), Is.True);
        Assert.That(type, Is.EqualTo(P2PMessageType.BufferSend));
        Assert.That(decoded.BufferId, Is.EqualTo(original.BufferId));
        Assert.That(decoded.ChunkIndex, Is.EqualTo(original.ChunkIndex));
        Assert.That(decoded.TotalChunks, Is.EqualTo(original.TotalChunks));
        Assert.That(decoded.TotalBytes, Is.EqualTo(original.TotalBytes));
        Assert.That(decoded.Data, Is.EqualTo(original.Data));
    }

    [Test]
    public void RoundTrip_RealisticChunk_64KB_PreservesData()
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

        // Frame length = header + bufferIdLen + payload
        Assert.That(frame.Length,
            Is.EqualTo(P2PBinaryFrame.FixedHeaderSize + 8 /* "tensor_a" */ + payload.Length));

        Assert.That(P2PBinaryFrame.TryDecodeBufferChunk(frame, out var decoded, out var type), Is.True);
        Assert.That(type, Is.EqualTo(P2PMessageType.BufferData));
        Assert.That(decoded.Data, Is.EqualTo(payload));
    }

    [Test]
    public void RoundTrip_LargeChunk_NearMaxFrame_Works()
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
        Assert.That(frame.Length, Is.LessThanOrEqualTo(P2PBinaryFrame.MaxFrameSize));

        Assert.That(P2PBinaryFrame.TryDecodeBufferChunk(frame, out var decoded, out _), Is.True);
        Assert.That(decoded.Data.Length, Is.EqualTo(payload.Length));
        // Spot-check first, middle, and last bytes rather than full compare for speed.
        Assert.That(decoded.Data[0], Is.EqualTo((byte)0));
        Assert.That(decoded.Data[payload.Length / 2], Is.EqualTo(payload[payload.Length / 2]));
        Assert.That(decoded.Data[^1], Is.EqualTo(payload[^1]));
    }

    [Test]
    public void Encode_ExceedsMaxFrame_Throws()
    {
        var oversized = new byte[P2PBinaryFrame.MaxFrameSize + 1];
        var chunk = MakeChunk(bufferId: "id", data: oversized);
        Assert.Throws<ArgumentException>(() =>
            P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferSend, chunk));
    }

    [Test]
    public void Encode_BufferIdExceedsMax_Throws()
    {
        var tooLong = new string('x', P2PBinaryFrame.MaxBufferIdBytes + 1);
        var chunk = MakeChunk(bufferId: tooLong);
        Assert.Throws<ArgumentException>(() =>
            P2PBinaryFrame.EncodeBufferChunk(P2PMessageType.BufferSend, chunk));
    }

    [Test]
    public void TryDecode_NullBytes_ReturnsFalse()
    {
        Assert.That(P2PBinaryFrame.TryDecodeBufferChunk(null!, out _, out _), Is.False);
    }

    [Test]
    public void TryDecode_EmptyBytes_ReturnsFalse()
    {
        Assert.That(P2PBinaryFrame.TryDecodeBufferChunk(Array.Empty<byte>(), out _, out _), Is.False);
    }

    [Test]
    public void TryDecode_NonBufferTypeByte_ReturnsFalse()
    {
        // 0x7B = '{' (JSON); should be rejected as non-binary.
        var bogus = new byte[P2PBinaryFrame.FixedHeaderSize];
        bogus[0] = 0x7B;
        Assert.That(P2PBinaryFrame.TryDecodeBufferChunk(bogus, out _, out _), Is.False);
    }

    [Test]
    public void TryDecode_TruncatedHeader_ReturnsFalse()
    {
        // Valid prefix but shorter than FixedHeaderSize - still not enough to even read bufferIdLen.
        var truncated = new byte[] { 0x02, 0x00, 0x05, 0x00 };
        Assert.That(P2PBinaryFrame.TryDecodeBufferChunk(truncated, out _, out _), Is.False);
    }

    [Test]
    public void TryDecode_BufferIdLenOverrunsFrame_ReturnsFalse()
    {
        // Claim 9999 bytes of bufferId in a 30-byte frame - should fail the header-end bounds check.
        var malformed = new byte[30];
        malformed[0] = 0x02;
        malformed[1] = (byte)(9999 >> 8);
        malformed[2] = (byte)(9999 & 0xFF);
        Assert.That(P2PBinaryFrame.TryDecodeBufferChunk(malformed, out _, out _), Is.False);
    }

    [Test]
    public void TryDecode_NegativeChunkIndex_ReturnsFalse()
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
        Assert.That(P2PBinaryFrame.TryDecodeBufferChunk(frame, out _, out _), Is.False);
    }

    [Test]
    public void TryDecode_ChunkIndexBeyondTotal_ReturnsFalse()
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
        Assert.That(P2PBinaryFrame.TryDecodeBufferChunk(frame, out _, out _), Is.False);
    }

    [Test]
    public void IsBinaryFrame_Detects_0x02_and_0x03()
    {
        Assert.That(P2PBinaryFrame.IsBinaryFrame(new byte[] { 0x02 }), Is.True);
        Assert.That(P2PBinaryFrame.IsBinaryFrame(new byte[] { 0x03 }), Is.True);
        Assert.That(P2PBinaryFrame.IsBinaryFrame(new byte[] { 0x7B }), Is.False); // JSON
        Assert.That(P2PBinaryFrame.IsBinaryFrame(Array.Empty<byte>()), Is.False);
        Assert.That(P2PBinaryFrame.IsBinaryFrame(null!), Is.False);
    }
}
