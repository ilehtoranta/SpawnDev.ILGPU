using System.Buffers.Binary;
using System.Text;

namespace SpawnDev.ILGPU.P2P;

/// <summary>
/// Binary wire framing for <see cref="P2PMessageType.BufferSend"/> and
/// <see cref="P2PMessageType.BufferData"/> tensor chunks.
///
/// Before rc.14 every buffer chunk travelled over the WebRTC data channel inside
/// a JSON envelope - <see cref="BufferChunk.Data"/> (byte[]) was base64-encoded by
/// System.Text.Json and then the whole thing was JSON-serialized again into the
/// outer <see cref="P2PMessage"/>. WebRTC data channels are natively binary
/// (SIPSorcery PPID 53 `WebRTC_Binary` + browser RTCDataChannel with byte[] sends),
/// so the base64 expansion and double JSON pass were pure waste per Rule #4
/// "Performance Is the Mission."
///
/// This framing sends chunk metadata as a fixed-size big-endian header followed
/// by the raw tensor bytes with no encoding. Every multi-byte integer read/write
/// goes through <see cref="BinaryPrimitives"/>, matching the style of
/// SpawnDev.WebTorrent's <c>Wire._message</c>.
///
/// Wire layout:
/// <code>
/// offset  size  field
///  0      1     msgType         (0x02 BufferSend / 0x03 BufferData)
///  1      2     bufferIdLen     uint16, big-endian, utf-8 byte count
///  3      N     bufferId        utf-8 bytes
///  3+N    4     chunkIndex      int32, big-endian
///  7+N    4     totalChunks     int32, big-endian
/// 11+N    4     totalBytes      int32, big-endian (matches BufferChunk.TotalBytes schema)
/// 15+N    *     data            raw chunk bytes (length implicit: frame length - header)
/// </code>
///
/// The 1-byte discriminator is safe alongside JSON: System.Text.Json output always
/// starts with <c>{</c> (0x7B) for an object, so bytes 0x00-0x1F are reserved for
/// binary framing. The receive side peeks <c>data[0]</c> to route.
/// </summary>
public static class P2PBinaryFrame
{
    /// <summary>
    /// Size of the fixed header (everything except bufferId and data).
    /// 1 (type) + 2 (bufferIdLen) + 4 (chunkIndex) + 4 (totalChunks) + 4 (totalBytes) = 15 bytes.
    /// </summary>
    public const int FixedHeaderSize = 1 + 2 + 4 + 4 + 4;

    /// <summary>
    /// Absolute maximum bufferId length in utf-8 bytes. Keeps bufferIdLen sane and bounded
    /// so a malformed header cannot point past the end of a short frame.
    /// </summary>
    public const int MaxBufferIdBytes = 1024;

    /// <summary>
    /// Absolute wire-message cap. SIPSorcery's <c>RTCDataChannel.send</c> throws
    /// <c>ApplicationException</c> above 262,144 bytes (SCTP_DEFAULT_MAX_MESSAGE_SIZE).
    /// </summary>
    public const int MaxFrameSize = 256 * 1024;

    /// <summary>
    /// Encode a BufferSend (0x02) or BufferData (0x03) chunk to a binary wire frame.
    /// Throws if the frame would exceed <see cref="MaxFrameSize"/> (caller should chunk smaller).
    /// </summary>
    public static byte[] EncodeBufferChunk(P2PMessageType type, BufferChunk chunk)
    {
        if (type != P2PMessageType.BufferSend && type != P2PMessageType.BufferData)
            throw new ArgumentException(
                $"Binary framing only supports BufferSend / BufferData, got {type}", nameof(type));
        ArgumentNullException.ThrowIfNull(chunk);
        ArgumentNullException.ThrowIfNull(chunk.Data);

        var bufferIdBytes = Encoding.UTF8.GetBytes(chunk.BufferId ?? string.Empty);
        if (bufferIdBytes.Length > MaxBufferIdBytes)
            throw new ArgumentException(
                $"bufferId utf-8 length {bufferIdBytes.Length} exceeds MaxBufferIdBytes {MaxBufferIdBytes}",
                nameof(chunk));

        var frameSize = FixedHeaderSize + bufferIdBytes.Length + chunk.Data.Length;
        if (frameSize > MaxFrameSize)
            throw new ArgumentException(
                $"Frame size {frameSize} exceeds MaxFrameSize {MaxFrameSize}; reduce MaxChunkSize.",
                nameof(chunk));

        var frame = new byte[frameSize];
        var span = frame.AsSpan();

        span[0] = (byte)(type == P2PMessageType.BufferSend ? 0x02 : 0x03);
        BinaryPrimitives.WriteUInt16BigEndian(span.Slice(1, 2), (ushort)bufferIdBytes.Length);
        bufferIdBytes.CopyTo(span.Slice(3, bufferIdBytes.Length));
        var cursor = 3 + bufferIdBytes.Length;
        BinaryPrimitives.WriteInt32BigEndian(span.Slice(cursor, 4), chunk.ChunkIndex);
        cursor += 4;
        BinaryPrimitives.WriteInt32BigEndian(span.Slice(cursor, 4), chunk.TotalChunks);
        cursor += 4;
        BinaryPrimitives.WriteInt32BigEndian(span.Slice(cursor, 4), chunk.TotalBytes);
        cursor += 4;
        chunk.Data.CopyTo(span.Slice(cursor, chunk.Data.Length));

        return frame;
    }

    /// <summary>
    /// Attempt to decode a binary buffer-chunk frame. Returns false for malformed or
    /// non-binary frames (never throws on bad input so the wire reader can cheaply
    /// discard and move on).
    /// </summary>
    public static bool TryDecodeBufferChunk(byte[] data, out BufferChunk chunk, out P2PMessageType type)
    {
        chunk = default!;
        type = default;

        if (data == null || data.Length < FixedHeaderSize) return false;
        var span = data.AsSpan();

        var typeByte = span[0];
        if (typeByte == 0x02) type = P2PMessageType.BufferSend;
        else if (typeByte == 0x03) type = P2PMessageType.BufferData;
        else return false;

        int bufferIdLen = BinaryPrimitives.ReadUInt16BigEndian(span.Slice(1, 2));
        if (bufferIdLen > MaxBufferIdBytes) return false;

        // Header fully in bounds?
        var headerEnd = FixedHeaderSize + bufferIdLen;
        if (headerEnd > span.Length) return false;

        string bufferId;
        try { bufferId = Encoding.UTF8.GetString(span.Slice(3, bufferIdLen)); }
        catch { return false; }

        var cursor = 3 + bufferIdLen;
        int chunkIndex = BinaryPrimitives.ReadInt32BigEndian(span.Slice(cursor, 4));
        cursor += 4;
        int totalChunks = BinaryPrimitives.ReadInt32BigEndian(span.Slice(cursor, 4));
        cursor += 4;
        int totalBytes = BinaryPrimitives.ReadInt32BigEndian(span.Slice(cursor, 4));
        cursor += 4;

        // Sanity: reassembler in P2PBufferTransfer.ReceiveChunk will re-validate these too,
        // but catching here lets us drop obviously-malformed frames without the allocation.
        if (chunkIndex < 0) return false;
        if (totalChunks < 1) return false;
        if (totalBytes < 0) return false;
        if (chunkIndex >= totalChunks) return false;

        var dataLen = span.Length - cursor;
        if (dataLen < 0) return false;
        var chunkData = span.Slice(cursor, dataLen).ToArray();

        chunk = new BufferChunk
        {
            BufferId = bufferId,
            ChunkIndex = chunkIndex,
            TotalChunks = totalChunks,
            TotalBytes = totalBytes,
            Data = chunkData,
        };
        return true;
    }

    /// <summary>
    /// Fast non-allocating check: is this a binary buffer-chunk frame?
    /// Used by the receive-side dispatcher before deciding to JSON-parse.
    /// </summary>
    public static bool IsBinaryFrame(byte[] data)
        => data != null && data.Length > 0 && (data[0] == 0x02 || data[0] == 0x03);
}
