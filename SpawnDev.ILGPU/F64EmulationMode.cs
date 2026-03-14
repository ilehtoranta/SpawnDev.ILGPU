namespace SpawnDev.ILGPU
{
    /// <summary>
    /// Controls how 64-bit float (double) values are emulated on GPU backends
    /// that lack native f64 support.
    /// </summary>
    public enum F64EmulationMode
    {
        /// <summary>
        /// Dekker double-float technique using two f32 values.
        /// ~48-53 bits of mantissa, 8 bytes per value. Good balance of precision and performance.
        /// </summary>
        Dekker,

        /// <summary>
        /// Ozaki quad-float technique using four f32 values.
        /// Strict IEEE 754 double precision, 16 bytes per value. ~2x slower than Dekker.
        /// </summary>
        Ozaki,

        /// <summary>
        /// No emulation — double is promoted to float (f32). Maximum performance, loses precision.
        /// </summary>
        Disabled,
    }
}
