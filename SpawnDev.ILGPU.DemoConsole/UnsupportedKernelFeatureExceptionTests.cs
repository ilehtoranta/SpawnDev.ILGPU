using ILGPU.Runtime;
using SpawnDev.ILGPU;
using SpawnDev.UnitTesting;

/// <summary>
/// Desktop unit tests for <see cref="UnsupportedKernelFeatureException"/>. Locks down
/// the public contract consumers will see when a kernel compiles for the wrong backend:
/// Feature name, Backend enum, Remediation text, and the
/// <see cref="NotSupportedException"/> base class (so existing catch-by-base code keeps
/// working).
///
/// The end-to-end throw path in <c>GLSLCodeGenerator.GenerateCode(GenericAtomic)</c> is
/// reachable only from the WebGL backend which the desktop context cannot instantiate;
/// that throw site is exercised indirectly whenever any browser test compiles an atomic
/// kernel for WebGL. These tests cover the public API surface directly so a drive-by
/// rename of a property doesn't silently break consumer catches.
/// </summary>
public class UnsupportedKernelFeatureExceptionTests
{
    [TestMethod]
    public Task Ctor_SetsAllThreeProperties()
    {
        var ex = new UnsupportedKernelFeatureException(
            feature: "Atomic.Add",
            backend: AcceleratorType.WebGL,
            remediation: "Use WebGPU.");

        if (ex.Feature != "Atomic.Add")
            throw new Exception($"Feature expected 'Atomic.Add', got '{ex.Feature}'");
        if (ex.Backend != AcceleratorType.WebGL)
            throw new Exception($"Backend expected WebGL, got {ex.Backend}");
        if (ex.Remediation != "Use WebGPU.")
            throw new Exception($"Remediation expected 'Use WebGPU.', got '{ex.Remediation}'");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task Message_IncludesFeatureBackendAndRemediation()
    {
        var ex = new UnsupportedKernelFeatureException(
            feature: "Atomic.CompareExchange",
            backend: AcceleratorType.WebGL,
            remediation: "Switch to WebGPU via AcceleratorRequirements { RequiresAtomics = true }.");

        if (!ex.Message.Contains("Atomic.CompareExchange"))
            throw new Exception($"Message must include feature. Got: '{ex.Message}'");
        if (!ex.Message.Contains("WebGL"))
            throw new Exception($"Message must include backend name. Got: '{ex.Message}'");
        if (!ex.Message.Contains("AcceleratorRequirements"))
            throw new Exception($"Message must include remediation text. Got: '{ex.Message}'");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task InheritsFromNotSupportedException_CatchByBase_Works()
    {
        // Regression: existing caller code likely catches NotSupportedException. The
        // typed exception must keep that inheritance so we don't silently break
        // catch blocks in consuming projects (SpawnDev.Codecs, SpawnDev.ILGPU.ML).
        try
        {
            throw new UnsupportedKernelFeatureException(
                "Atomic.Add",
                AcceleratorType.WebGL,
                "Use WebGPU.");
        }
        catch (NotSupportedException)
        {
            return Task.CompletedTask;
        }
        throw new Exception("Expected NotSupportedException catch to match the typed exception.");
    }

    [TestMethod]
    public Task InheritsFromNotSupportedException_CatchByBase_RetainsTypedProperties()
    {
        // Catching via the base type should not lose access to Feature / Backend /
        // Remediation — consumers may rethrow, log, or inspect via the runtime type.
        try
        {
            throw new UnsupportedKernelFeatureException(
                "Atomic.Add",
                AcceleratorType.WebGL,
                "Switch backends.");
        }
        catch (NotSupportedException caught)
        {
            if (caught is not UnsupportedKernelFeatureException typed)
                throw new Exception($"Expected runtime type UnsupportedKernelFeatureException, got {caught.GetType().Name}");
            if (typed.Feature != "Atomic.Add") throw new Exception("Feature not preserved through base catch.");
            if (typed.Backend != AcceleratorType.WebGL) throw new Exception("Backend not preserved through base catch.");
            if (typed.Remediation != "Switch backends.") throw new Exception("Remediation not preserved through base catch.");
            return Task.CompletedTask;
        }
    }

    [TestMethod]
    public Task ProductionShape_GenericAtomic_WebGLSite_MessageParseable()
    {
        // Mirrors the exact argument shape used at
        // GLSLCodeGenerator.GenerateCode(GenericAtomic): "Atomic.{Kind}" feature, WebGL
        // backend, remediation mentioning AcceleratorRequirements. If the throw-site
        // message template drifts, Tuvok's Codecs pipeline and any other consumer
        // depending on string-matching breaks silently.
        var ex = new UnsupportedKernelFeatureException(
            feature: "Atomic.Add",
            backend: AcceleratorType.WebGL,
            remediation: "WebGL2 vertex shaders have no atomic operations. Use WebGPU, Wasm, or a desktop backend. " +
                         "Prefer AcceleratorRequirements { RequiresAtomics = true } so the selection path filters WebGL out up-front.");

        if (!ex.Message.StartsWith("Kernel feature 'Atomic.Add'"))
            throw new Exception($"Message prefix drifted. Got: '{ex.Message}'");
        if (!ex.Message.Contains("not supported on the WebGL backend"))
            throw new Exception($"Message backend phrasing drifted. Got: '{ex.Message}'");
        return Task.CompletedTask;
    }

    [TestMethod]
    public Task ProductionShape_AtomicCAS_WebGLSite_MessageParseable()
    {
        // Mirrors GLSLCodeGenerator.GenerateCode(AtomicCAS) throw site.
        var ex = new UnsupportedKernelFeatureException(
            feature: "Atomic.CompareExchange",
            backend: AcceleratorType.WebGL,
            remediation: "WebGL2 vertex shaders have no atomic operations. Use WebGPU, Wasm, or a desktop backend. " +
                         "Prefer AcceleratorRequirements { RequiresAtomics = true } so the selection path filters WebGL out up-front.");

        if (ex.Feature != "Atomic.CompareExchange")
            throw new Exception($"Feature expected 'Atomic.CompareExchange', got '{ex.Feature}'");
        if (ex.Backend != AcceleratorType.WebGL)
            throw new Exception($"Backend expected WebGL, got {ex.Backend}");
        return Task.CompletedTask;
    }
}
