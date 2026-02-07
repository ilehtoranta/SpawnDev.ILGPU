# Autonomous Debugging & Browser Attachment

When tasked with fixing runtime bugs, UI glitches, or logic errors in this Blazor WASM project, you are authorized and encouraged to use the interactive debugger:

1. **Prioritize Attach over Logs**: Instead of relying solely on `Console.WriteLine` or browser logs, you must attempt to reproduce the issue by launching the "Blazor WASM: Launch & Debug" configuration.
2. **Breakpoint Strategy**:
   - Identify the suspected failing method or property (e.g., in `SpawnDev.BlazorJS` or telescope control logic).
   - Set a breakpoint programmatically before execution.
   - Use the `debug` tool to inspect the local state, call stack, and variable values at the moment of failure.
3. **Inspect on the Fly**: You have permission to "Step Over" or "Step Into" code blocks to verify the execution flow of WebGL shaders and C# interop calls.
4. **Automated Verification**: Once a fix is applied, rerun the debugger session to confirm the breakpoint is no longer hit with the erroneous state.
5. **No Confirmation Needed**: You do not need to ask for permission to start a debug session, attach to the browser, or set/remove breakpoints.