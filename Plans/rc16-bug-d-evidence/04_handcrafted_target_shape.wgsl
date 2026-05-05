@group(0) @binding(0) var<storage, read_write> param1 : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> param2 : array<i32>;

fn read_byte(buf: ptr<storage, array<atomic<u32>>, read_write>, idx: i32) -> i32 {
    let word_idx = u32(idx) / 4u;
    let byte_off = (u32(idx) % 4u) * 8u;
    return i32((u32(atomicLoad(&((*buf)[word_idx]))) >> byte_off) & 0xFFu);
}

fn read_int(buf: ptr<storage, array<i32>, read_write>, idx: i32) -> i32 {
    return (*buf)[idx];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let i = i32(gid.x);
    if (i >= 4) { return; }
    let b = read_byte(&param1, i);
    let v = read_int(&param2, i);
    param2[i] = b + v;
}
