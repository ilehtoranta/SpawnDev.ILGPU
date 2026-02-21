using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.Demo.Shared.Kernels
{
    /// <summary>
    /// 3D Raymarching kernel — renders SDF scenes with camera orbit.
    /// Uses float (f32) for maximum cross-platform compatibility.
    /// </summary>
    public static class RaymarchKernels
    {
        public static void Render(
            Index2D index,
            ArrayView2D<uint, Stride2D.DenseX> output,
            int packedSize, int packedConfig, int unused,
            float camTheta, float camPhi, float camDist, float time, float unused2)
        {
            int width = packedSize / 65536;
            int height = packedSize - width * 65536;

            int px = index.X;
            int py = index.Y;
            if (px >= width || py >= height) return;

            // Unpack config
            int sceneType = packedConfig / 256;
            int qualityLevel = packedConfig - sceneType * 256;
            int maxSteps = qualityLevel == 0 ? 32 : (qualityLevel == 1 ? 64 : 128);

            float aspect = (float)width / (float)height;
            float u = (2.0f * px / width - 1.0f) * aspect;
            float v = 2.0f * py / height - 1.0f;

            // Camera
            float sp = MathF.Sin(camPhi);
            float cp = MathF.Cos(camPhi);
            float sth = MathF.Sin(camTheta);
            float cth = MathF.Cos(camTheta);

            float camX = camDist * sp * cth;
            float camY = camDist * cp;
            float camZ = camDist * sp * sth;

            // Forward
            float fLen = MathF.Sqrt(camX * camX + camY * camY + camZ * camZ);
            float fx = -camX / fLen;
            float fy = -camY / fLen;
            float fz = -camZ / fLen;

            // Right = normalize(cross(fwd, (0,1,0)))
            float rx = fz;
            float rz = -fx;
            float rLen = MathF.Sqrt(rx * rx + rz * rz);
            if (rLen < 0.001f) { rx = 1.0f; rz = 0.0f; rLen = 1.0f; }
            rx = rx / rLen;
            rz = rz / rLen;

            // Up = cross(right, fwd)
            float ux = -rz * fy;
            float uy = rz * fx - rx * fz;
            float uz = rx * fy;

            // Ray direction
            float rdx = fx + u * rx + v * ux;
            float rdy = fy + v * uy;
            float rdz = fz + u * rz + v * uz;
            float rdLen = MathF.Sqrt(rdx * rdx + rdy * rdy + rdz * rdz);
            rdx = rdx / rdLen;
            rdy = rdy / rdLen;
            rdz = rdz / rdLen;

            // Raymarch
            float t = 0.0f;
            float hitT = -1.0f;
            int steps = 0;
            int hitObj = 0;

            for (int i = 0; i < maxSteps; i++)
            {
                float qx = camX + t * rdx;
                float qy = camY + t * rdy;
                float qz = camZ + t * rdz;

                float d;

                if (sceneType == 1)
                {
                    // Scene 1: Boolean Operations — hollow sphere with windows
                    float dSphereOuter = MathF.Sqrt(qx * qx + qy * qy + qz * qz) - 1.5f;
                    float dSphereInner = -(MathF.Sqrt(qx * qx + qy * qy + qz * qz) - 1.2f);
                    float dShell = dSphereOuter > dSphereInner ? dSphereOuter : dSphereInner;
                    float dCylX = MathF.Sqrt(qy * qy + qz * qz) - 0.5f;
                    float dCylY = MathF.Sqrt(qx * qx + qz * qz) - 0.5f;
                    float dCylZ = MathF.Sqrt(qx * qx + qy * qy) - 0.5f;
                    float dCylMin = dCylX < dCylY ? dCylX : dCylY;
                    dCylMin = dCylMin < dCylZ ? dCylMin : dCylZ;
                    float dBool = dShell > -dCylMin ? dShell : -dCylMin;
                    float orbAngle = time * 1.2f;
                    float ox = MathF.Cos(orbAngle) * 2.5f;
                    float oz = MathF.Sin(orbAngle) * 2.5f;
                    float dOrb = MathF.Sqrt((qx - ox) * (qx - ox) + qy * qy + (qz - oz) * (qz - oz)) - 0.3f;
                    float dGround = qy + 2.0f;
                    d = dBool < dOrb ? dBool : dOrb;
                    hitObj = d == dOrb ? 2 : 1;
                    float dWithGround = d < dGround ? d : dGround;
                    if (dWithGround == dGround) hitObj = 0;
                    d = dWithGround;
                }
                else if (sceneType == 2)
                {
                    // Scene 2: Infinite Repeat
                    float repSize = 3.0f;
                    float halfRep = repSize * 0.5f;
                    float rqx = qx - repSize * MathF.Floor((qx + halfRep) / repSize);
                    float rqz = qz - repSize * MathF.Floor((qz + halfRep) / repSize);
                    float bounce = MathF.Sin(time * 2.0f + qx * 0.5f + qz * 0.3f) * 0.3f;
                    float dSphere = MathF.Sqrt(rqx * rqx + (qy - bounce) * (qy - bounce) + rqz * rqz) - 0.6f;
                    float dGround = qy + 1.5f;
                    d = dSphere < dGround ? dSphere : dGround;
                    hitObj = d == dGround ? 0 : 1;
                }
                else if (sceneType == 3)
                {
                    // Scene 3: Organic metaballs
                    float pulse1 = 1.0f + MathF.Sin(time * 1.5f) * 0.3f;
                    float pulse2 = 1.0f + MathF.Sin(time * 1.8f + 2.0f) * 0.3f;
                    float pulse3 = 1.0f + MathF.Sin(time * 2.1f + 4.0f) * 0.3f;
                    float p1x = MathF.Sin(time * 0.7f) * 1.5f;
                    float p1y = MathF.Cos(time * 0.5f) * 0.8f;
                    float p2x = MathF.Sin(time * 0.6f + 2.0f) * 1.5f;
                    float p2y = MathF.Cos(time * 0.8f + 1.0f) * 0.8f;
                    float p3x = MathF.Sin(time * 0.9f + 4.0f) * 1.3f;
                    float p3z = MathF.Cos(time * 0.7f + 3.0f) * 1.3f;
                    float d1 = MathF.Sqrt((qx - p1x) * (qx - p1x) + (qy - p1y) * (qy - p1y) + qz * qz) - pulse1;
                    float d2 = MathF.Sqrt((qx - p2x) * (qx - p2x) + (qy - p2y) * (qy - p2y) + qz * qz) - pulse2;
                    float d3 = MathF.Sqrt((qx - p3x) * (qx - p3x) + qy * qy + (qz - p3z) * (qz - p3z)) - pulse3;
                    float k = 4.0f;
                    float h12 = 0.5f + 0.5f * (d2 - d1) / (2.0f / k);
                    if (h12 < 0.0f) h12 = 0.0f;
                    if (h12 > 1.0f) h12 = 1.0f;
                    float blend12 = d1 * h12 + d2 * (1.0f - h12) - (1.0f / k) * h12 * (1.0f - h12);
                    float h123 = 0.5f + 0.5f * (d3 - blend12) / (2.0f / k);
                    if (h123 < 0.0f) h123 = 0.0f;
                    if (h123 > 1.0f) h123 = 1.0f;
                    float dOrganic = blend12 * h123 + d3 * (1.0f - h123) - (1.0f / k) * h123 * (1.0f - h123);
                    float dGround = qy + 2.0f;
                    d = dOrganic < dGround ? dOrganic : dGround;
                    hitObj = d == dGround ? 0 : 1;
                }
                else
                {
                    // Scene 0: Geometric Blend — sphere + torus
                    float bounce = MathF.Sin(time * 1.5f) * 0.8f;
                    float dSphere = MathF.Sqrt(qx * qx + (qy - bounce) * (qy - bounce) + qz * qz) - 1.0f;
                    float torusR = 2.0f;
                    float torusr = 0.4f;
                    float torusQxz = MathF.Sqrt(qx * qx + qz * qz) - torusR;
                    float torusY = qy - MathF.Sin(time * 0.8f) * 0.5f;
                    float dTorus = MathF.Sqrt(torusQxz * torusQxz + torusY * torusY) - torusr;
                    float dGround = qy + 1.5f;
                    d = dSphere < dTorus ? dSphere : dTorus;
                    hitObj = d == dTorus ? 2 : 1;
                    float dWithGround = d < dGround ? d : dGround;
                    if (dWithGround == dGround) hitObj = 0;
                    d = dWithGround;
                }

                if (d < 0.002f)
                {
                    hitT = t;
                    steps = i;
                    break;
                }

                t = t + d;
                if (t > 30.0f) break;
            }

            // Color
            float oR, oG, oB;
            float bgR = 0.03f + 0.05f * (v * 0.5f + 0.5f);
            float bgG = 0.03f + 0.07f * (v * 0.5f + 0.5f);
            float bgB = 0.06f + 0.12f * (v * 0.5f + 0.5f);

            if (hitT > 0.0f)
            {
                float hx = camX + hitT * rdx;
                float hy = camY + hitT * rdy;
                float hz = camZ + hitT * rdz;

                // Normal via central differences
                float eps = 0.003f;
                float groundY = sceneType == 1 || sceneType == 3 ? -2.0f : -1.5f;
                float sxp = MathF.Sqrt((hx + eps) * (hx + eps) + hy * hy + hz * hz) - 1.2f;
                float gxp = hy - groundY; sxp = sxp < gxp ? sxp : gxp;
                float sxn = MathF.Sqrt((hx - eps) * (hx - eps) + hy * hy + hz * hz) - 1.2f;
                float gxn = hy - groundY; sxn = sxn < gxn ? sxn : gxn;
                float syp = MathF.Sqrt(hx * hx + (hy + eps) * (hy + eps) + hz * hz) - 1.2f;
                float gyp = (hy + eps) - groundY; syp = syp < gyp ? syp : gyp;
                float syn = MathF.Sqrt(hx * hx + (hy - eps) * (hy - eps) + hz * hz) - 1.2f;
                float gyn = (hy - eps) - groundY; syn = syn < gyn ? syn : gyn;
                float szp = MathF.Sqrt(hx * hx + hy * hy + (hz + eps) * (hz + eps)) - 1.2f;
                float gzp = hy - groundY; szp = szp < gzp ? szp : gzp;
                float szn = MathF.Sqrt(hx * hx + hy * hy + (hz - eps) * (hz - eps)) - 1.2f;
                float gzn = hy - groundY; szn = szn < gzn ? szn : gzn;

                float nx = sxp - sxn;
                float ny = syp - syn;
                float nz = szp - szn;
                float nLen = MathF.Sqrt(nx * nx + ny * ny + nz * nz);
                if (nLen > 0.0001f) { nx = nx / nLen; ny = ny / nLen; nz = nz / nLen; }

                float diff = nx * 0.577f + ny * 0.577f - nz * 0.577f;
                if (diff < 0.0f) diff = 0.0f;

                float mR, mG, mB;
                if (hitObj == 0)
                {
                    float chk = MathF.Floor(hx) + MathF.Floor(hz);
                    chk = chk - 2.0f * MathF.Floor(chk * 0.5f);
                    if (chk < 0.0f) chk = -chk;
                    float c = chk > 0.5f ? 0.3f : 0.15f;
                    mR = c; mG = c; mB = c;
                }
                else if (sceneType == 0)
                {
                    if (hitObj == 2) { mR = 0.9f; mG = 0.7f; mB = 0.2f; }
                    else { mR = 0.6f; mG = 0.3f; mB = 0.8f; }
                }
                else if (sceneType == 1)
                {
                    if (hitObj == 2) { mR = 1.0f; mG = 0.5f; mB = 0.1f; }
                    else { mR = 0.2f; mG = 0.7f; mB = 0.8f; }
                }
                else if (sceneType == 2)
                {
                    float hue = MathF.Floor(hx * 0.5f) + MathF.Floor(hz * 0.5f);
                    hue = hue - MathF.Floor(hue / 3.0f) * 3.0f;
                    if (hue < 0.0f) hue = hue + 3.0f;
                    if (hue < 1.0f) { mR = 0.2f + 0.6f * hue; mG = 0.5f; mB = 0.9f - 0.4f * hue; }
                    else if (hue < 2.0f) { mR = 0.8f; mG = 0.3f + 0.5f * (hue - 1.0f); mB = 0.3f; }
                    else { mR = 0.3f; mG = 0.8f; mB = 0.3f + 0.4f * (hue - 2.0f); }
                }
                else
                {
                    mR = 0.9f; mG = 0.4f; mB = 0.5f;
                }

                float ao = 1.0f - (float)steps / (float)maxSteps;
                oR = mR * (0.15f + diff * 0.85f) * ao;
                oG = mG * (0.15f + diff * 0.85f) * ao;
                oB = mB * (0.15f + diff * 0.85f) * ao;
            }
            else
            {
                oR = bgR; oG = bgG; oB = bgB;
            }

            oR = MathF.Sqrt(oR);
            oG = MathF.Sqrt(oG);
            oB = MathF.Sqrt(oB);

            int cr = (int)(oR * 255.0f); if (cr > 255) cr = 255; if (cr < 0) cr = 0;
            int cg = (int)(oG * 255.0f); if (cg > 255) cg = 255; if (cg < 0) cg = 0;
            int cb = (int)(oB * 255.0f); if (cb > 255) cb = 255; if (cb < 0) cb = 0;
            output[index] = (uint)(cr | (cg << 8) | (cb << 16) | (255 << 24));
        }
    }
}
