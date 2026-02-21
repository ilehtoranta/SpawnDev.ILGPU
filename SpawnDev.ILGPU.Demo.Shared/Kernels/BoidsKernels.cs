using ILGPU;
using ILGPU.Runtime;

namespace SpawnDev.ILGPU.Demo.Shared.Kernels
{
    /// <summary>
    /// GPU Boids flocking simulation + 3D rendering kernels.
    /// Two-pass: Simulate → Render.
    /// </summary>
    public static class BoidsKernels
    {
        /// <summary>
        /// Simulation kernel — O(N²) neighbor scan for flocking behavior.
        /// Each thread processes one boid.
        /// Buffer layout: [i*6+0..5] = posX, posY, posZ, velX, velY, velZ
        /// </summary>
        public static void Simulate(
            Index1D index,
            ArrayView1D<float, Stride1D.Dense> boidsIn,
            ArrayView1D<float, Stride1D.Dense> boidsOut,
            int count, int speciesCount,
            float separation, float alignment, float cohesion, float speedDt)
        {
            int i = index;
            if (i >= count) return;

            int stride = 6;
            int bi = i * stride;

            float px = boidsIn[bi + 0];
            float py = boidsIn[bi + 1];
            float pz = boidsIn[bi + 2];
            float vx = boidsIn[bi + 3];
            float vy = boidsIn[bi + 4];
            float vz = boidsIn[bi + 5];

            int mySpecies = i % speciesCount;

            float sepX = 0, sepY = 0, sepZ = 0;
            float alignX = 0, alignY = 0, alignZ = 0;
            float cohX = 0, cohY = 0, cohZ = 0;
            int neighbors = 0;

            float viewRadius = 3.0f;
            float viewRadSq = viewRadius * viewRadius;
            float sepRadius = 1.0f;
            float sepRadSq = sepRadius * sepRadius;

            int step = count > 2000 ? count / 500 : 1;
            if (step < 1) step = 1;

            for (int j = 0; j < count; j += step)
            {
                if (j == i) continue;
                int bj = j * stride;
                float dx = boidsIn[bj + 0] - px;
                float dy = boidsIn[bj + 1] - py;
                float dz = boidsIn[bj + 2] - pz;
                float distSq = dx * dx + dy * dy + dz * dz;

                if (distSq < viewRadSq && distSq > 0.0001f)
                {
                    int otherSpecies = j % speciesCount;
                    float sameSpecies = otherSpecies == mySpecies ? 1.0f : 0.3f;

                    if (distSq < sepRadSq)
                    {
                        float inv = 1.0f / (distSq + 0.01f);
                        sepX -= dx * inv;
                        sepY -= dy * inv;
                        sepZ -= dz * inv;
                    }

                    alignX += boidsIn[bj + 3] * sameSpecies;
                    alignY += boidsIn[bj + 4] * sameSpecies;
                    alignZ += boidsIn[bj + 5] * sameSpecies;

                    cohX += dx * sameSpecies;
                    cohY += dy * sameSpecies;
                    cohZ += dz * sameSpecies;

                    neighbors++;
                }
            }

            if (neighbors > 0)
            {
                float invN = 1.0f / (float)neighbors;
                vx += sepX * separation * 0.5f + alignX * invN * alignment * 0.1f + cohX * invN * cohesion * 0.05f;
                vy += sepY * separation * 0.5f + alignY * invN * alignment * 0.1f + cohY * invN * cohesion * 0.05f;
                vz += sepZ * separation * 0.5f + alignZ * invN * alignment * 0.1f + cohZ * invN * cohesion * 0.05f;
            }

            // Boundary containment
            float boundary = 12.0f;
            if (px > boundary) vx -= (px - boundary) * 0.3f;
            if (px < -boundary) vx -= (px + boundary) * 0.3f;
            if (py > boundary) vy -= (py - boundary) * 0.3f;
            if (py < -boundary) vy -= (py + boundary) * 0.3f;
            if (pz > boundary) vz -= (pz - boundary) * 0.3f;
            if (pz < -boundary) vz -= (pz + boundary) * 0.3f;

            // Limit speed
            float speed = MathF.Sqrt(vx * vx + vy * vy + vz * vz);
            float maxSpeed = 4.0f;
            if (speed > maxSpeed)
            {
                float s = maxSpeed / speed;
                vx *= s; vy *= s; vz *= s;
            }

            px += vx * speedDt;
            py += vy * speedDt;
            pz += vz * speedDt;

            boidsOut[bi + 0] = px;
            boidsOut[bi + 1] = py;
            boidsOut[bi + 2] = pz;
            boidsOut[bi + 3] = vx;
            boidsOut[bi + 4] = vy;
            boidsOut[bi + 5] = vz;
        }

        /// <summary>
        /// Render kernel — projects boids to screen with depth-based size and species coloring.
        /// </summary>
        public static void Render(
            Index2D index,
            ArrayView2D<uint, Stride2D.DenseX> output,
            ArrayView1D<float, Stride1D.Dense> boids,
            int boidCount, int speciesCount, int packedSize,
            float camTheta, float camPhi, float camDist,
            float unused1, float unused2)
        {
            int width = packedSize / 65536;
            int height = packedSize - width * 65536;
            int px = index.X;
            int py = index.Y;
            if (px >= width || py >= height) return;

            float sinPhi = MathF.Sin(camPhi);
            float cosPhi = MathF.Cos(camPhi);
            float sinTheta = MathF.Sin(camTheta);
            float cosTheta = MathF.Cos(camTheta);

            float camX = camDist * sinPhi * cosTheta;
            float camY = camDist * cosPhi;
            float camZ = camDist * sinPhi * sinTheta;

            float fwdX = -camX, fwdY = -camY, fwdZ = -camZ;
            float fwdLen = MathF.Sqrt(fwdX * fwdX + fwdY * fwdY + fwdZ * fwdZ);
            fwdX /= fwdLen; fwdY /= fwdLen; fwdZ /= fwdLen;

            float rightX = fwdZ, rightZ = -fwdX;
            float rightLen = MathF.Sqrt(rightX * rightX + rightZ * rightZ);
            if (rightLen < 0.001f) { rightX = 1; rightZ = 0; rightLen = 1; }
            rightX /= rightLen; rightZ /= rightLen;

            float upX = -fwdY * fwdX;
            float upY = fwdX * fwdX + fwdZ * fwdZ;
            float upZ = -fwdY * fwdZ;
            float upLen = MathF.Sqrt(upX * upX + upY * upY + upZ * upZ);
            if (upLen > 0.001f) { upX /= upLen; upY /= upLen; upZ /= upLen; }

            float bgV = (float)py / (float)height;
            float bgR = 0.02f + 0.04f * (1.0f - bgV);
            float bgG = 0.02f + 0.06f * (1.0f - bgV);
            float bgB = 0.05f + 0.1f * (1.0f - bgV);

            float finalR = bgR;
            float finalG = bgG;
            float finalB = bgB;

            float aspect = (float)width / (float)height;
            float fov = 1.2f;

            float screenU = (2.0f * px / width - 1.0f) * aspect;
            float screenV = 1.0f - 2.0f * py / height;

            int stride = 6;
            float closestDepth = 999.0f;

            for (int b = 0; b < boidCount; b++)
            {
                int bi = b * stride;
                float bx = boids[bi + 0] - camX;
                float by = boids[bi + 1] - camY;
                float bz = boids[bi + 2] - camZ;

                float viewZ = bx * fwdX + by * fwdY + bz * fwdZ;
                if (viewZ < 0.5f) continue;

                float viewX = bx * rightX + by * 0 + bz * rightZ;
                float viewY = bx * upX + by * upY + bz * upZ;

                float projX = viewX * fov / viewZ;
                float projY = viewY * fov / viewZ;

                float ddx = screenU - projX;
                float ddy = screenV - projY;
                float pixelDist = ddx * ddx + ddy * ddy;

                float dotSize = 0.012f / (viewZ * 0.08f);
                float dotSizeSq = dotSize * dotSize;

                if (pixelDist < dotSizeSq && viewZ < closestDepth)
                {
                    closestDepth = viewZ;

                    int species = b % speciesCount;
                    float intensity = 1.0f - pixelDist / dotSizeSq;
                    intensity = intensity * intensity;
                    float depthFade = 1.0f / (1.0f + viewZ * 0.03f);

                    float bvx = boids[bi + 3];
                    float bvy = boids[bi + 4];
                    float bvz = boids[bi + 5];
                    float speed = MathF.Sqrt(bvx * bvx + bvy * bvy + bvz * bvz);
                    float speedGlow = 0.7f + 0.3f * speed / 4.0f;
                    if (speedGlow > 1.0f) speedGlow = 1.0f;

                    float br = 0, bg = 0, bb = 0;
                    if (species == 0) { br = 0.2f; bg = 0.6f; bb = 1.0f; }
                    else if (species == 1) { br = 1.0f; bg = 0.4f; bb = 0.1f; }
                    else if (species == 2) { br = 0.3f; bg = 1.0f; bb = 0.4f; }
                    else if (species == 3) { br = 1.0f; bg = 0.2f; bb = 0.6f; }
                    else { br = 1.0f; bg = 0.9f; bb = 0.2f; }

                    float glow = intensity * depthFade * speedGlow;
                    finalR = br * glow + bgR * (1.0f - glow);
                    finalG = bg * glow + bgG * (1.0f - glow);
                    finalB = bb * glow + bgB * (1.0f - glow);
                }
            }

            finalR = MathF.Sqrt(finalR);
            finalG = MathF.Sqrt(finalG);
            finalB = MathF.Sqrt(finalB);

            int cr = (int)(finalR * 255.0f); if (cr > 255) cr = 255; if (cr < 0) cr = 0;
            int cg = (int)(finalG * 255.0f); if (cg > 255) cg = 255; if (cg < 0) cg = 0;
            int cb = (int)(finalB * 255.0f); if (cb > 255) cb = 255; if (cb < 0) cb = 0;
            output[index] = (uint)(cr | (cg << 8) | (cb << 16) | (0xFF << 24));
        }
    }
}
