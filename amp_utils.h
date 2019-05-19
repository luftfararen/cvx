#if !defined(AMP_UTILS_H)
#define AMP_UTILS_H
#include <amp.h>

namespace Concurrency {

template<class FUNC>
void for_dydx(int vr, int hr, FUNC func) __GPU
{
  for(int dy = -vr; dy <= vr; dy++) {
    for(int dx = -hr; dx <= hr; dx++) {
      func(dy, dx);
    }
  }
}

template<class T>
T clamp(T v, T l, T u) __GPU
{
  return v > u ? u : (v < l ? l : v);
}

graphics::float_2 coord(const index<2>& idx, const extent<2>& ext) restrict(amp)

{
  return graphics::float_2((idx[1] + 0.5f) / (float)ext[1], (idx[0] + 0.5f) / (float)ext[0]);
}

graphics::float_2 coord(float x, float y, const extent<2>& ext) restrict(amp)
{
  return graphics::float_2((y+ 0.5f) / (float)ext[1], (x + 0.5f) / (float)ext[0]);
}

}  // namespace Concurrency

#endif  //#if !defined(AMP_UTILS_H)