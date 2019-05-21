#if !defined(CVX_EXP_H)
#define CVX_EXP_H

#include "cvx.h"
namespace cvx {

template<int UNIT, class T, class TD, class FUNC>
inline void for_ps_pd_idx_old(const T* ps, TD* pd, int n, FUNC func)
{
  int nUNIT = n / UNIT * UNIT;
  int i = 0;
  const T* pst = ps;
  TD* pdt = pd;
  for(; i < nUNIT; i += UNIT) {
    for(int j = 0; j < UNIT; j++) {
      func(pst, pdt, j);
    }
    pst += UNIT;
    pdt += UNIT;
  }
  int j = 0;
  for(; i < n; i++) {
    func(pst, pdt, j);
    j++;
  }
}


template<int UNIT, class T, class TD, class FUNC>
inline void for_ps_pd_idx(const T* ps, TD* pd, int n, FUNC func)
{
  const int nUNIT = n / UNIT * UNIT;
  int i = 0;
  const T* pst = ps;
  TD* pdt = pd;
  for(; i < nUNIT; i += UNIT) {
    for(int j = 0; j < UNIT; j++) {
      func(pst, pdt, j);
    }
    pst += UNIT;
    pdt += UNIT;
  }
  if constexpr(UNIT > 8) {
    const int n8 = n / 8 * 8;
    for(; i < n8; i += 8) {
      for(int j = 0; j < 8; j++) {
        func(pst, pdt, j);
      }
      pst += 8;
      pdt += 8;
    }
  }
  int j = 0;
  for(; i < n; i++) {
    func(pst, pdt, j);
    j++;
  }
}
template<int UNIT, class T, class TD, class FUNC>
inline void for_ps_pd_r_old(const T* ps, TD* pd, int n, FUNC func)
{
  const int nUNIT = n / UNIT * UNIT;
  int i = 0;
  const T* pst = ps;
  TD* pdt = pd;
  for(; i < nUNIT; i += UNIT) {
    for(int j = 0; j < UNIT; j++) {
      pdt[j] = func(pst[j]);
    }
    pst += UNIT;
    pdt += UNIT;
  }
  int j = 0;
  for(; i < n; i++) {
    pdt[j] = func(pst[j]);
    j++;
  }
}



template<int UNIT, class TD, class FUNC>
inline void for_p_idx(int n, TD* p, FUNC func)
{
  int nn = n / UNIT * UNIT;
  int i = 0;
  TD* pdt = p;
  for(; i < nn; i += UNIT) {
    for(int j = 0; j < UNIT; j++) {
      func(pdt, j);
    }
    pdt += UNIT;
  }
  int j = 0;
  for(; i < n; i++) {
    func(pdt, j);
    j++;
  }
}


template<class MAT>
void fill_test_value(MAT& m)
{
  int i = 0;
  for(int y = 0; y < m.rows; y++) {
    for(int x = 0; x < m.cols; x++) {
      m(y, x) = i % 255;
      i++;
    }
  }
}

template<class MAT, class T>
void fill_all(MAT& m, T v)
{
#pragma omp parallel for
  for(int y = 0; y < m.rows; y++) {
    for(int x = 0; x < m.cols; x++) {
      m(y, x) = v;
    }
  }
}

template<class T>
bool check_value(const Mtx_<T>& src, const Mtx_<T>& ref, int border = 0)
{
  for(int y = -border; y < src.rows + border; y++) {
    auto ps = src.createPixPtr(y, 0);
    auto pr = ref.createPixPtr(y, 0);
    for(int x = -border; x < src.cols + border; x++) {
      if(pr(0, 0) != ps(0, 0)) goto ERROR_EIXT;
      ps++;
      pr++;
    }
  }
  return true;
ERROR_EIXT:
  std::cout << std::endl;
  print(ref);
  print(src);
  std::cout << "failed." << std::endl;
  return false;
}

template<class T>
void print(const Mtx_<T>& m)
{
  int vborder = m.vertBorder();
  int hborder = m.horzBorder();

  for(int y = -vborder; y < m.rows + vborder; y++) {
    for(int x = -hborder; x < m.cols + hborder; x++) {
      std::cout << (int)m(y, x) << " ";
    }
    std::cout << std::endl;
  }
}

template<class VECD, class FUNC, class VEC>
VECD conv(const VEC& s, FUNC conv)
{
  constexpr int ch = VEC::channels < VECD::channels ? VEC::channels : VECD::channels;
  VECD ret;
  for(int i = 0; i < ch; i++) {
    ret[i] = conv(s[i]);
  }
  return ret;
}

template<class VECD, class FUNC, class VEC>
void conv(const VEC& s, VECD& d, FUNC conv)
{
  constexpr int ch = VEC::channels < VECD::channels ? VEC::channels : VECD::channels;
  for(int i = 0; i < ch; i++) {
    conv(s[i], d[i]);
  }
}

template<class VEC, class FUNC>
VEC conv(const VEC& s, FUNC conv)
{
  constexpr int ch = VEC::channels;
  VEC ret;
  for(int i = 0; i < ch; i++) {
    ret[i] = conv(s[i]);
  }
  return ret;
}

template<class VECD, class FUNC, class VEC>
VECD convIdx(const VEC& s, FUNC conv)
{
  constexpr int ch = VEC::channels < VECD::channels ? VEC::channels : VECD::channels;
  VECD ret;
  for(int i = 0; i < ch; i++) {
    ret[i] = conv(s[i], i);
  }
  return ret;
}

template<class VEC, class FUNC>
VEC convIdx(const VEC& s, FUNC conv)
{
  constexpr int ch = VEC::channels;
  VEC ret;
  for(int i = 0; i < ch; i++) {
    ret[i] = conv(s[i], i);
  }
  return ret;
}

}  //namespace cvx

#endif  //#if !defined(CVX_EXP_H)