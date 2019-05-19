#if !defined(CVX_H)
#define CVX_H

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <chrono>
#include <thread>
#include <mutex>
#include <valarray>

namespace cvx {

struct Offset
{
  Offset(size_t _offset)
      : offset(_offset)
  {
  }
  //誤った利用を抑制するためにあえて戻り値はvoidにする
  void operator++() { ++offset; }
  void operator++(int) { offset++; }
  void operator+=(size_t v) { offset += v; }
  void operator-=(size_t v) { offset -= v; }

  Offset operator+(size_t v) { return Offset(offset + v); }
  Offset operator-(size_t v) { return Offset(offset - v); }

  size_t offset;
};

//二次元アクセス可能な画素値用ポインタークラス
template<class _T>
struct PixPtr
{
public:
	using T = std::remove_cv_t<_T>;

  PixPtr(T* ptr, size_t step)
      : ptr_(ptr)
      , step_(step)
  {
  }

	const T* ptr() const { return ptr_; }
  T* ptr() { return ptr_; }

  inline T* ptr(int dy, int dx) { return ptr_ + step_ * dy + dx; }
  inline const T* ptr(int dy, int dx) const { return ptr_ + step_ * dy + dx; }

  inline T& operator()(int dy, int dx) { return *ptr(dy, dx); }
  inline const T& operator()(int dy, int dx) const { return *ptr(dy, dx); }

  void operator++() { ++ptr_; }
  void operator++(int) { ptr_++; }
  void operator+=(size_t v) { ptr_ += v; }
  void operator-=(size_t v) { ptr_ -= v; }

  void moveLine(int dy) { ptr += dy * step_; }

  PixPtr copyPixPtr(int dy, int dx) { return PixPtr(ptr + dy * step_ + dx, step); }

  const PixPtr copyPixPtr(int dy, int dx) const { return PixPtr(ptr + dy * step_ + dx, step); }

private:
  size_t step_;
  T* ptr_;
};

template<class _T>
class Mtx_ : public cv::Mat_<_T>
{
public:
  using T = std::remove_cv_t<_T>; 
  using cv::Mat_<T>::Mat_;
  using cv::Mat_<T>::operator();  //overloadしているので必要
  using cv::Mat_<T>::stepT;
  using cv::Mat_<T>::rows;
  using cv::Mat_<T>::cols;

  Offset calcOffset(int y, int x) const { return Offset(stepT(0) * y + x); }
  Offset calcOffset(const PixPtr<const T>& pp) const {
		return Offset(dataT() - pp.ptr(0,0)); 
	}

  T& operator()(const Offset& offset) { return dataT()[offset.offset]; }

  const T& operator()(const Offset& offset) const { return dataT()[offset.offset]; }

  const T& operator()(int y, int x) const { return dataT()[y * stepT() + x]; }

  T& operator()(int y, int x) { return dataT()[y * stepT() + x]; }

  T* operator[](int y) { return dataT() + y * stepT(); }
  const T* operator[](int y) const { return dataT() + y * stepT(); }

  T operator()(float y, float x)
  {
    float fx0 = std::floor(x);
    float fy0 = std::floor(y);
    float fx1 = fx0 + 1.f;
    float fy1 = fy0 + 1.f;

    int hb = horzBorder();
    int vb = vertBorder();

    int x0 = static_cast<int>(fx0);
    if(x0 < -hb) x0 = -hb;
    if(x0 > cols + hb - 2) x0 = cols + hb - 2;

    int y0 = static_cast<int>(fy0);
    if(y0 < -vb) y0 = -vb;
    if(y0 > rows + vb - 2) y0 = rows + vb - 2;

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    auto p = (*this)[y0] + x0;
    T v00 = p[0];
    T v10 = p[stepT()];
    T v01 = p[1];
    T v11 = p[stepT() + 1];

    float v0 = v00 * (fx1 - x) + v01 * (x - fx0);
    float v1 = v10 * (fx1 - x) + v11 * (x - fx0);
    return static_cast<T>(v0 * (fy1 - y) + v1 * (y - fy0));
  }

  T operator()(cv::Point2f pt) { return (*this)(pt.y, pt.x); }

  static Mtx_<T> createWithBorder(int _rows, int _cols, int vborder, int hborder = -1)
  {
    if(hborder < 0) hborder = vborder;
    cv::Mat_<T> mat(_rows + vborder * 2, _cols + hborder * 2);
    Mtx_<T> mtx2 = mat(cv::Rect(hborder, vborder, _cols, _rows));
    return mtx2;
  }
	
  static Mtx_<T> createWithBorder(cv::Size sz, int vborder, int hborder = -1)
  {
    if(hborder < 0) hborder = vborder;
    return createWithBorder(sz.height, sz.width, vborder, hborder);
  }
  int horzBorder() const
  {
    cv::Size sz;
    cv::Point pt;
    this->locateROI(sz, pt);
    return pt.x;
  };
  int vertBorder() const
  {
    cv::Size sz;
    cv::Point pt;
    this->locateROI(sz, pt);
    return pt.y;
  };

  void extrapolate()
  {
    const auto vborder = vertBorder();
    const auto hborder = horzBorder();
#if 0
#else
    //Left
    for(int y = 0; y < rows; y++) {
      auto pix = (*this)(y, 0);
      auto offset = calcOffset(y, -hborder);
      for(int dx = 0; dx < hborder; dx++) {
        (*this)(offset) = pix;
        offset++;
      }
    }
    //Right
    for(int y = 0; y < rows; y++) {
      auto pix = (*this)(y, cols - 1);
      auto offset = calcOffset(y, cols);
      for(int dx = 0; dx < hborder; dx++) {
        (*this)(offset) = pix;
        offset++;
      }
    }
    //Top
#if 1
    for(int y = -1; y >= -vborder; y--) {
      std::copy_n(&datacT()[-hborder], stepT(), &dataT()[-hborder + y * stepT()]);
    }
#else
    for(int y = 0; y < vborder; y++) {
      auto offset_s = calcOffset(0, -hborder);
      auto offset_d = calcOffset(-y - 1, -hborder);
      for(int dx = 0; dx < this->cols + hborder * 2; dx++) {
        (*this)(offset_d) = (*this)(offset_s);
        offset_d++;
        offset_s++;
      }
    }
#endif
    //Bottom
#if 1
    for(int y = rows; y < rows + vborder; y++) {
      std::copy_n(&datacT()[-hborder + (rows - 1) * stepT()], stepT(),
                  &dataT()[-hborder + y * stepT()]);
    }
#else
    for(int y = 0; y < vborder; y++) {
      auto offset_s = calcOffset(rows - 1, -hborder);
      auto offset_d = calcOffset(rows, -hborder);
      for(int dx = 0; dx < cols + hborder * 2; dx++) {
        (*this)(offset_d) = (*this)(offset_s);
        offset_d++;
        offset_s++;
      }
    }
#endif
#endif
  }
  const PixPtr<T> createPixPtr(Offset offset) const
  {
    return PixPtr<T>(dataT() + offset.offset);
  }
 
  const PixPtr<T> createPixPtr(int y, int x) const
  {
    return PixPtr<T>(const_cast<T*>(dataT()) + y * stepT() + x, stepT());
  }
  PixPtr<T> createPixPtr(int y, int x) { return PixPtr<T>(dataT() + y * stepT() + x, stepT()); }

private:
  const T* dataT() const { return reinterpret_cast<const T*>(this->data); }
  const T* datacT() const { return reinterpret_cast<const T*>(this->data); }
  T* dataT() { return reinterpret_cast<T*>(this->data); }
};

using Mtx1b = Mtx_<uchar>;
using Mtx2b = Mtx_<cv::Vec2b>;
using Mtx3b = Mtx_<cv::Vec3b>;
using Mtx4b = Mtx_<cv::Vec4b>;
using Mtx1i = Mtx_<int>;
using Mtx1w = Mtx_<unsigned short>;
using Mtx1f = Mtx_<float>;
using Mtx2f = Mtx_<cv::Vec2f>;
using Mtx3f = Mtx_<cv::Vec3f>;
using Mtx4f = Mtx_<cv::Vec4f>;

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



template<class T, int RSZ>
struct Kernel
{
  static constexpr int SZ = RSZ * 2 + 1;
  inline T operator()(int dy, int dx) { return data[SZ * (dy + RSZ) + (dx + RSZ)]; }
  inline T operator()(int dy, int dx) const { return data[SZ * (dy + RSZ) + (dx + RSZ)]; }

  void copy(T* p) { std::memcpy(data, p, SZ * SZ * sizeof(T)); }
  T data[SZ * SZ];
  static constexpr int radius = RSZ;

  template<class FUNC>
  inline void for_(FUNC func)
  {
    for_dydx(radius, radius, [&](int dy, int dx) { func((*this)(dy, dx), dy, dx); });
  }

  template<class TP>
  T convolute(const PixPtr<TP>& pp) const
  {
    T sum = 0;
    for_dydx(radius, radius, [&](int dy, int dx) { sum += (*this)(dy, dx) * pp(dy, dx); });
    return sum;
  }
  Kernel() = default;

  template<class... A>
  Kernel(A... args)
  {
    int j = 0;
    for(int i : std::initializer_list<int>{args...}) {
      data[j] = i;
      //      if(j == ) break;
      j++;
    }
  }
};

template<class T, int RSZ>
struct SepKernel
{
  T operator()(int d) { return data[std::abs(d)]; }
  void copy(T* p) { std::memcpy(data, p, (RSZ + 1) * sizeof(T)); }
  T data[RSZ + 1];
  static constexpr int radius = RSZ;

  template<class TP>
  T convolute(const PixPtr<TP>& pp, bool horz)
  {
    //   std::vector<T> lines((2 * RSZ + 1) * pp.step_);
    T sum = data[0] * pp(0, 0);
    if(horz) {
      for(int d = 1; d <= RSZ; d++) {
        sum += data[d] * (pp(0, d) + pp(0, -d));
      }
    } else {
      for(int d = 1; d <= RSZ; d++) {
        sum += data[d] * (pp(d, 0) + pp(-d, 0));
      }
    }
    return sum;
  }
  template<class... A>
  SepKernel(A... args)
  {
    int j = 0;
    for(T i : std::initializer_list<int>{args...}) {
      data[j] = i;
      if(j == RSZ) break;
      j++;
    }
    j++;
    for(; j <= RSZ; j++) {
      data[j] = 0;
    }
  }
};
#if 1
template<int UNIT, class T, class TD, class FUNC>
inline void transform(const T* ps, TD* pd, int n, FUNC func)
{
  const int nUNIT = n / UNIT * UNIT;
  const int n8 = n / 8 * 8;
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
  for(; i < n8; i += 8) {
    for(int j = 0; j < 8; j++) {
      pdt[j] = func(pst[j]);
    }
    pst += 8;
    pdt += 8;
  }
  int j = 0;
  for(; i < n; i++) {
    pdt[j] = func(pst[j]);
    j++;
  }
}
#else

template<int UNIT, class T, class TD>
inline void transform(const T* ps, TD* pd, int n, std::function<TD(const T&)>)
{
  const int nUNIT = n / UNIT * UNIT;
  const int n8 = n / 8 * 8;
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
  for(; i < n8; i += 8) {
    for(int j = 0; j < 8; j++) {
      pdt[j] = func(pst[j]);
    }
    pst += 8;
    pdt += 8;
  }
  int j = 0;
  for(; i < n; i++) {
    pdt[j] = func(pst[j]);
    j++;
  }
}
#endif

//FUNC : TD(const PixPtr<T>&)
template<class T, class FUNC, class TD = T>
inline void paral_for_pp_r(const Mtx_<T>& src, Mtx_<TD>& dst,
                           FUNC func)
{
#pragma omp parallel for
  for(int y = 0; y < src.rows; y++) {
    auto ps = src.createPixPtr(y, 0);
    auto pd = dst.createPixPtr(y, 0);
    for(int x = 0; x < src.cols; x++) {
      pd(0, 0) = func(ps);
      ps++;
      pd++;
    }
  }
}

template<class T, class FUNC>
inline void paral_for_offset(const Mtx_<T>& src, FUNC func)
{
#pragma omp parallel for
  for(int y = 0; y < src.rows; y++) {
    auto offset = src.calcOffset(y, 0);
    for(int x = 0; x < src.cols; x++) {
      func(offset);
      offset++;
    }
  }
}
template<class T, class FUNC>
inline void paral_for_pp_offset(const Mtx_<T>& src, FUNC func)
{
#pragma omp parallel for
  for(int y = 0; y < src.rows; y++) {
    auto offset = src.calcOffset(y, 0);
    auto pp = src.createPixPtr(offset);
    for(int x = 0; x < src.cols; x++) {
      func(pp,offset);
      offset++;
    }
  }
}


}  //namespace cvx

#endif  //#if !defined(CVX_H)