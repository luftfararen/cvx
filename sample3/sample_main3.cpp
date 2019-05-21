#include <chrono>
#include <thread>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <array>
#include <random>

template<class INTEGRAL_TYPE>
constexpr int fast_iround(INTEGRAL_TYPE v)
{
  return static_cast<int>(v);
}

constexpr int fast_iround(float v)
{
  if(v >= 0) {
    return static_cast<int>(v + 0.5f);
  } else {
    return -static_cast<int>(-v + 0.5f);
  }
}

constexpr int fast_iround(double v)
{
  if(v >= 0) {
    return static_cast<int>(v + 0.5);
  } else {
    return -static_cast<int>(-v + 0.5);
  }
}

constexpr int fast_iround(long double v)
{
  if(v >= 0) {
    return static_cast<int>(v + 0.5l);
  } else {
    return -static_cast<int>(-v + 0.5l);
  }
}

//これが定義していあると、templateを使ったプログラミングに便利
template<class INTEGRAL_TYPE>
constexpr int iround(INTEGRAL_TYPE v)
{
  return static_cast<int>(v);
}

constexpr int iround(float _v)
{
  double v = _v;
  if(v >= 0) {
    return static_cast<int>(v + 0.5);
  } else {
    return -static_cast<int>(-v + 0.5);
  }
}

constexpr int iround(double _v)
{
  long double v = _v;
  if(v >= 0) {
    return static_cast<int>(v + 0.5l);
  } else {
    return -static_cast<int>(-v + 0.5l);
  }
}

inline int iround(long double v) { return static_cast<int>(std::llround(v)); }

//round関数がconstexprでないので、consteprにできない
inline int iround2(float v) { return static_cast<int>(std::round(v)); }
inline int iround2(double v) { return static_cast<int>(std::round(v)); }
inline int iround2(long double v) { return static_cast<int>(std::round(v)); }

//lround関数がconstexprでないので、consteprにできない
inline int iround3(float v) { return static_cast<int>(std::lround(v)); }
inline int iround3(double v) { return static_cast<int>(std::lround(v)); }
inline int iround3(long double v) { return static_cast<int>(std::lround(v)); }

inline int iround4(float _v)
{
#if 1
  double v = _v;
  if(v >= 0) {
    return static_cast<int>(v + 0.5);
  } else {
    return -static_cast<int>(-v + 0.5);
  }
#else
  if(abs(_v) >= 1 << 23) {
    if(_v >= 0) {
      return static_cast<int>(_v - 0.5) + 1;
    } else {
      return -(static_cast<int>(-_v - 0.5) + 1);
    }
  } else {
    if(_v >= 0) {
      return static_cast<int>(_v + 0.5);
    } else {
      return -static_cast<int>(-_v + 0.5);
    }
  }

#endif
}

inline int iround4(double v) { return static_cast<int>(std::lround(v)); }
inline int iround4(long double v) { return static_cast<int>(std::lround(v)); }

inline int iround6(float v) { return static_cast<int>(v + (v > 0 ? 0.5f : -0.5f)); }
inline int iround6(double v) { return static_cast<int>(v + (v > 0 ? 0.5 : -0.5)); }
inline int iround6(long double v) { return static_cast<int>(v + (v > 0 ? 0.5l : -0.5l)); }

constexpr int iround5(float v)
{
  if(v >= 0) {
    return static_cast<int>(2 * v + 1.f) / 2;
  } else {
    return -static_cast<int>(-2 * v + 1.f) / 2;
  }
}

constexpr int iround5(double v)
{
  if(v >= 0) {
    return static_cast<int>(2 * v + 1.0) / 2;
  } else {
    return -static_cast<int>(-2 * v + 1.0) / 2;
  }
}

constexpr int iround5(long double v)
{
  if(v >= 0) {
    return static_cast<int>(2 * v + 1.0l) / 2;
  } else {
    return -static_cast<int>(-2 * v + 1.0l) / 2;
  }
}

struct StopWatch
{
public:
  StopWatch() { pre_ = std::chrono::high_resolution_clock::now(); }

  ///returns lap time in ms.
  double lap()
  {
    auto tmp = std::chrono::high_resolution_clock::now();  //sotres time.
    auto dur = tmp - pre_;
    pre_ = tmp;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(dur).count() / 1000000.0;
  }

private:
  std::chrono::high_resolution_clock::time_point pre_;
};

template<size_t n>
double average(std::array<int, n> const& a)
{
  double sum = 0;
  for(auto e : a) {
    sum += e;
  }
  return sum / a.size();
}

template<class T>
void test()
{
  using namespace std;

  static constexpr int asize = 1024;  //キャッシュに収まるサイズ
  static constexpr int loop_count = asize * 1024;

  using dest_t = std::array<int, asize>;

  dest_t dest0;
  dest_t dest1;
  dest_t dest2;
  dest_t dest3;

  using src_t = std::array<T, asize>;
  src_t src;

  //実装した関数は、正負で条件分岐があるので投機的実行を抑制するために、
  //正負双方で均一に乱数を作る
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::uniform_real_distribution<T> dist(-10000.0, 10000.0);
  std::generate(src.begin(), src.end(), [&] { return dist(engine); });

  StopWatch sw;

  sw.lap();
  for(int z = 0; z < loop_count; z += 8) {
    auto pd = &dest0[z % asize];
    auto ps = &src[z % asize];
    for(int j = 0; j < 8; j++) {
      pd[j] = fast_iround(ps[j]);
    }
    pd += 8;
    ps += 8;
  }
  auto e = sw.lap();
  cout << "fast self implementation:" << e << " ms" << endl;

  sw.lap();
  for(int z = 0; z < loop_count; z += 8) {
    auto pd = &dest1[z % asize];
    auto ps = &src[z % asize];
    for(int j = 0; j < 8; j++) {
      pd[j] = iround(ps[j]);
    }
    pd += 8;
    ps += 8;
  }
  e = sw.lap();
  cout << "self implementation: " << e << " ms" << endl;

  sw.lap();
  for(int z = 0; z < loop_count; z += 8) {
    auto pd = &dest2[z % asize];
    auto ps = &src[z % asize];
    for(int j = 0; j < 8; j++) {
      pd[j] = iround2(ps[j]);
    }
    pd += 8;
    ps += 8;
  }
  e = sw.lap();
  cout << "round base: " << e << " ms" << endl;

  sw.lap();
  for(int z = 0; z < loop_count; z += 8) {
    auto pd = &dest3[z % asize];
    auto ps = &src[z % asize];
    for(int j = 0; j < 8; j++) {
      pd[j] = iround3(ps[j]);
    }
    pd += 8;
    ps += 8;
  }
  e = sw.lap();
  cout << "lround base: " << e << " ms" << endl;

  ;  //最適化でdestが消えないため
  cout << average(dest0) << endl;
  cout << average(dest1) << endl;
  cout << average(dest2) << endl;
  cout << average(dest3) << endl;
}

void main()
{
  using namespace std;
  std::cout << std::fixed << std::setprecision(2);
  cout << "float" << endl;
  test<float>();

  cout << endl << "double" << endl;
  test<double>();

  cout << endl << "long double" << endl;
  test<long double>();
}